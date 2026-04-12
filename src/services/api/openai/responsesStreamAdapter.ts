import type { BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import type {
  Response,
  ResponseCompletedEvent,
  ResponseFailedEvent,
  ResponseFunctionCallArgumentsDeltaEvent,
  ResponseFunctionCallArgumentsDoneEvent,
  ResponseIncompleteEvent,
  ResponseOutputItemDoneEvent,
  ResponseReasoningTextDeltaEvent,
  ResponseReasoningTextDoneEvent,
  ResponseRefusalDeltaEvent,
  ResponseRefusalDoneEvent,
  ResponseStreamEvent,
  ResponseTextDeltaEvent,
  ResponseTextDoneEvent,
} from 'openai/resources/responses/responses.mjs'

type UsageState = {
  input_tokens: number
  output_tokens: number
  cache_creation_input_tokens: number
  cache_read_input_tokens: number
}

export async function* adaptOpenAIResponsesStreamToAnthropic(
  stream: AsyncIterable<ResponseStreamEvent>,
  model: string,
): AsyncGenerator<BetaRawMessageStreamEvent, void> {
  const textBlocks = new Map<string, number>()
  const reasoningBlocks = new Map<string, number>()
  const toolBlocks = new Map<string, { blockIndex: number; closed: boolean }>()

  let messageId = ''
  let started = false
  let currentBlockIndex = -1
  let usage: UsageState = {
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  }
  let pendingStopReason: string | null = null
  let hasToolCalls = false

  for await (const event of stream) {
    switch (event.type) {
      case 'response.created': {
        started = true
        messageId = event.response.id
        usage = responseUsageToAnthropic(event.response)
        yield {
          type: 'message_start',
          message: {
            id: messageId,
            type: 'message',
            role: 'assistant',
            content: [],
            model,
            stop_reason: null,
            stop_sequence: null,
            usage,
          },
        } as unknown as BetaRawMessageStreamEvent
        break
      }

      case 'response.content_part.added': {
        if (!started) break
        if (event.part.type === 'output_text') {
          currentBlockIndex++
          textBlocks.set(
            makeContentKey(event.output_index, event.content_index),
            currentBlockIndex,
          )
          yield {
            type: 'content_block_start',
            index: currentBlockIndex,
            content_block: { type: 'text', text: '' },
          } as unknown as BetaRawMessageStreamEvent
        } else if (event.part.type === 'reasoning_text') {
          currentBlockIndex++
          reasoningBlocks.set(
            makeContentKey(event.output_index, event.content_index),
            currentBlockIndex,
          )
          yield {
            type: 'content_block_start',
            index: currentBlockIndex,
            content_block: { type: 'thinking', thinking: '', signature: '' },
          } as unknown as BetaRawMessageStreamEvent
        } else if (event.part.type === 'refusal') {
          currentBlockIndex++
          textBlocks.set(
            makeContentKey(event.output_index, event.content_index),
            currentBlockIndex,
          )
          yield {
            type: 'content_block_start',
            index: currentBlockIndex,
            content_block: { type: 'text', text: '' },
          } as unknown as BetaRawMessageStreamEvent
        }
        break
      }

      case 'response.output_item.added': {
        if (!started) break
        if (event.item.type === 'function_call') {
          hasToolCalls = true
          currentBlockIndex++
          toolBlocks.set(event.item.id || event.item.call_id, {
            blockIndex: currentBlockIndex,
            closed: false,
          })
          yield {
            type: 'content_block_start',
            index: currentBlockIndex,
            content_block: {
              type: 'tool_use',
              id: event.item.call_id,
              name: event.item.name,
              input: {},
            },
          } as unknown as BetaRawMessageStreamEvent
        }
        break
      }

      case 'response.output_text.delta': {
        const index = textBlocks.get(
          makeContentKey(event.output_index, event.content_index),
        )
        if (index === undefined) break
        yield {
          type: 'content_block_delta',
          index,
          delta: { type: 'text_delta', text: event.delta },
        } as unknown as BetaRawMessageStreamEvent
        break
      }

      case 'response.output_text.done': {
        const index = textBlocks.get(
          makeContentKey(event.output_index, event.content_index),
        )
        if (index === undefined) break
        yield {
          type: 'content_block_stop',
          index,
        } as unknown as BetaRawMessageStreamEvent
        break
      }

      case 'response.refusal.delta': {
        const index = textBlocks.get(
          makeContentKey(event.output_index, event.content_index),
        )
        if (index === undefined) break
        yield {
          type: 'content_block_delta',
          index,
          delta: { type: 'text_delta', text: event.delta },
        } as unknown as BetaRawMessageStreamEvent
        break
      }

      case 'response.refusal.done': {
        const index = textBlocks.get(
          makeContentKey(event.output_index, event.content_index),
        )
        if (index === undefined) break
        yield {
          type: 'content_block_stop',
          index,
        } as unknown as BetaRawMessageStreamEvent
        break
      }

      case 'response.reasoning_text.delta': {
        const index = reasoningBlocks.get(
          makeContentKey(event.output_index, event.content_index),
        )
        if (index === undefined) break
        yield {
          type: 'content_block_delta',
          index,
          delta: { type: 'thinking_delta', thinking: event.delta },
        } as unknown as BetaRawMessageStreamEvent
        break
      }

      case 'response.reasoning_text.done': {
        const index = reasoningBlocks.get(
          makeContentKey(event.output_index, event.content_index),
        )
        if (index === undefined) break
        yield {
          type: 'content_block_stop',
          index,
        } as unknown as BetaRawMessageStreamEvent
        break
      }

      case 'response.function_call_arguments.delta': {
        const toolBlock = toolBlocks.get(event.item_id)
        if (!toolBlock) break
        yield {
          type: 'content_block_delta',
          index: toolBlock.blockIndex,
          delta: {
            type: 'input_json_delta',
            partial_json: event.delta,
          },
        } as unknown as BetaRawMessageStreamEvent
        break
      }

      case 'response.function_call_arguments.done': {
        const toolBlock = toolBlocks.get(event.item_id)
        if (!toolBlock || toolBlock.closed) break
        toolBlock.closed = true
        yield {
          type: 'content_block_stop',
          index: toolBlock.blockIndex,
        } as unknown as BetaRawMessageStreamEvent
        break
      }

      case 'response.output_item.done': {
        if (event.item.type === 'function_call') {
          const toolBlock = toolBlocks.get(event.item.id || event.item.call_id)
          if (toolBlock && !toolBlock.closed) {
            toolBlock.closed = true
            yield {
              type: 'content_block_stop',
              index: toolBlock.blockIndex,
            } as unknown as BetaRawMessageStreamEvent
          }
        }
        break
      }

      case 'response.completed':
      case 'response.incomplete':
      case 'response.failed': {
        usage = responseUsageToAnthropic(event.response)
        pendingStopReason = deriveStopReason(event, hasToolCalls)
        break
      }

      default:
        break
    }
  }

  if (started) {
    yield {
      type: 'message_delta',
      delta: {
        stop_reason: pendingStopReason,
        stop_sequence: null,
      },
      usage,
    } as unknown as BetaRawMessageStreamEvent
    yield {
      type: 'message_stop',
    } as unknown as BetaRawMessageStreamEvent
  }
}

function makeContentKey(outputIndex: number, contentIndex: number): string {
  return `${outputIndex}:${contentIndex}`
}

function responseUsageToAnthropic(response: Response): UsageState {
  return {
    input_tokens: response.usage?.input_tokens ?? 0,
    output_tokens: response.usage?.output_tokens ?? 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens:
      response.usage?.input_tokens_details?.cached_tokens ?? 0,
  }
}

function deriveStopReason(
  event: ResponseCompletedEvent | ResponseIncompleteEvent | ResponseFailedEvent,
  hasToolCalls: boolean,
): 'end_turn' | 'tool_use' | 'max_tokens' | null {
  if (event.type === 'response.incomplete') {
    if (event.response.incomplete_details?.reason === 'max_output_tokens') {
      return 'max_tokens'
    }
    return 'end_turn'
  }

  if (event.type === 'response.failed') {
    return null
  }

  return hasToolCalls ? 'tool_use' : 'end_turn'
}
