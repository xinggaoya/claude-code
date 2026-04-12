import type {
  BetaToolResultBlockParam,
  BetaToolUseBlock,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import type { AssistantMessage, UserMessage } from '../../../types/message.js'
import type { SystemPrompt } from '../../../utils/systemPromptType.js'
import type {
  EasyInputMessage,
  ResponseFunctionToolCall,
  ResponseInputContent,
  ResponseInputImage,
  ResponseInputItem,
} from 'openai/resources/responses/responses.mjs'

export function systemPromptToOpenAIInstructions(
  systemPrompt: SystemPrompt,
): string | undefined {
  const text = systemPrompt.filter(Boolean).join('\n\n').trim()
  return text.length > 0 ? text : undefined
}

/**
 * Convert internal message history into Responses API input items.
 *
 * We preserve tool-call structure explicitly:
 * - assistant tool_use -> function_call items
 * - user tool_result   -> function_call_output items
 *
 * This is more faithful than flattening everything into plain messages and is
 * the key piece needed for multi-step agentic/tool workflows on Responses API.
 */
export function anthropicMessagesToOpenAIResponsesInput(
  messages: (UserMessage | AssistantMessage)[],
): ResponseInputItem[] {
  const result: ResponseInputItem[] = []

  for (const message of messages) {
    if (message.type === 'user') {
      result.push(...convertUserMessage(message))
    } else if (message.type === 'assistant') {
      result.push(...convertAssistantMessage(message))
    }
  }

  return result
}

function convertUserMessage(message: UserMessage): ResponseInputItem[] {
  const content = message.message.content

  if (typeof content === 'string') {
    if (content.length === 0) return []
    return [
      {
        type: 'message',
        role: 'user',
        content,
      } satisfies EasyInputMessage,
    ]
  }

  if (!Array.isArray(content)) {
    return []
  }

  const result: ResponseInputItem[] = []
  let pendingContent: ResponseInputContent[] = []

  const flushUserMessage = (): void => {
    if (pendingContent.length === 0) return
    result.push({
      type: 'message',
      role: 'user',
      content: pendingContent,
    })
    pendingContent = []
  }

  for (const block of content) {
    if (typeof block === 'string') {
      pendingContent.push({ type: 'input_text', text: block })
      continue
    }

    switch (block.type) {
      case 'text':
        pendingContent.push({ type: 'input_text', text: block.text })
        break
      case 'image': {
        const imagePart = convertImageBlock(
          block as unknown as Record<string, unknown>,
        )
        if (imagePart) pendingContent.push(imagePart)
        break
      }
      case 'tool_result':
        flushUserMessage()
        result.push(convertToolResult(block))
        break
      default:
        break
    }
  }

  flushUserMessage()
  return result
}

function convertAssistantMessage(
  message: AssistantMessage,
): ResponseInputItem[] {
  const content = message.message.content

  if (typeof content === 'string') {
    if (content.length === 0) return []
    return [
      {
        type: 'message',
        role: 'assistant',
        content,
      } satisfies EasyInputMessage,
    ]
  }

  if (!Array.isArray(content)) {
    return []
  }

  const result: ResponseInputItem[] = []
  let pendingText: string[] = []

  const flushAssistantText = (): void => {
    const text = pendingText.join('\n').trim()
    if (!text) {
      pendingText = []
      return
    }

    result.push({
      type: 'message',
      role: 'assistant',
      content: text,
    } satisfies EasyInputMessage)
    pendingText = []
  }

  for (const block of content) {
    if (typeof block === 'string') {
      pendingText.push(block)
      continue
    }

    switch (block.type) {
      case 'text':
        pendingText.push(block.text)
        break
      case 'tool_use':
        flushAssistantText()
        result.push(convertToolUse(block))
        break
      case 'thinking':
      case 'redacted_thinking':
        // We intentionally drop historical thinking blocks here. The current
        // codebase does not persist OpenAI Responses reasoning item ids, so a
        // lossy but stable conversion is safer than fabricating incompatible
        // reasoning items.
        break
      default:
        break
    }
  }

  flushAssistantText()
  return result
}

function convertToolUse(
  block: BetaToolUseBlock,
): ResponseFunctionToolCall {
  return {
    type: 'function_call',
    call_id: block.id,
    name: block.name,
    arguments:
      typeof block.input === 'string'
        ? block.input
        : JSON.stringify(block.input),
  }
}

function convertToolResult(block: BetaToolResultBlockParam): ResponseInputItem {
  return {
    type: 'function_call_output',
    call_id: block.tool_use_id,
    output: toolResultContentToString(block.content ?? ''),
  }
}

function toolResultContentToString(content: string | unknown[]): string {
  if (typeof content === 'string') {
    return content
  }

  return content
    .map(item => {
      if (typeof item === 'string') return item
      if (
        item &&
        typeof item === 'object' &&
        'text' in item &&
        typeof item.text === 'string'
      ) {
        return item.text
      }
      return ''
    })
    .filter(Boolean)
    .join('\n')
}

function convertImageBlock(
  block: Record<string, unknown>,
): ResponseInputImage | null {
  const source = block.source as Record<string, unknown> | undefined
  if (!source) return null

  if (source.type === 'base64' && typeof source.data === 'string') {
    const mediaType = (source.media_type as string) || 'image/png'
    return {
      type: 'input_image',
      image_url: `data:${mediaType};base64,${source.data}`,
      detail: 'auto',
    }
  }

  if (source.type === 'url' && typeof source.url === 'string') {
    return {
      type: 'input_image',
      image_url: source.url,
      detail: 'auto',
    }
  }

  return null
}
