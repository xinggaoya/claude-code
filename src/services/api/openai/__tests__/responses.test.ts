import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import {
  buildOpenAIResponsesRequestBody,
  shouldFallbackFromResponsesError,
  shouldUseOpenAIResponsesAPI,
} from '../index.js'
import {
  anthropicMessagesToOpenAIResponsesInput,
  systemPromptToOpenAIInstructions,
} from '../responsesConvertMessages.js'

describe('Responses API routing helpers', () => {
  const originalEnv = {
    OPENAI_BASE_URL: process.env.OPENAI_BASE_URL,
    OPENAI_USE_RESPONSES: process.env.OPENAI_USE_RESPONSES,
    OPENAI_USE_CHAT_COMPLETIONS: process.env.OPENAI_USE_CHAT_COMPLETIONS,
  }

  beforeEach(() => {
    delete process.env.OPENAI_BASE_URL
    delete process.env.OPENAI_USE_RESPONSES
    delete process.env.OPENAI_USE_CHAT_COMPLETIONS
  })

  afterEach(() => {
    for (const [key, value] of Object.entries(originalEnv)) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
  })

  test('uses Responses API for official OpenAI models on default base URL', () => {
    expect(shouldUseOpenAIResponsesAPI('gpt-5.4')).toBe(true)
    expect(shouldUseOpenAIResponsesAPI('o3')).toBe(true)
  })

  test('keeps chat completions for non-official or compatibility models by default', () => {
    expect(shouldUseOpenAIResponsesAPI('deepseek-reasoner')).toBe(false)
    expect(shouldUseOpenAIResponsesAPI('test-model')).toBe(false)
  })

  test('keeps chat completions on custom gateways unless explicitly forced', () => {
    process.env.OPENAI_BASE_URL = 'https://my-gateway.example.com/v1'
    expect(shouldUseOpenAIResponsesAPI('gpt-5.4')).toBe(false)
    process.env.OPENAI_USE_RESPONSES = '1'
    expect(shouldUseOpenAIResponsesAPI('gpt-5.4')).toBe(true)
  })

  test('allows explicit chat-completions override', () => {
    process.env.OPENAI_USE_CHAT_COMPLETIONS = '1'
    expect(shouldUseOpenAIResponsesAPI('gpt-5.4')).toBe(false)
  })

  test('falls back from upstream /responses failures', () => {
    expect(
      shouldFallbackFromResponsesError(
        new Error('502 Upstream request failed'),
      ),
    ).toBe(true)
    expect(
      shouldFallbackFromResponsesError({
        status: 404,
        message: 'Not Found',
      }),
    ).toBe(true)
    expect(
      shouldFallbackFromResponsesError(new Error('authentication failed')),
    ).toBe(false)
  })
})

describe('Responses API request building', () => {
  test('builds instructions + input + reasoning payload', () => {
    const body = buildOpenAIResponsesRequestBody({
      model: 'gpt-5.4',
      instructions: 'You are helpful.',
      input: [{ type: 'message', role: 'user', content: 'hello' }],
      tools: [
        {
          type: 'function',
          name: 'bash',
          parameters: { type: 'object' },
          strict: true,
        },
      ],
      toolChoice: 'auto',
      reasoningEffort: 'high',
      maxTokens: 2048,
      temperatureOverride: 0.2,
    })

    expect(body.instructions).toBe('You are helpful.')
    expect(body.max_output_tokens).toBe(2048)
    expect(body.reasoning).toEqual({ effort: 'high' })
    expect(body.stream).toBe(true)
    expect(body.temperature).toBe(0.2)
  })
})

describe('Responses input conversion', () => {
  test('preserves tool names for Responses requests', () => {
    const input = anthropicMessagesToOpenAIResponsesInput([
      {
        type: 'assistant',
        message: {
          content: [
            {
              type: 'tool_use',
              id: 'call_1',
              name: 'Read',
              input: { file_path: '/tmp/a' },
            },
          ],
        },
      } as any,
    ])

    expect(input).toEqual([
      {
        type: 'function_call',
        call_id: 'call_1',
        name: 'Read',
        arguments: '{"file_path":"/tmp/a"}',
      },
    ])
  })

  test('converts assistant tool calls and tool results into Responses input items', () => {
    const input = anthropicMessagesToOpenAIResponsesInput(
      [
        {
          type: 'assistant',
          message: {
            content: [
              { type: 'text', text: 'I will inspect files.' },
              {
                type: 'tool_use',
                id: 'call_1',
                name: 'bash',
                input: { command: 'ls' },
              },
            ],
          },
        } as any,
        {
          type: 'user',
          message: {
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'call_1',
                content: 'file-a\nfile-b',
              },
              { type: 'text', text: 'continue' },
            ],
          },
        } as any,
      ],
    )

    expect(input).toEqual([
      { type: 'message', role: 'assistant', content: 'I will inspect files.' },
      {
        type: 'function_call',
        call_id: 'call_1',
        name: 'bash',
        arguments: '{"command":"ls"}',
      },
      {
        type: 'function_call_output',
        call_id: 'call_1',
        output: 'file-a\nfile-b',
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'continue' }],
      },
    ])
  })

  test('converts system prompt array into instructions string', () => {
    expect(systemPromptToOpenAIInstructions(['one', 'two'] as any)).toBe(
      'one\n\ntwo',
    )
  })
})

describe('Responses stream adapter', () => {
  let adapterImportCounter = 0

  async function collect(events: any[]) {
    const collected: any[] = []
    async function* source() {
      for (const event of events) yield event
    }
    const { adaptOpenAIResponsesStreamToAnthropic } = await import(
      `../responsesStreamAdapter.js?responses-test-real-adapter=${adapterImportCounter++}`
    )
    for await (const event of adaptOpenAIResponsesStreamToAnthropic(source(), 'gpt-5.4')) {
      collected.push(event)
    }
    return collected
  }

  test('converts output text stream into anthropic text blocks', async () => {
    const events = await collect([
      {
        type: 'response.created',
        response: {
          id: 'resp_1',
          usage: {
            input_tokens: 10,
            output_tokens: 0,
            input_tokens_details: { cached_tokens: 2 },
          },
        },
        sequence_number: 1,
      },
      {
        type: 'response.content_part.added',
        output_index: 0,
        content_index: 0,
        item_id: 'msg_1',
        part: { type: 'output_text', text: '', annotations: [] },
        sequence_number: 2,
      },
      {
        type: 'response.output_text.delta',
        output_index: 0,
        content_index: 0,
        item_id: 'msg_1',
        delta: 'Hello',
        logprobs: [],
        sequence_number: 3,
      },
      {
        type: 'response.output_text.done',
        output_index: 0,
        content_index: 0,
        item_id: 'msg_1',
        text: 'Hello',
        logprobs: [],
        sequence_number: 4,
      },
      {
        type: 'response.completed',
        response: {
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            input_tokens_details: { cached_tokens: 2 },
          },
        },
        sequence_number: 5,
      },
    ])

    expect(events[0].type).toBe('message_start')
    expect(events[1].type).toBe('content_block_start')
    expect(events[2].delta.type).toBe('text_delta')
    expect(events[2].delta.text).toBe('Hello')
    expect(events.at(-2).type).toBe('message_delta')
    expect(events.at(-2).delta.stop_reason).toBe('end_turn')
    expect(events.at(-2).usage.cache_read_input_tokens).toBe(2)
    expect(events.at(-1).type).toBe('message_stop')
  })

  test('converts function_call stream into anthropic tool_use blocks', async () => {
    const events = await collect(
      [
        {
          type: 'response.created',
          response: {
            id: 'resp_2',
            usage: {
              input_tokens: 1,
              output_tokens: 0,
              input_tokens_details: { cached_tokens: 0 },
            },
          },
          sequence_number: 1,
        },
        {
          type: 'response.output_item.added',
          output_index: 0,
          item: {
            type: 'function_call',
            id: 'fc_1',
            call_id: 'call_1',
            name: 'Read',
            arguments: '',
          },
          sequence_number: 2,
        },
        {
          type: 'response.function_call_arguments.delta',
          output_index: 0,
          item_id: 'fc_1',
          delta: '{"command":"ls"}',
          sequence_number: 3,
        },
        {
          type: 'response.function_call_arguments.done',
          output_index: 0,
          item_id: 'fc_1',
          name: 'Read',
          arguments: '{"command":"ls"}',
          sequence_number: 4,
        },
        {
          type: 'response.completed',
          response: {
            usage: {
              input_tokens: 1,
              output_tokens: 3,
              input_tokens_details: { cached_tokens: 0 },
            },
          },
          sequence_number: 5,
        },
      ],
    )

    expect(events[1].content_block.type).toBe('tool_use')
    expect(events[1].content_block.name).toBe('Read')
    expect(events[2].delta.type).toBe('input_json_delta')
    expect(events.at(-2).delta.stop_reason).toBe('tool_use')
  })
})
