/**
 * Tests for queryModelOpenAI in index.ts.
 *
 * Focused on the two bugs fixed:
 *  1. stop_reason was always null in the assembled AssistantMessage because
 *     partialMessage (from message_start) has stop_reason: null, and the
 *     stop_reason captured from message_delta was never applied.
 *  2. partialMessage was not reset to null after message_stop, so the safety
 *     fallback at the end of the loop would yield a second identical
 *     AssistantMessage (causing doubled content in the next API request).
 *
 * Strategy: mock getOpenAIClient + adaptOpenAIStreamToAnthropic so we can
 * feed pre-built Anthropic events directly into queryModelOpenAI and inspect
 * what it emits — without any real HTTP calls.
 */
import { describe, expect, test, mock, beforeEach, afterEach } from 'bun:test'
import type { BetaRawMessageStreamEvent } from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import type {
  AssistantMessage,
  StreamEvent,
} from '../../../../types/message.js'

// ─── helpers ─────────────────────────────────────────────────────────────────

/** Build a minimal message_start event */
function makeMessageStart(
  overrides: Record<string, any> = {},
): BetaRawMessageStreamEvent {
  return {
    type: 'message_start',
    message: {
      id: 'msg_test',
      type: 'message',
      role: 'assistant',
      content: [],
      model: 'test-model',
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
      ...overrides,
    },
  } as any
}

/** Build a content_block_start event for the given block type */
function makeContentBlockStart(
  index: number,
  type: 'text' | 'tool_use' | 'thinking',
  extra: Record<string, any> = {},
): BetaRawMessageStreamEvent {
  const block =
    type === 'text'
      ? { type: 'text', text: '' }
      : type === 'tool_use'
        ? { type: 'tool_use', id: 'toolu_test', name: 'bash', input: {} }
        : { type: 'thinking', thinking: '', signature: '' }
  return {
    type: 'content_block_start',
    index,
    content_block: { ...block, ...extra },
  } as any
}

/** Build a text_delta content_block_delta event */
function makeTextDelta(index: number, text: string): BetaRawMessageStreamEvent {
  return {
    type: 'content_block_delta',
    index,
    delta: { type: 'text_delta', text },
  } as any
}

/** Build an input_json_delta content_block_delta event */
function makeInputJsonDelta(
  index: number,
  json: string,
): BetaRawMessageStreamEvent {
  return {
    type: 'content_block_delta',
    index,
    delta: { type: 'input_json_delta', partial_json: json },
  } as any
}

/** Build a thinking_delta content_block_delta event */
function makeThinkingDelta(
  index: number,
  thinking: string,
): BetaRawMessageStreamEvent {
  return {
    type: 'content_block_delta',
    index,
    delta: { type: 'thinking_delta', thinking },
  } as any
}

/** Build a content_block_stop event */
function makeContentBlockStop(index: number): BetaRawMessageStreamEvent {
  return { type: 'content_block_stop', index } as any
}

/** Build a message_delta event with stop_reason and output_tokens */
function makeMessageDelta(
  stopReason: string,
  outputTokens: number,
): BetaRawMessageStreamEvent {
  return {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: outputTokens },
  } as any
}

/** Build a message_stop event */
function makeMessageStop(): BetaRawMessageStreamEvent {
  return { type: 'message_stop' } as any
}

/** Async generator from a fixed array of events */
async function* eventStream(events: BetaRawMessageStreamEvent[]) {
  for (const e of events) yield e
}

/** Collect all outputs from queryModelOpenAI into typed buckets */
let _moduleImportCounter = 0

async function importQueryModelOpenAI() {
  // 通过唯一 query string 避免复用其他测试文件已缓存的 index.js。
  return import(`../index.js?query-model-openai-test=${_moduleImportCounter++}`)
}

async function runQueryModel(
  events: BetaRawMessageStreamEvent[],
  envOverrides: Record<string, string | undefined> = {},
) {
  // Wire events into the mocked stream adapter
  _nextEvents = events
  _lastCreateArgs = null

  const effectiveEnv: Record<string, string | undefined> = {
    OPENAI_MODEL: undefined,
    OPENAI_BASE_URL: undefined,
    OPENAI_USE_RESPONSES: undefined,
    OPENAI_USE_CHAT_COMPLETIONS: '1',
    OPENAI_REASONING_EFFORT: undefined,
    OPENAI_ENABLE_THINKING: undefined,
    ...envOverrides,
  }

  // Save + apply env overrides
  const saved: Record<string, string | undefined> = {}
  for (const [k, v] of Object.entries(effectiveEnv)) {
    saved[k] = process.env[k]
    if (v === undefined) delete process.env[k]
    else process.env[k] = v
  }

  try {
    // We inline mock.module inside the try block.
    // Bun resolves mock.module at the call site synchronously (hoisted),
    // so we register once per test file, then re-import each time.
    const { queryModelOpenAI } = await importQueryModelOpenAI()

    const assistantMessages: AssistantMessage[] = []
    const streamEvents: StreamEvent[] = []
    const otherOutputs: any[] = []

    const minimalOptions: any = {
      model: 'test-model',
      tools: [],
      agents: [],
      querySource: 'main_loop',
      getToolPermissionContext: async () => ({
        alwaysAllow: [],
        alwaysDeny: [],
        needsPermission: [],
        mode: 'default',
        isBypassingPermissions: false,
      }),
    }

    for await (const item of queryModelOpenAI(
      [],
      [] as any,
      [],
      new AbortController().signal,
      minimalOptions,
    )) {
      if (item.type === 'assistant') {
        assistantMessages.push(item as AssistantMessage)
      } else if (item.type === 'stream_event') {
        streamEvents.push(item as StreamEvent)
      } else {
        otherOutputs.push(item)
      }
    }

    if (process.env.DEBUG_QUERY_MODEL_TEST === '1') {
      console.log(
        JSON.stringify(
          { assistantMessages, streamEvents, otherOutputs, lastCreateArgs: _lastCreateArgs },
          null,
          2,
        ),
      )
    }

    return { assistantMessages, streamEvents, otherOutputs }
  } finally {
    // Restore env
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k]
      else process.env[k] = v
    }
  }
}

// ─── mock setup ──────────────────────────────────────────────────────────────

// We mock at module level. Bun's mock.module replaces the module for the
// entire file, so we configure the stream per-test via a shared variable.
let _nextEvents: BetaRawMessageStreamEvent[] = []

/** Captured arguments from the last chat.completions.create() call */
let _lastCreateArgs: Record<string, any> | null = null
let _mockResponsesCreate = async (args: Record<string, any>) => {
  _lastCreateArgs = args
  return { [Symbol.asyncIterator]: async function* () {} }
}

mock.module('../client.js', () => ({
  getOpenAIClient: () => ({
    chat: {
      completions: {
        create: async (args: Record<string, any>) => {
          _lastCreateArgs = args
          return { [Symbol.asyncIterator]: async function* () {} }
        },
      },
    },
    responses: {
      create: async (args: Record<string, any>) => _mockResponsesCreate(args),
    },
  }),
}))

mock.module('../streamAdapter.js', () => ({
  adaptOpenAIStreamToAnthropic: (_stream: any, _model: string) =>
    eventStream(_nextEvents),
}))

mock.module('../responsesStreamAdapter.js', () => ({
  adaptOpenAIResponsesStreamToAnthropic: (_stream: any, _model: string) =>
    eventStream(_nextEvents),
}))

mock.module('../modelMapping.js', () => ({
  resolveOpenAIModel: (m: string) => process.env.OPENAI_MODEL || m,
}))

mock.module('../convertMessages.js', () => ({
  anthropicMessagesToOpenAI: () => [],
}))

mock.module('../convertTools.js', () => ({
  anthropicToolsToOpenAI: () => [],
  anthropicToolChoiceToOpenAI: () => undefined,
  anthropicToolsToOpenAIResponses: () => [],
  anthropicToolChoiceToOpenAIResponses: () => undefined,
}))

mock.module('../../../../utils/context.js', () => ({
  MODEL_CONTEXT_WINDOW_DEFAULT: 200_000,
  COMPACT_MAX_OUTPUT_TOKENS: 20_000,
  CAPPED_DEFAULT_MAX_TOKENS: 8_000,
  ESCALATED_MAX_TOKENS: 64_000,
  getModelMaxOutputTokens: () => ({ upperLimit: 8192, default: 8192 }),
  getContextWindowForModel: () => 200_000,
  modelSupports1M: () => false,
  is1mContextDisabled: () => false,
  has1mContext: () => false,
  getSonnet1mExpTreatmentEnabled: () => false,
  calculateContextPercentages: () => ({
    contextWindow: 200_000,
    used: 0,
    remaining: 200_000,
    percentage: 0,
  }),
  getMaxThinkingTokensForModel: () => 0,
}))

mock.module('../../../../utils/messages.js', () => ({
  normalizeMessagesForAPI: (msgs: any) => msgs,
  normalizeContentFromAPI: (blocks: any[]) => blocks,
  createAssistantAPIErrorMessage: (opts: any) => ({
    type: 'assistant',
    message: {
      content: [{ type: 'text', text: opts.content }],
      apiError: opts.apiError,
    },
    uuid: 'error-uuid',
    timestamp: new Date().toISOString(),
  }),
}))

mock.module('../../../../utils/api.js', () => ({
  toolToAPISchema: async (t: any) => t,
}))

mock.module('../../../../utils/toolSearch.js', () => ({
  isToolSearchEnabled: async () => false,
  extractDiscoveredToolNames: () => new Set(),
}))

mock.module('../../../../tools/ToolSearchTool/prompt.js', () => ({
  isDeferredTool: () => false,
  TOOL_SEARCH_TOOL_NAME: '__tool_search__',
}))

mock.module('../../../../cost-tracker.js', () => ({
  addToTotalSessionCost: () => {},
}))

mock.module('../../../../utils/modelCost.js', () => ({
  COST_TIER_15_75: 0,
  calculateUSDCost: () => 0,
  calculateCostFromTokens: () => 0,
  formatModelPricing: () => '$0',
  COST_TIER_3_15: 0,
  COST_TIER_5_25: 0,
  COST_TIER_30_150: 0,
  COST_HAIKU_35: 0,
  COST_HAIKU_45: 0,
  getOpus46CostTier: () => 0,
  getModelCosts: () => 0,
  getModelPricingString: () => '$0',
  MODEL_COSTS: {},
}))

mock.module('../../../../utils/debug.js', () => ({
  getMinDebugLogLevel: () => 'error',
  isDebugMode: () => false,
  enableDebugLogging: () => false,
  getDebugFilter: () => null,
  isDebugToStdErr: () => false,
  getDebugFilePath: () => null,
  setHasFormattedOutput: () => {},
  getHasFormattedOutput: () => false,
  logForDebugging: () => {},
  getDebugLogPath: () => '',
  logAntError: () => {},
}))

// ─── tests ───────────────────────────────────────────────────────────────────

describe('queryModelOpenAI — stop_reason propagation', () => {
  test('assembled AssistantMessage has stop_reason end_turn (not null)', async () => {
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'Hello'),
      makeContentBlockStop(0),
      makeMessageDelta('end_turn', 10),
      makeMessageStop(),
    ]

    const { assistantMessages } = await runQueryModel(_nextEvents)

    expect(assistantMessages).toHaveLength(1)
    expect(assistantMessages[0]!.message.stop_reason).toBe('end_turn')
  })

  test('assembled AssistantMessage has stop_reason tool_use', async () => {
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'tool_use'),
      makeInputJsonDelta(0, '{"cmd":"ls"}'),
      makeContentBlockStop(0),
      makeMessageDelta('tool_use', 20),
      makeMessageStop(),
    ]

    const { assistantMessages } = await runQueryModel(_nextEvents)

    expect(assistantMessages).toHaveLength(1)
    expect(assistantMessages[0]!.message.stop_reason).toBe('tool_use')
  })

  test('assembled AssistantMessage has stop_reason max_tokens', async () => {
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'truncated'),
      makeContentBlockStop(0),
      makeMessageDelta('max_tokens', 8192),
      makeMessageStop(),
    ]

    const { assistantMessages } = await runQueryModel(_nextEvents)

    // Two assistant-typed items: the content message + the max_output_tokens error signal.
    // The error signal is emitted as a synthetic assistant message by createAssistantAPIErrorMessage.
    expect(assistantMessages).toHaveLength(2)
    const contentMsg = assistantMessages[0]!
    expect(contentMsg.message.stop_reason).toBe('max_tokens')
    // Second item is the error signal (has apiError set)
    const errorMsg = assistantMessages[1]!.message as any
    expect(errorMsg.apiError).toBe('max_output_tokens')
  })

  test('stop_reason is null when no message_delta was received (safety fallback path)', async () => {
    // Stream ends without message_stop — triggers the safety fallback branch.
    // stop_reason stays null since no message_delta was ever seen.
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'partial'),
      makeContentBlockStop(0),
      // No message_delta / message_stop
    ]

    const { assistantMessages } = await runQueryModel(_nextEvents)

    // Safety fallback should yield the partial content
    expect(assistantMessages).toHaveLength(1)
    expect(assistantMessages[0]!.message.stop_reason).toBeNull()
  })
})

describe('queryModelOpenAI — usage accumulation', () => {
  test('usage in assembled message reflects all four fields from message_delta', async () => {
    // message_start has all fields=0 (trailing-chunk pattern: usage not yet available).
    // message_delta carries the real values after stream ends.
    // The spread in the message_delta handler must override all zeros from message_start,
    // including cache_read_input_tokens which was previously missing from message_delta.
    _nextEvents = [
      makeMessageStart({
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 0,
        },
      }),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'response'),
      makeContentBlockStop(0),
      // message_delta carries all four Anthropic usage fields (as emitted by the fixed streamAdapter)
      {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: {
          input_tokens: 30011,
          output_tokens: 190,
          cache_read_input_tokens: 19904,
          cache_creation_input_tokens: 0,
        },
      } as any,
      makeMessageStop(),
    ]

    const { assistantMessages } = await runQueryModel(_nextEvents)

    expect(assistantMessages).toHaveLength(1)
    const usage = assistantMessages[0]!.message.usage as any
    expect(usage.input_tokens).toBe(30011)
    expect(usage.output_tokens).toBe(190)
    // cache_read_input_tokens from message_delta overrides the 0 from message_start
    expect(usage.cache_read_input_tokens).toBe(19904)
    expect(usage.cache_creation_input_tokens).toBe(0)
  })

  test('usage is zero when no usage events arrive (prevents false autocompact)', async () => {
    // If usage stays 0, tokenCountWithEstimation will undercount — so at least
    // verify the field exists and is numeric (to detect regressions).
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'hi'),
      makeContentBlockStop(0),
      makeMessageDelta('end_turn', 0),
      makeMessageStop(),
    ]

    const { assistantMessages } = await runQueryModel(_nextEvents)

    const usage = assistantMessages[0]!.message.usage as any
    expect(typeof usage.input_tokens).toBe('number')
    expect(typeof usage.output_tokens).toBe('number')
  })
})

describe('queryModelOpenAI — no duplicate AssistantMessage (partialMessage reset)', () => {
  test('yields exactly one AssistantMessage per message_stop when content is present', async () => {
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'only once'),
      makeContentBlockStop(0),
      makeMessageDelta('end_turn', 5),
      makeMessageStop(),
    ]

    const { assistantMessages } = await runQueryModel(_nextEvents)

    // Before the fix, partialMessage was not reset to null, so the safety
    // fallback at the end of the loop would yield a second message with the
    // same message.id — causing mergeAssistantMessages to concatenate content.
    expect(assistantMessages).toHaveLength(1)
  })

  test('thinking + text response yields exactly one AssistantMessage', async () => {
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'thinking'),
      makeThinkingDelta(0, 'let me think'),
      makeContentBlockStop(0),
      makeContentBlockStart(1, 'text'),
      makeTextDelta(1, 'answer'),
      makeContentBlockStop(1),
      makeMessageDelta('end_turn', 30),
      makeMessageStop(),
    ]

    const { assistantMessages } = await runQueryModel(_nextEvents)

    expect(assistantMessages).toHaveLength(1)
  })

  test('safety fallback path still yields message when stream ends without message_stop', async () => {
    // Simulates a stream that cuts off without the normal termination sequence.
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'abrupt end'),
      // No content_block_stop, no message_delta, no message_stop
    ]

    const { assistantMessages } = await runQueryModel(_nextEvents)

    expect(assistantMessages).toHaveLength(1)
  })
})

describe('queryModelOpenAI — stream_events forwarded', () => {
  test('every adapted event is also yielded as stream_event for real-time display', async () => {
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'hello'),
      makeContentBlockStop(0),
      makeMessageDelta('end_turn', 5),
      makeMessageStop(),
    ]

    const { streamEvents } = await runQueryModel(_nextEvents)

    const eventTypes = streamEvents.map(e => (e as any).event?.type)
    expect(eventTypes).toContain('message_start')
    expect(eventTypes).toContain('content_block_start')
    expect(eventTypes).toContain('content_block_delta')
    expect(eventTypes).toContain('content_block_stop')
    expect(eventTypes).toContain('message_delta')
    expect(eventTypes).toContain('message_stop')
  })
})

describe('queryModelOpenAI — max_tokens forwarded to request', () => {
  test('buildOpenAIRequestBody includes max_tokens in the request payload', async () => {
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'hi'),
      makeContentBlockStop(0),
      makeMessageDelta('end_turn', 5),
      makeMessageStop(),
    ]

    await runQueryModel(_nextEvents)

    expect(_lastCreateArgs).not.toBeNull()
    expect(_lastCreateArgs!.max_tokens).toBe(8192)
  })

  test('calls Responses API when explicitly enabled', async () => {
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'hi'),
      makeContentBlockStop(0),
      makeMessageDelta('end_turn', 5),
      makeMessageStop(),
    ]

    await runQueryModel(_nextEvents, {
      OPENAI_MODEL: 'gpt-5.4',
      OPENAI_USE_CHAT_COMPLETIONS: undefined,
      OPENAI_USE_RESPONSES: '1',
    })

    expect(_lastCreateArgs).not.toBeNull()
    expect(_lastCreateArgs!.max_output_tokens).toBe(8192)
    expect(_lastCreateArgs!.stream).toBe(true)
  })

  test('falls back to Chat Completions when Responses API fails with retryable error', async () => {
    const savedResponses = process.env.OPENAI_USE_RESPONSES
    const savedChat = process.env.OPENAI_USE_CHAT_COMPLETIONS
    const savedModel = process.env.OPENAI_MODEL
    const savedDebug = process.env.DEBUG_QUERY_MODEL_TEST

    process.env.OPENAI_USE_RESPONSES = '1'
    delete process.env.OPENAI_USE_CHAT_COMPLETIONS
    process.env.OPENAI_MODEL = 'gpt-5.4'
    delete process.env.DEBUG_QUERY_MODEL_TEST
    _lastCreateArgs = null
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'fallback ok'),
      makeContentBlockStop(0),
      makeMessageDelta('end_turn', 5),
      makeMessageStop(),
    ]

    const originalResponsesCreate = _mockResponsesCreate
    _mockResponsesCreate = async () => {
      const error = new Error('502 Upstream request failed') as Error & {
        status?: number
      }
      error.status = 502
      throw error
    }

    try {
      const { queryModelOpenAI } = await importQueryModelOpenAI()
      for await (const _ of queryModelOpenAI(
        [],
        [] as any,
        [],
        new AbortController().signal,
        {
          model: 'claude-opus-4-6',
          tools: [],
          agents: [],
          querySource: 'main_loop',
          getToolPermissionContext: async () => ({
            alwaysAllow: [],
            alwaysDeny: [],
            needsPermission: [],
            mode: 'default',
            isBypassingPermissions: false,
          }),
        } as any,
      )) {
        // consume
      }
    } finally {
      _mockResponsesCreate = originalResponsesCreate
      if (savedResponses === undefined) {
        delete process.env.OPENAI_USE_RESPONSES
      } else {
        process.env.OPENAI_USE_RESPONSES = savedResponses
      }
      if (savedChat === undefined) {
        delete process.env.OPENAI_USE_CHAT_COMPLETIONS
      } else {
        process.env.OPENAI_USE_CHAT_COMPLETIONS = savedChat
      }
      if (savedModel === undefined) {
        delete process.env.OPENAI_MODEL
      } else {
        process.env.OPENAI_MODEL = savedModel
      }
      if (savedDebug === undefined) {
        delete process.env.DEBUG_QUERY_MODEL_TEST
      } else {
        process.env.DEBUG_QUERY_MODEL_TEST = savedDebug
      }
    }

    expect(_lastCreateArgs).not.toBeNull()
    expect(_lastCreateArgs!.max_tokens).toBe(8192)
  })

  test('forwards mapped reasoning effort for OpenAI reasoning models', async () => {
    _nextEvents = [
      makeMessageStart(),
      makeContentBlockStart(0, 'text'),
      makeTextDelta(0, 'hi'),
      makeContentBlockStop(0),
      makeMessageDelta('end_turn', 5),
      makeMessageStop(),
    ]
    _lastCreateArgs = null

    const saved = process.env.OPENAI_REASONING_EFFORT
    const savedModel = process.env.OPENAI_MODEL
    const savedChat = process.env.OPENAI_USE_CHAT_COMPLETIONS
    delete process.env.OPENAI_REASONING_EFFORT
    process.env.OPENAI_MODEL = 'gpt-5.4'
    process.env.OPENAI_USE_CHAT_COMPLETIONS = '1'
    try {
      const { queryModelOpenAI } = await importQueryModelOpenAI()
      for await (const _ of queryModelOpenAI(
        [],
        { type: 'text', text: '' } as any,
        [],
        new AbortController().signal,
        {
          model: 'claude-opus-4-6',
          effortValue: 'max',
          tools: [],
          agents: [],
          querySource: 'main_loop',
          getToolPermissionContext: async () => ({
            alwaysAllow: [],
            alwaysDeny: [],
            needsPermission: [],
            mode: 'default',
            isBypassingPermissions: false,
          }),
        } as any,
      )) {
        // consume
      }
    } finally {
      if (saved === undefined) {
        delete process.env.OPENAI_REASONING_EFFORT
      } else {
        process.env.OPENAI_REASONING_EFFORT = saved
      }
      if (savedModel === undefined) {
        delete process.env.OPENAI_MODEL
      } else {
        process.env.OPENAI_MODEL = savedModel
      }
      if (savedChat === undefined) {
        delete process.env.OPENAI_USE_CHAT_COMPLETIONS
      } else {
        process.env.OPENAI_USE_CHAT_COMPLETIONS = savedChat
      }
    }

    expect(_lastCreateArgs).not.toBeNull()
    expect(_lastCreateArgs!.reasoning_effort).toBe('xhigh')
  })
})
