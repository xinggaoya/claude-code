import { describe, expect, mock, test } from 'bun:test'
import { z } from 'zod/v4'
import { buildTool } from '../../../Tool.js'

type MockBehavior = {
  delayMs?: number
  isError?: boolean
  text: string
}

const toolBehaviors = new Map<string, MockBehavior>()

function makeToolResultMessage(
  toolUseId: string,
  assistantUuid: string,
  text: string,
  isError = false,
): any {
  return {
    type: 'user',
    uuid: `user-${toolUseId}`,
    timestamp: new Date().toISOString(),
    toolUseResult: text,
    sourceToolAssistantUUID: assistantUuid,
    message: {
      role: 'user',
      content: [
        {
          type: 'tool_result',
          tool_use_id: toolUseId,
          content: isError ? `<tool_use_error>${text}</tool_use_error>` : text,
          is_error: isError,
        },
      ],
    },
  }
}

mock.module('../toolExecution.js', () => ({
  runToolUse: async function* (
    block: any,
    assistantMessage: any,
    _canUseTool: any,
    _context: any,
  ) {
    const command = block.input?.command
    const behavior = toolBehaviors.get(command) ?? {
      text: `completed ${command}`,
    }

    if (behavior.delayMs) {
      await new Promise(resolve => setTimeout(resolve, behavior.delayMs))
    }

    yield {
      message: makeToolResultMessage(
        block.id,
        assistantMessage.uuid,
        behavior.text,
        behavior.isError,
      ),
    }
  },
}))

describe('StreamingToolExecutor', () => {
  test('does not cancel sibling Bash calls when one parallel Bash fails', async () => {
    const { StreamingToolExecutor } = await import('../StreamingToolExecutor.js')

    toolBehaviors.clear()
    toolBehaviors.set('fail', { text: 'bash failed', isError: true })
    toolBehaviors.set('slow-success', {
      text: 'bash succeeded',
      delayMs: 20,
    })

    const bashTool = buildTool({
      name: 'Bash',
      inputSchema: z.object({ command: z.string() }),
      maxResultSizeChars: 10_000,
      call: async () => ({ data: '' }),
      description: async () => 'Run a bash command',
      prompt: async () => 'Run a bash command',
      isConcurrencySafe: () => true,
      mapToolResultToToolResultBlockParam: (
        content: unknown,
        toolUseID: string,
      ) => ({
        type: 'tool_result' as const,
        tool_use_id: toolUseID,
        content: String(content),
      }),
      renderToolUseMessage: () => null,
    })

    let inProgress = new Set<string>()
    const context = {
      abortController: new AbortController(),
      setInProgressToolUseIDs: (updater: (prev: Set<string>) => Set<string>) => {
        inProgress = updater(inProgress)
      },
      setHasInterruptibleToolInProgress: () => {},
      options: {},
    } as any

    const executor = new StreamingToolExecutor(
      [bashTool],
      async () => ({ behavior: 'allow' }),
      context,
    )

    const assistantMessage = {
      type: 'assistant',
      uuid: 'assistant-1',
      timestamp: new Date().toISOString(),
      message: { role: 'assistant', content: [] },
    } as any

    executor.addTool(
      { id: 'tool-fail', name: 'Bash', input: { command: 'fail' } } as any,
      assistantMessage,
    )
    executor.addTool(
      {
        id: 'tool-success',
        name: 'Bash',
        input: { command: 'slow-success' },
      } as any,
      assistantMessage,
    )

    const updates: any[] = []
    for await (const update of executor.getRemainingResults()) {
      updates.push(update)
    }

    const results = updates
      .map(update => update.message?.toolUseResult)
      .filter(Boolean)

    expect(results).toContain('bash failed')
    expect(results).toContain('bash succeeded')
    expect(
      results.some((result: string) =>
        result.includes('Cancelled: parallel tool call'),
      ),
    ).toBe(false)
  })
})
