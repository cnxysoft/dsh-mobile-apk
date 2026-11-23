import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { versionedFixture } from './lib/fixture.mjs'
import { patchImageExpiry } from '../image-expiry-E1.mjs'
import { patchRequestPreview } from '../image-request-preview-P1.mjs'
import { planPiStreaming } from '../pi-upstream-streaming-020.mjs'
import { REASONING_FILES, planReasoning } from '../data/reasoning-envelope-1304.mjs'
import {
  LONG_SESSION_CACHE_ADAPTER,
  LONG_SESSION_CACHE_FILES,
  LONG_SESSION_CACHE_RESPONSES,
  LONG_SESSION_CACHE_TRANSFORM,
  planLongSessionCache,
} from '../data/long-session-history-cache.mjs'

const NM = 'usr/lib/node_modules/@deepseek-ai/dsh/node_modules/'

function loadFixtureMap() {
  const map = new Map()
  const add = (rel, pkg, file) => map.set(rel, readFileSync(versionedFixture(pkg, file), 'utf8'))
  add(NM + '@deepseek-ai/dsh-llm-pi-ai/package.json', 'dsh-llm-pi-ai', 'package.json')
  add(LONG_SESSION_CACHE_ADAPTER, 'dsh-llm-pi-ai', 'lib/index.js')
  add(NM + '@deepseek-ai/dsh-client-ui-chat/lib/client.js', 'dsh-client-ui-chat', 'lib/client.js')
  add(NM + '@earendil-works/pi-ai/package.json', 'pi-ai', 'package.json')
  add(NM + '@earendil-works/pi-ai/dist/models.js', 'pi-ai', 'dist/models.js')
  for (const name of [
    'anthropic-messages', 'bedrock-converse-stream', 'mistral-conversations',
    'openai-completions', 'openai-responses-shared', 'pi-messages', 'transform-messages',
  ]) {
    add(NM + '@earendil-works/pi-ai/dist/api/' + name + '.js', 'pi-ai', 'dist/api/' + name + '.js')
  }
  return map
}

function applyPrerequisiteStack(files) {
  const read = file => {
    if (!files.has(file)) throw new Error('missing fixture: ' + file)
    return files.get(file)
  }
  // 1. P1 + E1 on dsh-llm-pi-ai
  let adapter = read(LONG_SESSION_CACHE_ADAPTER)
  adapter = patchRequestPreview(adapter, 'pi')
  adapter = patchImageExpiry(adapter, 'pi')
  files.set(LONG_SESSION_CACHE_ADAPTER, adapter)

  // 2. pi-upstream-streaming-020
  const piRoot = NM + '@earendil-works/pi-ai'
  for (const p of planPiStreaming(piRoot, f => read(piRoot + '/' + f))) {
    files.set(piRoot + '/' + p.file, p.after)
  }

  // 3. sealed-reasoning-1304
  for (const p of planReasoning(read)) {
    files.set(p.file, p.after)
  }
}

test('long-session-history-cache composes cleanly with P1/E1/streaming/reasoning-1304, is idempotent, and passes syntax check', () => {
  const files = loadFixtureMap()
  applyPrerequisiteStack(files)
  const read = file => files.get(file)

  const plan = planLongSessionCache(read)
  assert.equal(plan.length, LONG_SESSION_CACHE_FILES.length)
  for (const step of plan) {
    assert.notEqual(step.before, step.after, `${step.file} must change on first apply`)
    files.set(step.file, step.after)
    const check = spawnSync(process.execPath, ['--check', '--input-type=module'], { input: step.after, encoding: 'utf8' })
    assert.equal(check.status, 0, `${step.file} syntax error: ${check.stderr}`)
  }

  // Idempotent re-apply
  const second = planLongSessionCache(read)
  assert.ok(second.every(step => step.before === step.after), 'second apply must be a no-op')

  // Every prerequisite patch's own --check must still report clean applied state!
  assert.equal(patchImageExpiry(read(LONG_SESSION_CACHE_ADAPTER), 'pi'), read(LONG_SESSION_CACHE_ADAPTER))
  assert.equal(patchRequestPreview(read(LONG_SESSION_CACHE_ADAPTER), 'pi'), read(LONG_SESSION_CACHE_ADAPTER))
  const piRoot = NM + '@earendil-works/pi-ai'
  assert.ok(planPiStreaming(piRoot, f => read(piRoot + '/' + f)).every(s => s.before === s.after))
  assert.ok(planReasoning(read).every(s => s.before === s.after))
})

test('toPiAssistant caches frozen messages, preserves prefixDigests, and never caches degraded or unfrozen messages', () => {
  const files = loadFixtureMap()
  applyPrerequisiteStack(files)
  const read = file => files.get(file)
  const patchedAdapter = planLongSessionCache(read).find(s => s.file === LONG_SESSION_CACHE_ADAPTER).after

  const rStart = patchedAdapter.indexOf('//#region lib/types/replay.js')
  const rEnd = patchedAdapter.indexOf('//#endregion', rStart)
  class LlmError extends Error {
    constructor(msg, code) { super(msg); this.code = code }
  }
  const { toPiAssistant } = new Function('LlmError', patchedAdapter.slice(rStart, rEnd) + '\nreturn { toPiAssistant }')(LlmError)

  const frozenMsg = Object.freeze({
    role: 'assistant',
    content: Object.freeze([
      Object.freeze({ type: 'text', text: 'hello' }),
      Object.freeze({ type: 'tool-call', id: 'call_1|fc_1', name: 'read', arguments: '{"path":"a.txt"}' }),
    ]),
    source: Object.freeze({
      kind: 'model',
      provider: 'openai',
      model: 'gpt-5.4',
      replayState: Object.freeze({
        response: Object.freeze({
          kind: 'pi-ai',
          version: 2,
          api: 'openai-responses',
          provider: 'openai',
          model: 'gpt-5.4',
          stopReason: 'toolUse',
          prefixDigest: 'sha256:abc',
        }),
        blocks: Object.freeze([
          Object.freeze({ type: 'text', textSignature: '{"v":1,"id":"msg_1"}' }),
          Object.freeze({ type: 'tool-call' }),
        ]),
      }),
    }),
  })

  const d1 = new Map()
  const a1 = toPiAssistant(frozenMsg, () => {}, d1)
  assert.equal(d1.get(a1), 'sha256:abc')
  assert.equal(Object.isFrozen(a1), true)

  const d2 = new Map()
  const a2 = toPiAssistant(frozenMsg, () => {}, d2)
  assert.equal(a2, a1, 'frozen message must reuse cached PiAssistant reference')
  assert.equal(d2.get(a2), 'sha256:abc', 'cache hit must still populate prefixDigests')

  // Negative control: degraded replayState must invoke onDegrade every time
  const badFrozen = Object.freeze({
    role: 'assistant',
    content: Object.freeze([Object.freeze({ type: 'text', text: 'hi' })]),
    source: Object.freeze({ kind: 'model', provider: 'openai', model: 'gpt-5.4', replayState: Object.freeze({ bad: true }) }),
  })
  let degrades = 0
  toPiAssistant(badFrozen, () => { degrades++ })
  toPiAssistant(badFrozen, () => { degrades++ })
  assert.equal(degrades, 2, 'degraded replayState must never suppress onDegrade')

  // Negative control: unfrozen message is never cached
  const mutableMsg = {
    role: 'assistant',
    content: [{ type: 'text', text: 'v1' }],
    source: { kind: 'model', provider: 'openai', model: 'gpt-5.4' },
  }
  const m1 = toPiAssistant(mutableMsg)
  mutableMsg.content[0].text = 'v2'
  const m2 = toPiAssistant(mutableMsg)
  assert.notEqual(m1, m2)
  assert.equal(m2.content[0].text, 'v2')
})

test('transformMessages and convertResponsesMessages cache frozen same-model items with byte-identical output and safe fallback/surrogate handling', () => {
  const files = loadFixtureMap()
  applyPrerequisiteStack(files)
  const read = file => files.get(file)
  const plan = planLongSessionCache(read)
  const tmSrc = plan.find(s => s.file === LONG_SESSION_CACHE_TRANSFORM).after.replace(/^export /gm, '')
  const respSrc = plan.find(s => s.file === LONG_SESSION_CACHE_RESPONSES).after
    .replace(/^import .*$/gm, '')
    .replace(/^export /gm, '')

  const transformMessages = new Function(tmSrc + '\nreturn transformMessages;')()
  const origSanitize = (t) => t.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '')
  const convertResponsesMessages = new Function(
    'calculateCost', 'shortHash', 'parseStreamingJson', 'parseToolCallArguments', 'sanitizeSurrogates',
    'getSystemMessageText', 'renderSystemMessageUpdate', 'resolveTranscript', 'resolveTranscriptTools',
    'appendGrammarToolInputJsonDelta', 'getGrammarToolInput', 'getJsonSchemaToolParameters',
    'resolveGrammarConstrainedSampling', 'resolveJsonSchemaStrictSampling', 'transformMessages',
    respSrc + '\nreturn convertResponsesMessages;'
  )(
    null, s => 'h' + s.length, null, null, origSanitize,
    m => typeof m.content === 'string' ? m.content : '',
    m => typeof m.content === 'string' ? m.content : '',
    ctx => ctx,
    () => ({ requestTools: [], anchorsAdditions: false }),
    null, () => '', null, null, null, transformMessages
  )

  const modelA = { provider: 'openai', api: 'openai-responses', id: 'gpt-5.4', reasoning: true, input: ['text', 'image'] }
  const modelB = { provider: 'openai', api: 'openai-responses', id: 'gpt-5.3', reasoning: true, input: ['text', 'image'] }
  const allowed = new Set(['openai'])

  const signedAssistant = Object.freeze({
    role: 'assistant',
    provider: 'openai',
    api: 'openai-responses',
    model: 'gpt-5.4',
    stopReason: 'toolUse',
    content: Object.freeze([
      Object.freeze({ type: 'thinking', thinking: 'plan', thinkingSignature: JSON.stringify({ type: 'reasoning', id: 'rs_1', summary: [], encrypted_content: 'enc_abc' }) }),
      Object.freeze({ type: 'text', text: 'working', textSignature: JSON.stringify({ v: 1, id: 'msg_signed_1' }) }),
      Object.freeze({ type: 'toolCall', id: 'call_1|fc_1', name: 'bash', arguments: { cmd: 'ls' } }),
    ]),
  })
  const unsignedAssistant = Object.freeze({
    role: 'assistant',
    provider: 'openai',
    api: 'openai-responses',
    model: 'gpt-5.4',
    stopReason: 'stop',
    content: Object.freeze([
      Object.freeze({ type: 'text', text: 'no signature' }),
    ]),
  })
  const toolResultWithLoneSurrogate = {
    role: 'toolResult',
    toolCallId: 'call_1|fc_1',
    toolName: 'bash',
    content: [{ type: 'text', text: 'ok \uD83D end' }],
    isError: false,
  }

  // Same-model transform preserves frozen reference
  const t1 = transformMessages([signedAssistant, toolResultWithLoneSurrogate], modelA)
  const t2 = transformMessages([signedAssistant, toolResultWithLoneSurrogate], modelA)
  assert.equal(t1[0], signedAssistant)
  assert.equal(t2[0], signedAssistant)

  // Cross-model transform drops thinking and does not return original reference
  const tCross = transformMessages([signedAssistant], modelB)
  assert.notEqual(tCross[0], signedAssistant)
  assert.equal(tCross[0].content.some(b => b.type === 'thinking'), false)

  // Responses conversion: warm turn is bit-for-bit identical, lone surrogate still stripped
  const r1 = convertResponsesMessages(modelA, { messages: [signedAssistant, toolResultWithLoneSurrogate] }, allowed)
  const r2 = convertResponsesMessages(modelA, { messages: [signedAssistant, toolResultWithLoneSurrogate] }, allowed)
  assert.equal(JSON.stringify(r1), JSON.stringify(r2))
  assert.equal(r1[3].output, 'ok  end', 'lone surrogate must still be stripped by fast path')

  // Negative control: unsigned assistant uses msgIndex fallback id and must NOT return stale msgIndex when shifted
  const u1 = convertResponsesMessages(modelA, { messages: [unsignedAssistant] }, allowed)
  const u2 = convertResponsesMessages(modelA, { messages: [{ role: 'user', content: 'prefix' }, unsignedAssistant] }, allowed)
  assert.equal(u1[0].id, 'msg_pi_0')
  assert.equal(u2[1].id, 'msg_pi_1', 'index-dependent fallback id must never be mis-cached across index shift')
})
