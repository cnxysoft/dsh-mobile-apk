import test from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'
import { versionedFixture } from './lib/fixture.mjs'
import {
  SENSEVOICE_INDEX_FILE,
  SENSEVOICE_PKG_FILE,
  hasSenseVoiceAndroidRuntime,
  patchSenseVoiceAndroidRuntime,
  planSenseVoiceAndroidRuntime,
} from '../data/sensevoice-android-runtime.mjs'

function loadFixtureMap() {
  const map = new Map()
  map.set(
    SENSEVOICE_PKG_FILE,
    readFileSync(versionedFixture('dsh-experimental-speech-to-text-sensevoice', 'package.json'), 'utf8'),
  )
  map.set(
    SENSEVOICE_INDEX_FILE,
    readFileSync(versionedFixture('dsh-experimental-speech-to-text-sensevoice', 'lib/index.js'), 'utf8'),
  )
  return map
}

test('sensevoice-android-runtime applies cleanly, is idempotent, ASCII-safe, and passes syntax check', () => {
  const files = loadFixtureMap()
  const read = (file) => files.get(file)

  const plan = planSenseVoiceAndroidRuntime(read)
  assert.equal(plan.length, 1)
  const step = plan[0]
  assert.notEqual(step.before, step.after)
  assert.equal(hasSenseVoiceAndroidRuntime(step.before), false)
  assert.equal(hasSenseVoiceAndroidRuntime(step.after), true)

  // Must stay strictly ASCII (repo patch discipline).
  const nonAscii = [...step.after].filter((ch) => ch.codePointAt(0) > 0x7f)
  assert.equal(nonAscii.length, 0, 'patched sensevoice index.js must remain pure ASCII')

  const syntax = spawnSync(process.execPath, ['--check', '--input-type=module'], {
    input: step.after,
    encoding: 'utf8',
  })
  assert.equal(syntax.status, 0, `syntax error: ${syntax.stderr}`)

  // Idempotent second pass.
  files.set(step.file, step.after)
  const second = planSenseVoiceAndroidRuntime(read)
  assert.equal(second[0].before, second[0].after)
  assert.equal(patchSenseVoiceAndroidRuntime(step.after), step.after)
})

test('patched resolveRuntime accepts android-arm64 and android-x64 while rejecting unknown platforms', () => {
  const files = loadFixtureMap()
  const patched = planSenseVoiceAndroidRuntime((f) => files.get(f))[0].after
  const start = patched.indexOf('function resolveRuntime(config) {')
  const end = patched.indexOf('async function verifyRuntime(', start)
  assert.ok(start >= 0 && end > start)

  const runWithPlatform = (platform, arch) => {
    const fakeProcess = { platform, arch }
    const fakeReadFileSync = () => JSON.stringify({
      models: { int8: { name: 'model.int8.onnx' }, fp32: { name: 'model.onnx' } },
      tokens: { name: 'tokens.txt' },
      vad: { name: 'silero_vad.onnx' },
    })
    const fn = new Function(
      'process', 'readFileSync', 'join', 'fileURLToPath',
      patched.slice(start, end).replaceAll('import.meta.url', '"file:///fake/lib/index.js"') + '\nreturn resolveRuntime;',
    )(
      fakeProcess,
      fakeReadFileSync,
      (...parts) => parts.join('/'),
      () => '/fake/worker.js',
    )
    return fn({ dataRoot: '/data/speech', precision: 'int8' })
  }

  const arm64 = runWithPlatform('android', 'arm64')
  assert.equal(arm64.paths.model, '/data/speech/models/sensevoice-onnx/model.int8.onnx')
  assert.equal(arm64.paths.tokens, '/data/speech/models/sensevoice-onnx/tokens.txt')
  assert.equal(arm64.paths.vad, '/data/speech/models/silero/silero_vad.onnx')

  const x64 = runWithPlatform('android', 'x64')
  assert.equal(x64.paths.model, '/data/speech/models/sensevoice-onnx/model.int8.onnx')

  assert.throws(() => runWithPlatform('freebsd', 'x64'), /Local speech is unavailable for freebsd-x64/)
})

test('patched Android SenseVoiceWorker bridges /load, /session/<id>/transcribe, and /unload over loopback HTTP', async () => {
  const files = loadFixtureMap()
  const patched = planSenseVoiceAndroidRuntime((f) => files.get(f))[0].after

  const token = 'a'.repeat(64)
  const calls = []
  const server = createServer((req, res) => {
    if (req.headers.authorization !== `Bearer ${token}`) {
      res.writeHead(401, { 'content-type': 'application/json' }).end(JSON.stringify({ error: 'Unauthorized' }))
      return
    }
    const chunks = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', () => {
      const body = Buffer.concat(chunks)
      calls.push({ method: req.method, url: req.url, body })
      if (req.url === '/load') {
        const parsed = JSON.parse(body.toString('utf8'))
        assert.ok(parsed.model && parsed.tokens && parsed.vad)
        res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ sessionId: 'sess-42' }))
        return
      }
      if (req.url === '/session/sess-42/transcribe?language=zh') {
        res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({
          text: 'hello android',
          audioSeconds: 0.5,
          inferenceSeconds: 0.08,
        }))
        return
      }
      if (req.url === '/unload') {
        const parsed = JSON.parse(body.toString('utf8'))
        assert.equal(parsed.sessionId, 'sess-42')
        res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: true }))
        return
      }
      res.writeHead(404).end()
    })
  })

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const { port } = server.address()
  const workDir = mkdtempSync(join(tmpdir(), 'sv-android-test-'))
  try {
    // Extract the Android branch from patched SenseVoiceWorker.start() and run it against the real mock server.
    const branchStart = patched.indexOf('if (process.platform === "android") {')
    const branchEnd = patched.indexOf('const token = randomBytes(32).toString("hex");', branchStart)
    assert.ok(branchStart > 0 && branchEnd > branchStart)
    const branchCode = patched.slice(branchStart, branchEnd)

    const fakeProcess = {
      platform: 'android',
      env: {
        DSH_ANDROID_SPEECH_URL: `http://127.0.0.1:${port}`,
        DSH_ANDROID_SPEECH_TOKEN: token,
      },
    }
    const runStart = new Function(
      'process', 'fetch', 'setup', 'runtime',
      `return (async function() { ${branchCode} })`,
    )
    const published = []
    const host = {
      config: { dataRoot: workDir, threads: 2, maxAudioBytes: 4 * 1024 * 1024 },
      worker: undefined,
      publish(s) { published.push(s) },
    }
    const ac = new AbortController()
    const worker = await runStart(
      fakeProcess,
      globalThis.fetch,
      { signal: ac.signal },
      { model: join(workDir, 'model.int8.onnx'), tokens: join(workDir, 'tokens.txt'), vad: join(workDir, 'silero_vad.onnx') },
    ).call(host)

    assert.equal(host.worker, worker)
    assert.deepEqual(published, [{ phase: 'ready' }])
    assert.equal(worker.url, `http://127.0.0.1:${port}/session/sess-42`)
    assert.equal(worker.token, token)

    const tr = await fetch(`${worker.url}/transcribe?language=zh`, {
      method: 'POST',
      headers: { authorization: `Bearer ${worker.token}`, 'content-type': 'audio/wav' },
      body: Buffer.alloc(64),
    })
    assert.equal(tr.status, 200)
    assert.deepEqual(await tr.json(), { text: 'hello android', audioSeconds: 0.5, inferenceSeconds: 0.08 })

    worker.handle.terminate()
    await worker.handle.waitForExit()
    assert.deepEqual(
      calls.map((c) => `${c.method} ${c.url}`),
      ['POST /load', 'POST /session/sess-42/transcribe?language=zh', 'POST /unload'],
    )
  } finally {
    server.close()
    rmSync(workDir, { recursive: true, force: true })
  }
})
