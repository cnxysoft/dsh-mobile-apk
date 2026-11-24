/**
 * Android native runtime bridge for `@deepseek-ai/dsh-experimental-speech-to-text-sensevoice`.
 *
 * Upstream `sensevoice` rejects `android-arm64` / `android-x64` in `resolveRuntime` and
 * loads desktop `sherpa-onnx-node` inside a spawned Node worker. On Android Bionic,
 * `sherpa-onnx-node` has no native binding; instead, the APK shell hosts a lazy,
 * authenticated loopback server backed by the official Android NDK `libsherpa-onnx-jni.so`
 * (v1.13.8, statically linked ONNX Runtime) and exposes `DSH_ANDROID_SPEECH_URL` +
 * `DSH_ANDROID_SPEECH_TOKEN` to the engine process.
 *
 * This patch:
 *   1. Adds `android-arm64` and `android-x64` to `resolveRuntime` so DSH enables local
 *      SenseVoice preparation, model download/verification, and custom `modelDirectory` /
 *      `vadModelPath` reuse on Android.
 *   2. Routes `SenseVoiceWorker.start()` on `process.platform === "android"` to the shell's
 *      loopback speech bridge (`/load`, `/session/<id>/transcribe`, `/unload`) without
 *      spawning an extra Node child process.
 */
import { assertEngineBaseline } from '../engine-versions.mjs'

export const SENSEVOICE_ANDROID_MARKER = 'dsh-mobile sensevoice android runtime'

const NM = 'usr/lib/node_modules/@deepseek-ai/dsh/node_modules/'
export const SENSEVOICE_INDEX_FILE = NM + '@deepseek-ai/dsh-experimental-speech-to-text-sensevoice/lib/index.js'
export const SENSEVOICE_PKG_FILE = NM + '@deepseek-ai/dsh-experimental-speech-to-text-sensevoice/package.json'

function replaceOnceStrict(source, before, after, label) {
  const eol = source.includes('\r\n') ? '\r\n' : '\n'
  const b = before.replace(/\r?\n/g, eol)
  const a = after.replace(/\r?\n/g, eol)
  const count = source.split(b).length - 1
  if (count !== 1) throw new Error(`sensevoice-android-runtime: anchor drift (${label}, matches=${count})`)
  return source.replace(b, a)
}

export function hasSenseVoiceAndroidRuntime(source) {
  return source.includes('// ' + SENSEVOICE_ANDROID_MARKER) &&
    source.includes('"android-arm64"') &&
    source.includes('DSH_ANDROID_SPEECH_URL')
}

export function patchSenseVoiceAndroidRuntime(source) {
  if (source.includes('// ' + SENSEVOICE_ANDROID_MARKER)) {
    if (!hasSenseVoiceAndroidRuntime(source)) {
      throw new Error('sensevoice-android-runtime: incomplete installed patch')
    }
    return source
  }
  let out = source
  out = replaceOnceStrict(
    out,
    `function resolveRuntime(config) {
\tif (![
\t\t"darwin-arm64",
\t\t"darwin-x64",
\t\t"linux-arm64",
\t\t"linux-x64",
\t\t"win32-x64"
\t].includes(\`\${process.platform}-\${process.arch}\`)) throw new Error(\`Local speech is unavailable for \${process.platform}-\${process.arch}\`);`,
    `function resolveRuntime(config) {
\tif (![
\t\t"android-arm64",
\t\t"android-x64",
\t\t"darwin-arm64",
\t\t"darwin-x64",
\t\t"linux-arm64",
\t\t"linux-x64",
\t\t"win32-x64"
\t].includes(\`\${process.platform}-\${process.arch}\`)) throw new Error(\`Local speech is unavailable for \${process.platform}-\${process.arch}\`);`,
    'resolveRuntime.platforms'
  )
  out = replaceOnceStrict(
    out,
    `\t\t\tawait mkdir(this.config.dataRoot, { recursive: true });
\t\t\tconst token = randomBytes(32).toString("hex");
\t\t\tconst handle = this.ctx.subprocess.spawn({`,
    `\t\t\tawait mkdir(this.config.dataRoot, { recursive: true });
\t\t\tif (process.platform === "android") {
\t\t\t\tconst bridgeUrl = process.env.DSH_ANDROID_SPEECH_URL;
\t\t\t\tconst bridgeToken = process.env.DSH_ANDROID_SPEECH_TOKEN;
\t\t\t\tif (!bridgeUrl || !bridgeToken) throw new Error("Android native speech bridge is unavailable");
\t\t\t\tconst loadResp = await fetch(\`\${bridgeUrl}/load\`, {
\t\t\t\t\tmethod: "POST",
\t\t\t\t\theaders: {
\t\t\t\t\t\tauthorization: \`Bearer \${bridgeToken}\`,
\t\t\t\t\t\t"content-type": "application/json"
\t\t\t\t\t},
\t\t\t\t\tbody: JSON.stringify(Object.assign({}, this.config, runtime)),
\t\t\t\t\tsignal: setup.signal
\t\t\t\t});
\t\t\t\tconst loadBody = await loadResp.json().catch(() => ({}));
\t\t\t\tif (!loadResp.ok || typeof loadBody?.sessionId !== "string" || loadBody.sessionId.length === 0) {
\t\t\t\t\tthrow new Error(typeof loadBody?.error === "string" ? loadBody.error : \`Android speech load failed (HTTP \${loadResp.status})\`);
\t\t\t\t}
\t\t\t\tconst sessionId = loadBody.sessionId;
\t\t\t\tlet unloading = null;
\t\t\t\tconst unload = () => {
\t\t\t\t\tif (unloading) return unloading;
\t\t\t\t\tunloading = fetch(\`\${bridgeUrl}/unload\`, {
\t\t\t\t\t\tmethod: "POST",
\t\t\t\t\t\theaders: {
\t\t\t\t\t\t\tauthorization: \`Bearer \${bridgeToken}\`,
\t\t\t\t\t\t\t"content-type": "application/json"
\t\t\t\t\t\t},
\t\t\t\t\t\tbody: JSON.stringify({ sessionId })
\t\t\t\t\t}).then(() => void 0, () => void 0);
\t\t\t\t\treturn unloading;
\t\t\t\t};
\t\t\t\tif (setup.signal.aborted) {
\t\t\t\t\tawait unload();
\t\t\t\t\tsetup.signal.throwIfAborted();
\t\t\t\t}
\t\t\t\tconst worker = {
\t\t\t\t\thandle: {
\t\t\t\t\t\tterminate() { void unload(); },
\t\t\t\t\t\twaitForExit() { return unload(); }
\t\t\t\t\t},
\t\t\t\t\turl: \`\${bridgeUrl}/session/\${encodeURIComponent(sessionId)}\`,
\t\t\t\t\ttoken: bridgeToken,
\t\t\t\t\tclosed: false
\t\t\t\t};
\t\t\t\tthis.worker = worker;
\t\t\t\tthis.publish({ phase: "ready" });
\t\t\t\treturn worker;
\t\t\t}
\t\t\tconst token = randomBytes(32).toString("hex");
\t\t\tconst handle = this.ctx.subprocess.spawn({`,
    'SenseVoiceWorker.start'
  )
  return out + '\n// ' + SENSEVOICE_ANDROID_MARKER + '\n'
}

export function planSenseVoiceAndroidRuntime(read) {
  const pkg = JSON.parse(read(SENSEVOICE_PKG_FILE))
  if (pkg.name !== '@deepseek-ai/dsh-experimental-speech-to-text-sensevoice') {
    throw new Error('sensevoice-android-runtime: package name mismatch')
  }
  assertEngineBaseline(pkg, 'sensevoice-android-runtime @deepseek-ai/dsh-experimental-speech-to-text-sensevoice')
  const before = read(SENSEVOICE_INDEX_FILE)
  return [{ file: SENSEVOICE_INDEX_FILE, before, after: patchSenseVoiceAndroidRuntime(before) }]
}
