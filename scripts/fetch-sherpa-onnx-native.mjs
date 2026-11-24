#!/usr/bin/env node
/**
 * Fetch and extract per-ABI `libsherpa-onnx-jni.so` from the pinned upstream
 * `sherpa-onnx-static-link-onnxruntime-<version>.aar` release asset.
 *
 * Why per-ABI extraction instead of a multi-ABI Gradle AAR dependency:
 *   This repository builds one APK per ABI (`arm64-v8a` or `x86_64`) without
 *   Gradle `splits`. Extracting only the requested ABI keeps the arm64 APK
 *   free of unused `armeabi-v7a` / `x86` / `x86_64` native libraries (~27 MB
 *   saved) and keeps `:app:testDebugUnitTest` completely offline.
 *
 * Usage:
 *   node scripts/fetch-sherpa-onnx-native.mjs --abis arm64-v8a --out <jniDir> [--cache <cacheDir>]
 *   node scripts/fetch-sherpa-onnx-native.mjs --self-test
 */
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { inflateRawSync } from 'node:zlib'

export const SHERPA_ONNX_VERSION = '1.13.8'
export const SHERPA_AAR_NAME = `sherpa-onnx-static-link-onnxruntime-${SHERPA_ONNX_VERSION}.aar`
export const SHERPA_AAR_URL =
  `https://github.com/k2-fsa/sherpa-onnx/releases/download/v${SHERPA_ONNX_VERSION}/${SHERPA_AAR_NAME}`
export const SHERPA_JNI_LIB_NAME = 'libsherpa-onnx-jni.so'

export const ABI_ELF_MACHINE = Object.freeze({
  'arm64-v8a': 183, // EM_AARCH64
  x86_64: 62, // EM_X86_64
})

const ELF_MAGIC = 0x464c457f

/** Validate that `bytes` is a 64-bit ELF matching `abi`. */
export function verifyElfForAbi(bytes, abi) {
  const expectedMachine = ABI_ELF_MACHINE[abi]
  if (expectedMachine === undefined) {
    throw new Error(`Unsupported sherpa-onnx ABI: ${abi}`)
  }
  if (!Buffer.isBuffer(bytes) || bytes.length < 20) {
    throw new Error(`${SHERPA_JNI_LIB_NAME} (${abi}): file too small to be ELF (${bytes?.length ?? 0} bytes)`)
  }
  if (bytes.readUInt32LE(0) !== ELF_MAGIC) {
    throw new Error(`${SHERPA_JNI_LIB_NAME} (${abi}): invalid ELF magic`)
  }
  if (bytes[4] !== 2) {
    throw new Error(`${SHERPA_JNI_LIB_NAME} (${abi}): expected ELF64 (class=2), got class=${bytes[4]}`)
  }
  const machine = bytes.readUInt16LE(18)
  if (machine !== expectedMachine) {
    throw new Error(
      `${SHERPA_JNI_LIB_NAME} (${abi}): ELF machine mismatch (got ${machine}, expected ${expectedMachine})`,
    )
  }
  return { size: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'), machine }
}

/**
 * Pure-Node ZIP entry extractor (supports method 0 stored and method 8 deflate).
 * Avoids platform-specific unzip/python dependencies across Windows and Linux.
 */
export function extractZipEntry(zipBuffer, wantedName) {
  const eocdSig = 0x06054b50
  const maxComment = Math.min(zipBuffer.length, 65557)
  let eocdOffset = -1
  for (let i = zipBuffer.length - 22; i >= zipBuffer.length - maxComment; i -= 1) {
    if (i >= 0 && zipBuffer.readUInt32LE(i) === eocdSig) {
      eocdOffset = i
      break
    }
  }
  if (eocdOffset < 0) throw new Error('Invalid ZIP/AAR archive: EOCD record not found')

  const totalEntries = zipBuffer.readUInt16LE(eocdOffset + 10)
  const cdOffset = zipBuffer.readUInt32LE(eocdOffset + 16)
  let pos = cdOffset
  for (let idx = 0; idx < totalEntries; idx += 1) {
    if (pos + 46 > zipBuffer.length || zipBuffer.readUInt32LE(pos) !== 0x02014b50) {
      throw new Error('Invalid ZIP/AAR archive: corrupt central directory')
    }
    const method = zipBuffer.readUInt16LE(pos + 10)
    const compSize = zipBuffer.readUInt32LE(pos + 20)
    const uncompSize = zipBuffer.readUInt32LE(pos + 24)
    const nameLen = zipBuffer.readUInt16LE(pos + 28)
    const extraLen = zipBuffer.readUInt16LE(pos + 30)
    const commentLen = zipBuffer.readUInt16LE(pos + 32)
    const localHeaderOffset = zipBuffer.readUInt32LE(pos + 42)
    const name = zipBuffer.subarray(pos + 46, pos + 46 + nameLen).toString('utf8')
    pos += 46 + nameLen + extraLen + commentLen

    if (name !== wantedName) continue

    if (localHeaderOffset + 30 > zipBuffer.length || zipBuffer.readUInt32LE(localHeaderOffset) !== 0x04034b50) {
      throw new Error(`Invalid ZIP/AAR local header for ${wantedName}`)
    }
    const localNameLen = zipBuffer.readUInt16LE(localHeaderOffset + 26)
    const localExtraLen = zipBuffer.readUInt16LE(localHeaderOffset + 28)
    const dataStart = localHeaderOffset + 30 + localNameLen + localExtraLen
    const compressed = zipBuffer.subarray(dataStart, dataStart + compSize)
    if (method === 0) {
      if (compressed.length !== uncompSize) throw new Error(`Size mismatch for stored entry ${wantedName}`)
      return Buffer.from(compressed)
    }
    if (method === 8) {
      const inflated = inflateRawSync(compressed)
      if (inflated.length !== uncompSize) throw new Error(`Size mismatch for deflated entry ${wantedName}`)
      return inflated
    }
    throw new Error(`Unsupported ZIP compression method ${method} for ${wantedName}`)
  }
  return null
}

function downloadAar(url, destFile) {
  mkdirSync(dirname(destFile), { recursive: true })
  const tempFile = `${destFile}.tmp-${process.pid}`
  rmSync(tempFile, { force: true })
  const curlCmd = process.platform === 'win32' ? 'curl.exe' : 'curl'
  const args = ['-fSL', '--retry', '3', '--retry-delay', '2', '-o', tempFile, url]
  const res = spawnSync(curlCmd, args, { stdio: 'inherit' })
  if (res.error || res.status !== 0 || !existsSync(tempFile) || statSync(tempFile).size < 1024) {
    rmSync(tempFile, { force: true })
    throw new Error(`Failed to download ${url} (exit=${res.status ?? res.error?.message ?? 'unknown'})`)
  }
  renameSync(tempFile, destFile)
}

function parseArgs(argv) {
  const opts = { abis: [], out: null, cache: null, selfTest: false }
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (arg === '--self-test') opts.selfTest = true
    else if (arg === '--abis') opts.abis = String(argv[++i] ?? '').split(',').map((s) => s.trim()).filter(Boolean)
    else if (arg === '--out') opts.out = argv[++i] ?? null
    else if (arg === '--cache') opts.cache = argv[++i] ?? null
    else throw new Error(`Unknown argument: ${arg}`)
  }
  return opts
}

function runSelfTest() {
  const fakeArm64 = Buffer.alloc(64)
  fakeArm64.writeUInt32LE(ELF_MAGIC, 0)
  fakeArm64[4] = 2
  fakeArm64.writeUInt16LE(183, 18)
  const ok = verifyElfForAbi(fakeArm64, 'arm64-v8a')
  if (ok.machine !== 183) throw new Error('expected machine 183')
  let threw = false
  try { verifyElfForAbi(fakeArm64, 'x86_64') } catch { threw = true }
  if (!threw) throw new Error('expected mismatch error for arm64 ELF checked against x86_64')
  console.log('FETCH-SHERPA-ONNX-NATIVE SELF-TEST PASSED')
}

function main() {
  const opts = parseArgs(process.argv.slice(2))
  if (opts.selfTest) {
    runSelfTest()
    return
  }
  if (!opts.out || opts.abis.length === 0) {
    console.error('Usage: node scripts/fetch-sherpa-onnx-native.mjs --abis <arm64-v8a[,x86_64]> --out <jniDir> [--cache <cacheDir>]')
    process.exit(2)
  }
  for (const abi of opts.abis) {
    if (!(abi in ABI_ELF_MACHINE)) {
      throw new Error(`Unsupported ABI "${abi}" (expected one of: ${Object.keys(ABI_ELF_MACHINE).join(', ')})`)
    }
  }
  const outDir = resolve(opts.out)
  const cacheDir = resolve(opts.cache ?? join(dirname(outDir), '..', 'cache', 'sherpa-onnx'))
  const overridePath = process.env.DSH_SHERPA_AAR_PATH
  const aarFile = overridePath ? resolve(overridePath) : join(cacheDir, SHERPA_AAR_NAME)

  // Fast path: if every requested ABI .so is already staged and valid, skip reading the AAR.
  const allPresent = opts.abis.every((abi) => {
    const target = join(outDir, abi, SHERPA_JNI_LIB_NAME)
    if (!existsSync(target)) return false
    try {
      verifyElfForAbi(readFileSync(target), abi)
      return true
    } catch {
      return false
    }
  })

  if (!allPresent) {
    if (!existsSync(aarFile)) {
      if (overridePath) throw new Error(`DSH_SHERPA_AAR_PATH not found: ${aarFile}`)
      console.log(`[sherpa-onnx] downloading ${SHERPA_AAR_NAME}...`)
      downloadAar(SHERPA_AAR_URL, aarFile)
    }
    const zipBytes = readFileSync(aarFile)
    const aarSha = createHash('sha256').update(zipBytes).digest('hex')
    console.log(`[sherpa-onnx] using ${aarFile} (size=${zipBytes.length}, sha256=${aarSha})`)

    mkdirSync(outDir, { recursive: true })
    for (const abi of opts.abis) {
      const entryName = `jni/${abi}/${SHERPA_JNI_LIB_NAME}`
      const libBytes = extractZipEntry(zipBytes, entryName)
      if (!libBytes) {
        throw new Error(`Missing ${entryName} inside ${aarFile}`)
      }
      const info = verifyElfForAbi(libBytes, abi)
      const abiDir = join(outDir, abi)
      mkdirSync(abiDir, { recursive: true })
      const dest = join(abiDir, SHERPA_JNI_LIB_NAME)
      const tmp = `${dest}.tmp-${process.pid}`
      writeFileSync(tmp, libBytes)
      renameSync(tmp, dest)
      console.log(`[sherpa-onnx] staged ${abi}/${SHERPA_JNI_LIB_NAME} (${info.size} bytes, sha256=${info.sha256})`)
    }
  }

  // Prune any non-requested ABI directory so per-ABI APKs never carry foreign ELF libraries.
  if (existsSync(outDir)) {
    const keep = new Set(opts.abis)
    for (const entry of readdirSync(outDir)) {
      if (!keep.has(entry)) {
        rmSync(join(outDir, entry), { recursive: true, force: true })
      }
    }
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main()
}
