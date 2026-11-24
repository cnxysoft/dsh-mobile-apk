package com.dsharnessmobile.shell

import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSenseVoiceModelConfig
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.net.URLDecoder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Locale
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import org.json.JSONObject

/** Configuration for one Android native SenseVoice + Silero VAD session. */
internal data class SpeechSessionConfig(
  val model: String,
  val tokens: String,
  val vad: String,
  val threads: Int = 2,
  val segmentSeconds: Double = 30.0,
  val vadThreshold: Double = 0.5,
  val minSpeechSeconds: Double = 0.25,
  val minSilenceSeconds: Double = 0.5,
  val vadOnsetPaddingSeconds: Double = 0.3,
  val maxAudioBytes: Int = 4 * 1024 * 1024,
)

/** Recognition result matching `@deepseek-ai/dsh-experimental-speech-to-text/types` Transcript. */
internal data class SpeechTranscript(
  val text: String,
  val audioSeconds: Double,
  val inferenceSeconds: Double,
)

/** Client-side input error (maps to HTTP 400/413 with `code: "invalid-input"`). */
internal class SpeechInputException(
  message: String,
  val httpStatus: Int = 400,
) : IllegalArgumentException(message)

internal interface SpeechRecognizerHandle : AutoCloseable {
  fun transcribe(samples: FloatArray, audioSeconds: Double, language: String): SpeechTranscript
}

internal fun interface SpeechRecognizerFactory {
  fun create(config: SpeechSessionConfig): SpeechRecognizerHandle
}

/** Pure WAV / PCM16 / onset-padding helpers shared by production and JVM unit tests. */
internal object SpeechWaveCodec {
  val SUPPORTED_LANGUAGES: Set<String> = setOf("auto", "zh", "en", "yue", "ja", "ko")
  const val SAMPLE_RATE = 16_000
  const val VAD_WINDOW_SAMPLES = 512

  fun validateLanguage(language: String): String {
    if (language !in SUPPORTED_LANGUAGES) {
      throw SpeechInputException("Unsupported SenseVoice language", httpStatus = 400)
    }
    return language
  }

  /**
   * Decode a canonical 16 kHz mono PCM16 WAV into normalized Float32 samples in [-1, 1].
   */
  fun decodePcm16Mono16k(wav: ByteArray, maxAudioBytes: Int): Pair<FloatArray, Double> {
    if (wav.size < 46 || wav.size > maxAudioBytes) {
      throw SpeechInputException("Invalid speech audio size", httpStatus = 413)
    }
    if (asciiAt(wav, 0, 4) != "RIFF" || asciiAt(wav, 8, 4) != "WAVE") {
      throw SpeechInputException("Invalid speech WAV: missing RIFF/WAVE header", httpStatus = 400)
    }
    val buf = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
    var fmtSeen = false
    var audioFormat = 0
    var channels = 0
    var sampleRate = 0
    var bitsPerSample = 0
    var dataOffset = -1
    var dataSize = -1

    var pos = 12
    while (pos + 8 <= wav.size) {
      val chunkId = asciiAt(wav, pos, 4)
      val chunkSize = buf.getInt(pos + 4).toLong() and 0xFFFF_FFFFL
      val chunkDataStart = pos + 8
      if (chunkSize < 0 || chunkDataStart + chunkSize > wav.size) {
        throw SpeechInputException("Invalid speech WAV: truncated chunk $chunkId", httpStatus = 400)
      }
      if (chunkId == "fmt ") {
        if (chunkSize < 16) {
          throw SpeechInputException("Invalid speech WAV: short fmt chunk", httpStatus = 400)
        }
        audioFormat = buf.getShort(chunkDataStart).toInt() and 0xFFFF
        channels = buf.getShort(chunkDataStart + 2).toInt() and 0xFFFF
        sampleRate = buf.getInt(chunkDataStart + 4)
        bitsPerSample = buf.getShort(chunkDataStart + 14).toInt() and 0xFFFF
        fmtSeen = true
      } else if (chunkId == "data") {
        dataOffset = chunkDataStart
        dataSize = chunkSize.toInt()
        break
      }
      val padded = chunkSize + (chunkSize and 1L)
      val next = chunkDataStart + padded
      if (next <= pos || next > wav.size) break
      pos = next.toInt()
    }

    if (!fmtSeen || dataOffset < 0 || dataSize < 2 || (dataSize and 1) != 0) {
      throw SpeechInputException("Invalid speech WAV: missing fmt or data chunk", httpStatus = 400)
    }
    if (audioFormat != 1 || channels != 1 || sampleRate != SAMPLE_RATE || bitsPerSample != 16) {
      throw SpeechInputException(
        "Invalid speech WAV: expected 16kHz mono PCM16 (got format=$audioFormat ch=$channels rate=$sampleRate bits=$bitsPerSample)",
        httpStatus = 400,
      )
    }
    val sampleCount = dataSize / 2
    val samples = FloatArray(sampleCount)
    var offset = dataOffset
    for (i in 0 until sampleCount) {
      val s = buf.getShort(offset).toInt()
      samples[i] = s / 32768.0f
      offset += 2
    }
    val audioSeconds = sampleCount.toDouble() / SAMPLE_RATE.toDouble()
    return samples to audioSeconds
  }

  /**
   * Restore leading onset samples before the first VAD speech segment so word-initial
   * consonants are not clipped (mirrors upstream SenseVoice `inference.ts`).
   */
  fun prependOnsetPadding(
    allSamples: FloatArray,
    segmentStart: Int,
    segmentSamples: FloatArray,
    onsetPaddingSamples: Int,
  ): FloatArray {
    if (onsetPaddingSamples <= 0 || segmentStart <= 0) return segmentSamples
    val clampedStart = min(segmentStart, allSamples.size)
    val from = max(0, clampedStart - onsetPaddingSamples)
    if (from >= clampedStart) return segmentSamples
    val prefixLen = clampedStart - from
    val restored = FloatArray(prefixLen + segmentSamples.size)
    System.arraycopy(allSamples, from, restored, 0, prefixLen)
    System.arraycopy(segmentSamples, 0, restored, prefixLen, segmentSamples.size)
    return restored
  }

  private fun asciiAt(bytes: ByteArray, offset: Int, length: Int): String =
    String(bytes, offset, length, Charsets.US_ASCII)
}

/** Native JNI backend powered by `com.k2fsa.sherpa.onnx` (`libsherpa-onnx-jni.so` v1.13.8). */
internal object SherpaOnnxSpeechBackend : SpeechRecognizerFactory {
  override fun create(config: SpeechSessionConfig): SpeechRecognizerHandle {
    val recognizerConfig = OfflineRecognizerConfig(
      featConfig = FeatureConfig(sampleRate = SpeechWaveCodec.SAMPLE_RATE, featureDim = 80),
      modelConfig = OfflineModelConfig(
        senseVoice = OfflineSenseVoiceModelConfig(
          model = config.model,
          language = "auto",
          useInverseTextNormalization = true,
        ),
        tokens = config.tokens,
        numThreads = config.threads.coerceIn(1, 8),
        provider = "cpu",
        debug = false,
      ),
    )
    val vadConfig = VadModelConfig(
      sileroVadModelConfig = SileroVadModelConfig(
        model = config.vad,
        threshold = config.vadThreshold.toFloat(),
        minSilenceDuration = config.minSilenceSeconds.toFloat(),
        minSpeechDuration = config.minSpeechSeconds.toFloat(),
        maxSpeechDuration = config.segmentSeconds.toFloat(),
        windowSize = SpeechWaveCodec.VAD_WINDOW_SAMPLES,
      ),
      sampleRate = SpeechWaveCodec.SAMPLE_RATE,
      numThreads = config.threads.coerceIn(1, 4),
      provider = "cpu",
      debug = false,
    )
    val recognizer = OfflineRecognizer(assetManager = null, config = recognizerConfig)
    val detector = try {
      Vad(assetManager = null, config = vadConfig)
    } catch (t: Throwable) {
      runCatching { recognizer.release() }
      throw t
    }
    return NativeHandle(recognizer, detector, recognizerConfig, config.vadOnsetPaddingSeconds)
  }

  private class NativeHandle(
    private val recognizer: OfflineRecognizer,
    private val detector: Vad,
    private val recognizerConfig: OfflineRecognizerConfig,
    private val vadOnsetPaddingSeconds: Double,
  ) : SpeechRecognizerHandle {
    private val closed = AtomicBoolean(false)

    override fun transcribe(samples: FloatArray, audioSeconds: Double, language: String): SpeechTranscript {
      check(!closed.get()) { "Speech recognizer has been unloaded" }
      recognizerConfig.modelConfig.senseVoice.language = language
      recognizer.setConfig(recognizerConfig)
      detector.reset()
      val startedNs = System.nanoTime()
      val texts = mutableListOf<String>()
      val onsetPadding = (vadOnsetPaddingSeconds * SpeechWaveCodec.SAMPLE_RATE).roundToInt().coerceAtLeast(0)
      var isFirstSegment = true

      fun drain() {
        while (!detector.empty()) {
          val segment = detector.front()
          var speech = segment.samples
          if (isFirstSegment) {
            isFirstSegment = false
            speech = SpeechWaveCodec.prependOnsetPadding(samples, segment.start, speech, onsetPadding)
          }
          val stream = recognizer.createStream()
          stream.use { s ->
            s.acceptWaveform(speech, SpeechWaveCodec.SAMPLE_RATE)
            recognizer.decode(s)
            val piece = recognizer.getResult(s).text.trim()
            if (piece.isNotEmpty()) texts.add(piece)
          }
          detector.pop()
        }
      }

      var offset = 0
      while (offset < samples.size) {
        val end = min(offset + SpeechWaveCodec.VAD_WINDOW_SAMPLES, samples.size)
        detector.acceptWaveform(samples.copyOfRange(offset, end))
        drain()
        offset = end
      }
      detector.flush()
      drain()

      val inferenceSeconds = (System.nanoTime() - startedNs) / 1_000_000_000.0
      return SpeechTranscript(
        text = texts.joinToString(" ").trim(),
        audioSeconds = audioSeconds,
        inferenceSeconds = inferenceSeconds,
      )
    }

    override fun close() {
      if (closed.compareAndSet(false, true)) {
        runCatching { detector.release() }
        runCatching { recognizer.release() }
      }
    }
  }
}

/**
 * Authenticated loopback HTTP server for Android native SenseVoice inference.
 *
 * Security & resource invariants:
 * - Binds strictly to `127.0.0.1:0` with a 64-hex per-process bearer secret.
 * - Loads `libsherpa-onnx-jni.so` and ONNX models lazily on `POST /load`.
 * - Serializes load / transcribe / unload under one lock so JNI objects are
 *   never touched concurrently or freed mid-inference.
 * - Releases native memory immediately on `POST /unload`, `unloadCurrent()`,
 *   or `stop()`.
 */
internal class AndroidSpeechHttpServer(
  private val factory: SpeechRecognizerFactory = SherpaOnnxSpeechBackend,
) {
  companion object {
    private const val TAG = "dsh-speech"
    private const val MAX_HEADER_BYTES = 16 * 1024
    private const val MAX_LOAD_BODY_BYTES = 64 * 1024
    private const val ABSOLUTE_MAX_AUDIO_BYTES = 32 * 1024 * 1024
    private const val SOCKET_TIMEOUT_MS = 30_000

    private fun generateToken(): String {
      val bytes = ByteArray(32)
      SecureRandom().nextBytes(bytes)
      val sb = StringBuilder(64)
      for (b in bytes) sb.append(String.format(Locale.ROOT, "%02x", b.toInt() and 0xFF))
      return sb.toString()
    }
  }

  private data class ActiveSession(
    val id: String,
    val config: SpeechSessionConfig,
    val handle: SpeechRecognizerHandle,
  )

  private val serverSocket = ServerSocket(0, 8, InetAddress.getByName("127.0.0.1"))
  val port: Int = serverSocket.localPort
  val url: String = "http://127.0.0.1:$port"
  val token: String = generateToken()
  private val expectedAuthBytes = ("Bearer $token").toByteArray(Charsets.UTF_8)

  private val stopped = AtomicBoolean(false)
  private val stateLock = Any()
  private var activeSession: ActiveSession? = null

  private val workerExecutor = Executors.newSingleThreadExecutor { task ->
    Thread(task, "dsh-speech-worker").apply { isDaemon = true }
  }
  private val acceptThread = Thread({ acceptLoop() }, "dsh-speech-accept").apply {
    isDaemon = true
    start()
  }

  private fun acceptLoop() {
    while (!stopped.get()) {
      val socket = try {
        serverSocket.accept()
      } catch (_: SocketException) {
        break
      } catch (t: Throwable) {
        if (!stopped.get()) {
          Log.w(TAG, "accept error: " + (t.message ?: t.javaClass.simpleName))
        }
        continue
      }
      try {
        workerExecutor.execute { handleSocket(socket) }
      } catch (_: Throwable) {
        runCatching { socket.close() }
      }
    }
  }

  private fun handleSocket(socket: Socket) {
    socket.use { sock ->
      runCatching { sock.soTimeout = SOCKET_TIMEOUT_MS }
      val input = BufferedInputStream(sock.getInputStream())
      val output = sock.getOutputStream()
      try {
        val head = readHttpHead(input) ?: return
        val lines = head.split("\r\n")
        if (lines.isEmpty()) {
          writeJson(output, 400, JSONObject().put("error", "Malformed request"))
          return
        }
        val reqParts = lines[0].split(" ")
        if (reqParts.size < 2) {
          writeJson(output, 400, JSONObject().put("error", "Malformed request line"))
          return
        }
        val method = reqParts[0].uppercase(Locale.ROOT)
        val rawTarget = reqParts[1]
        val headers = HashMap<String, String>()
        for (i in 1 until lines.size) {
          val colon = lines[i].indexOf(':')
          if (colon > 0) {
            val k = lines[i].substring(0, colon).trim().lowercase(Locale.ROOT)
            val v = lines[i].substring(colon + 1).trim()
            headers[k] = v
          }
        }
        val auth = headers["authorization"].orEmpty().toByteArray(Charsets.UTF_8)
        if (!MessageDigest.isEqual(auth, expectedAuthBytes)) {
          writeJson(output, 401, JSONObject().put("error", "Unauthorized"))
          return
        }
        val qIdx = rawTarget.indexOf('?')
        val path = if (qIdx >= 0) rawTarget.substring(0, qIdx) else rawTarget
        val query = if (qIdx >= 0) rawTarget.substring(qIdx + 1) else ""

        if (method != "POST") {
          writeJson(output, 404, JSONObject().put("error", "Unknown endpoint"))
          return
        }
        when {
          path == "/load" -> handleLoad(input, headers, output)
          path == "/unload" -> handleUnload(input, headers, output)
          path.startsWith("/session/") && path.endsWith("/transcribe") -> {
            val sessionId = path.removePrefix("/session/").removeSuffix("/transcribe")
            handleTranscribe(sessionId, query, input, headers, output)
          }
          else -> writeJson(output, 404, JSONObject().put("error", "Unknown endpoint"))
        }
      } catch (e: SpeechInputException) {
        runCatching {
          writeJson(
            output,
            e.httpStatus,
            JSONObject().put("error", e.message ?: "Invalid speech input").put("code", "invalid-input"),
          )
        }
      } catch (t: Throwable) {
        val msg = t.message ?: t.javaClass.simpleName
        Log.w(TAG, "speech request failed: $msg")
        runCatching {
          writeJson(output, 500, JSONObject().put("error", msg))
        }
      }
    }
  }

  private fun handleLoad(input: InputStream, headers: Map<String, String>, output: java.io.OutputStream) {
    val body = readBody(input, headers, MAX_LOAD_BODY_BYTES)
    val json = JSONObject(String(body, Charsets.UTF_8))
    val model = json.optString("model", "").trim()
    val tokens = json.optString("tokens", "").trim()
    val vad = json.optString("vad", "").trim()
    if (model.isEmpty() || tokens.isEmpty() || vad.isEmpty()) {
      writeJson(output, 400, JSONObject().put("error", "Missing model, tokens, or vad path"))
      return
    }
    for ((label, filePath) in listOf("model" to model, "tokens" to tokens, "vad" to vad)) {
      if (!File(filePath).isFile) {
        writeJson(output, 400, JSONObject().put("error", "Speech asset missing ($label): $filePath"))
        return
      }
    }
    val config = SpeechSessionConfig(
      model = model,
      tokens = tokens,
      vad = vad,
      threads = json.optInt("threads", 2).coerceIn(1, 8),
      segmentSeconds = json.optDouble("segmentSeconds", 30.0).coerceIn(1.0, 120.0),
      vadThreshold = json.optDouble("vadThreshold", 0.5).coerceIn(0.0, 1.0),
      minSpeechSeconds = json.optDouble("minSpeechSeconds", 0.25).coerceAtLeast(0.0),
      minSilenceSeconds = json.optDouble("minSilenceSeconds", 0.5).coerceAtLeast(0.01),
      vadOnsetPaddingSeconds = json.optDouble("vadOnsetPaddingSeconds", 0.3).coerceIn(0.0, 1.0),
      maxAudioBytes = json.optInt("maxAudioBytes", 4 * 1024 * 1024).coerceIn(46, ABSOLUTE_MAX_AUDIO_BYTES),
    )
    val sessionId = synchronized(stateLock) {
      activeSession?.let { old ->
        activeSession = null
        runCatching { old.handle.close() }
      }
      val handle = factory.create(config)
      val id = UUID.randomUUID().toString()
      activeSession = ActiveSession(id = id, config = config, handle = handle)
      id
    }
    writeJson(output, 200, JSONObject().put("sessionId", sessionId))
  }

  private fun handleUnload(input: InputStream, headers: Map<String, String>, output: java.io.OutputStream) {
    val body = readBody(input, headers, MAX_LOAD_BODY_BYTES)
    val requestedId = if (body.isNotEmpty()) {
      runCatching { JSONObject(String(body, Charsets.UTF_8)).optString("sessionId", "") }.getOrDefault("")
    } else ""
    synchronized(stateLock) {
      val current = activeSession
      if (current != null && (requestedId.isEmpty() || current.id == requestedId)) {
        activeSession = null
        runCatching { current.handle.close() }
      }
    }
    writeJson(output, 200, JSONObject().put("ok", true))
  }

  private fun handleTranscribe(
    sessionId: String,
    query: String,
    input: InputStream,
    headers: Map<String, String>,
    output: java.io.OutputStream,
  ) {
    val language = SpeechWaveCodec.validateLanguage(parseQueryParam(query, "language") ?: "auto")
    val session = synchronized(stateLock) {
      val current = activeSession
      if (current != null && current.id == sessionId) current else null
    }
    if (session == null) {
      writeJson(output, 503, JSONObject().put("error", "Speech session is not loaded"))
      return
    }
    val maxBytes = session.config.maxAudioBytes
    val declaredLen = headers["content-length"]?.toLongOrNull()
    if (declaredLen != null && (declaredLen < 46L || declaredLen > maxBytes.toLong())) {
      throw SpeechInputException("Invalid speech audio size", httpStatus = 413)
    }
    val audioBytes = readBody(input, headers, maxBytes)
    val (samples, audioSeconds) = SpeechWaveCodec.decodePcm16Mono16k(audioBytes, maxBytes)
    val transcript = synchronized(stateLock) {
      val current = activeSession
      if (current == null || current.id != sessionId) {
        throw IllegalStateException("Speech session was unloaded")
      }
      current.handle.transcribe(samples, audioSeconds, language)
    }
    writeJson(
      output,
      200,
      JSONObject()
        .put("text", transcript.text)
        .put("audioSeconds", transcript.audioSeconds)
        .put("inferenceSeconds", transcript.inferenceSeconds),
    )
  }

  fun unloadCurrent() {
    synchronized(stateLock) {
      val current = activeSession
      activeSession = null
      if (current != null) {
        runCatching { current.handle.close() }
      }
    }
  }

  fun stop() {
    if (!stopped.compareAndSet(false, true)) return
    runCatching { serverSocket.close() }
    unloadCurrent()
    workerExecutor.shutdownNow()
    runCatching { acceptThread.interrupt() }
  }

  private fun readHttpHead(input: InputStream): String? {
    val out = ByteArrayOutputStream(512)
    var state = 0
    while (out.size() < MAX_HEADER_BYTES) {
      val b = input.read()
      if (b < 0) return if (out.size() == 0) null else out.toString("ISO-8859-1")
      out.write(b)
      state = when {
        state == 0 && b == '\r'.code -> 1
        state == 1 && b == '\n'.code -> 2
        state == 2 && b == '\r'.code -> 3
        state == 3 && b == '\n'.code -> return out.toString("ISO-8859-1").removeSuffix("\r\n\r\n")
        b == '\r'.code -> 1
        else -> 0
      }
    }
    throw SpeechInputException("HTTP headers exceed limit", httpStatus = 413)
  }

  private fun readBody(input: InputStream, headers: Map<String, String>, maxBytes: Int): ByteArray {
    val transferEncoding = headers["transfer-encoding"].orEmpty().lowercase(Locale.ROOT)
    if (transferEncoding.contains("chunked")) {
      return readChunkedBody(input, maxBytes)
    }
    val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
    if (contentLength < 0 || contentLength > maxBytes) {
      throw SpeechInputException("Invalid speech audio size", httpStatus = 413)
    }
    if (contentLength == 0) return ByteArray(0)
    val data = ByteArray(contentLength)
    var readTotal = 0
    while (readTotal < contentLength) {
      val n = input.read(data, readTotal, contentLength - readTotal)
      if (n < 0) throw SpeechInputException("Truncated HTTP body", httpStatus = 400)
      readTotal += n
    }
    return data
  }

  private fun readChunkedBody(input: InputStream, maxBytes: Int): ByteArray {
    val out = ByteArrayOutputStream()
    while (true) {
      val sizeLine = readCrLfLine(input) ?: throw SpeechInputException("Truncated chunked body", httpStatus = 400)
      val semi = sizeLine.indexOf(';')
      val hex = (if (semi >= 0) sizeLine.substring(0, semi) else sizeLine).trim()
      val chunkSize = hex.toIntOrNull(16) ?: throw SpeechInputException("Invalid chunk size", httpStatus = 400)
      if (chunkSize == 0) {
        readCrLfLine(input)
        return out.toByteArray()
      }
      if (out.size() + chunkSize > maxBytes) {
        throw SpeechInputException("Invalid speech audio size", httpStatus = 413)
      }
      val buf = ByteArray(chunkSize)
      var offset = 0
      while (offset < chunkSize) {
        val n = input.read(buf, offset, chunkSize - offset)
        if (n < 0) throw SpeechInputException("Truncated chunk data", httpStatus = 400)
        offset += n
      }
      out.write(buf)
      readCrLfLine(input)
    }
  }

  private fun readCrLfLine(input: InputStream): String? {
    val out = ByteArrayOutputStream(32)
    while (out.size() < 256) {
      val b = input.read()
      if (b < 0) return null
      if (b == '\n'.code) {
        val raw = out.toByteArray()
        val end = if (raw.isNotEmpty() && raw[raw.size - 1] == '\r'.code.toByte()) raw.size - 1 else raw.size
        return String(raw, 0, end, Charsets.US_ASCII)
      }
      out.write(b)
    }
    throw SpeechInputException("Chunk header line too long", httpStatus = 400)
  }

  private fun parseQueryParam(query: String, key: String): String? {
    if (query.isEmpty()) return null
    for (part in query.split('&')) {
      val eq = part.indexOf('=')
      val rawKey = if (eq >= 0) part.substring(0, eq) else part
      val decodedKey = runCatching { URLDecoder.decode(rawKey, "UTF-8") }.getOrDefault(rawKey)
      if (decodedKey == key) {
        val rawVal = if (eq >= 0) part.substring(eq + 1) else ""
        return runCatching { URLDecoder.decode(rawVal, "UTF-8") }.getOrDefault(rawVal)
      }
    }
    return null
  }

  private fun writeJson(output: java.io.OutputStream, status: Int, json: JSONObject) {
    val reason = when (status) {
      200 -> "OK"
      400 -> "Bad Request"
      401 -> "Unauthorized"
      404 -> "Not Found"
      413 -> "Payload Too Large"
      503 -> "Service Unavailable"
      else -> "Internal Server Error"
    }
    val payload = json.toString().toByteArray(Charsets.UTF_8)
    val header = ("HTTP/1.1 $status $reason\r\n" +
      "Content-Type: application/json\r\n" +
      "Content-Length: ${payload.size}\r\n" +
      "Connection: close\r\n\r\n").toByteArray(Charsets.US_ASCII)
    output.write(header)
    output.write(payload)
    output.flush()
  }
}

/** Process-wide lazy bridge for Android native SenseVoice transcription. */
internal object AndroidSpeechRuntime {
  data class Endpoint(val url: String, val token: String)

  private val lock = Any()
  @Volatile private var server: AndroidSpeechHttpServer? = null

  fun ensureStarted(): Endpoint? = synchronized(lock) {
    val existing = server
    if (existing != null) return Endpoint(existing.url, existing.token)
    return try {
      val created = AndroidSpeechHttpServer()
      server = created
      LogCollector.log("dsh-speech", "loopback speech bridge listening on 127.0.0.1:" + created.port)
      Endpoint(created.url, created.token)
    } catch (t: Throwable) {
      LogCollector.log("dsh-speech", "failed to bind speech bridge: " + (t.message ?: t.javaClass.simpleName))
      null
    }
  }

  /** Environment variables injected into the DSH engine process via `EngineManager.shellEnv()`. */
  fun envVars(): Map<String, String> {
    val ep = ensureStarted() ?: return emptyMap()
    return mapOf(
      "DSH_ANDROID_SPEECH_URL" to ep.url,
      "DSH_ANDROID_SPEECH_TOKEN" to ep.token,
    )
  }

  fun unload() {
    synchronized(lock) { server }?.unloadCurrent()
  }

  fun stop() {
    val current = synchronized(lock) {
      val s = server
      server = null
      s
    }
    current?.stop()
  }
}
