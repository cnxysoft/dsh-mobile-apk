package com.dsharnessmobile.shell

import java.io.File
import java.net.HttpURLConnection
import java.net.Proxy
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicInteger
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class AndroidSpeechRuntimeTest {
  private fun buildPcm16Mono16kWav(samples: ShortArray): ByteArray {
    val dataSize = samples.size * 2
    val out = ByteArray(44 + dataSize)
    val buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN)
    out[0] = 'R'.code.toByte(); out[1] = 'I'.code.toByte(); out[2] = 'F'.code.toByte(); out[3] = 'F'.code.toByte()
    buf.putInt(4, 36 + dataSize)
    out[8] = 'W'.code.toByte(); out[9] = 'A'.code.toByte(); out[10] = 'V'.code.toByte(); out[11] = 'E'.code.toByte()
    out[12] = 'f'.code.toByte(); out[13] = 'm'.code.toByte(); out[14] = 't'.code.toByte(); out[15] = ' '.code.toByte()
    buf.putInt(16, 16)
    buf.putShort(20, 1)
    buf.putShort(22, 1)
    buf.putInt(24, 16_000)
    buf.putInt(28, 32_000)
    buf.putShort(32, 2)
    buf.putShort(34, 16)
    out[36] = 'd'.code.toByte(); out[37] = 'a'.code.toByte(); out[38] = 't'.code.toByte(); out[39] = 'a'.code.toByte()
    buf.putInt(40, dataSize)
    for (i in samples.indices) buf.putShort(44 + i * 2, samples[i])
    return out
  }

  @Test
  fun waveCodecDecodesValidPcm16Mono16kAndRejectsBadInputs() {
    val wav = buildPcm16Mono16kWav(shortArrayOf(0, 16384, -32768, 32767))
    val (samples, secs) = SpeechWaveCodec.decodePcm16Mono16k(wav, 1024 * 1024)
    assertEquals(4, samples.size)
    assertEquals(0.0f, samples[0], 1e-6f)
    assertEquals(0.5f, samples[1], 1e-4f)
    assertEquals(-1.0f, samples[2], 1e-6f)
    assertEquals(4.0 / 16000.0, secs, 1e-9)

    try {
      SpeechWaveCodec.decodePcm16Mono16k(ByteArray(20), 1024)
      fail("expected SpeechInputException for short buffer")
    } catch (e: SpeechInputException) {
      assertEquals(413, e.httpStatus)
    }

    try {
      SpeechWaveCodec.validateLanguage("xx")
      fail("expected SpeechInputException for unknown language")
    } catch (e: SpeechInputException) {
      assertEquals(400, e.httpStatus)
    }
  }

  @Test
  fun onsetPaddingRestoresPrefixBeforeFirstSegment() {
    val all = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f)
    val segment = floatArrayOf(0.5f, 0.6f)
    val padded = SpeechWaveCodec.prependOnsetPadding(all, segmentStart = 4, segmentSamples = segment, onsetPaddingSamples = 2)
    assertEquals(4, padded.size)
    assertEquals(0.3f, padded[0], 1e-6f)
    assertEquals(0.4f, padded[1], 1e-6f)
    assertEquals(0.5f, padded[2], 1e-6f)
    assertEquals(0.6f, padded[3], 1e-6f)
  }

  @Test
  fun loopbackServerEnforcesBearerAuthAndRunsLoadTranscribeUnloadLifecycle() {
    val closedCount = AtomicInteger(0)
    val tmpDir = java.nio.file.Files.createTempDirectory("dsh-speech-test").toFile()
    val modelFile = File(tmpDir, "model.int8.onnx").apply { writeBytes(byteArrayOf(1)) }
    val tokensFile = File(tmpDir, "tokens.txt").apply { writeText("a 0\n") }
    val vadFile = File(tmpDir, "silero_vad.onnx").apply { writeBytes(byteArrayOf(2)) }

    val server = AndroidSpeechHttpServer { _ ->
      object : SpeechRecognizerHandle {
        override fun transcribe(samples: FloatArray, audioSeconds: Double, language: String): SpeechTranscript =
          SpeechTranscript(text = "ok:$language:${samples.size}", audioSeconds = audioSeconds, inferenceSeconds = 0.01)

        override fun close() {
          closedCount.incrementAndGet()
        }
      }
    }

    try {
      fun post(path: String, token: String?, contentType: String, body: ByteArray): Pair<Int, JSONObject> {
        val conn = URL(server.url + path).openConnection(Proxy.NO_PROXY) as HttpURLConnection
        conn.requestMethod = "POST"
        conn.doOutput = true
        conn.setRequestProperty("Content-Type", contentType)
        if (token != null) conn.setRequestProperty("Authorization", "Bearer $token")
        conn.outputStream.use { it.write(body) }
        val status = conn.responseCode
        val stream = if (status in 200..299) conn.inputStream else conn.errorStream
        val text = stream.readBytes().toString(Charsets.UTF_8)
        conn.disconnect()
        return status to JSONObject(text)
      }

      // 1. Missing/wrong auth -> 401
      val (unauthStatus, _) = post("/load", "wrong", "application/json", "{}".toByteArray())
      assertEquals(401, unauthStatus)

      // 2. Load session -> 200 { sessionId }
      val loadPayload = JSONObject()
        .put("model", modelFile.absolutePath)
        .put("tokens", tokensFile.absolutePath)
        .put("vad", vadFile.absolutePath)
        .put("threads", 2)
        .toString()
        .toByteArray()
      val (loadStatus, loadJson) = post("/load", server.token, "application/json", loadPayload)
      assertEquals(200, loadStatus)
      val sessionId = loadJson.getString("sessionId")
      assertTrue(sessionId.isNotBlank())

      // 3. Transcribe valid WAV -> 200 Transcript
      val wav = buildPcm16Mono16kWav(ShortArray(160))
      val (trStatus, trJson) = post("/session/$sessionId/transcribe?language=zh", server.token, "audio/wav", wav)
      assertEquals(200, trStatus)
      assertEquals("ok:zh:160", trJson.getString("text"))
      assertEquals(0.01, trJson.getDouble("audioSeconds"), 1e-6)

      // 4. Invalid language -> 400 invalid-input (session stays reusable!)
      val (badLangStatus, badLangJson) = post("/session/$sessionId/transcribe?language=klingon", server.token, "audio/wav", wav)
      assertEquals(400, badLangStatus)
      assertEquals("invalid-input", badLangJson.getString("code"))

      // 5. Unload -> 200 + native handle closed; subsequent transcribe -> 503
      val (unloadStatus, _) = post("/unload", server.token, "application/json", JSONObject().put("sessionId", sessionId).toString().toByteArray())
      assertEquals(200, unloadStatus)
      assertEquals(1, closedCount.get())

      val (expiredStatus, _) = post("/session/$sessionId/transcribe?language=auto", server.token, "audio/wav", wav)
      assertEquals(503, expiredStatus)
    } finally {
      server.stop()
      tmpDir.deleteRecursively()
    }
  }
}
