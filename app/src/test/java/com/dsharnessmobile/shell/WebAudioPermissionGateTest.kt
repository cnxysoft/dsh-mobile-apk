package com.dsharnessmobile.shell

import java.io.File
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WebAudioPermissionGateTest {
  private val repoRoot = File(System.getProperty("user.dir") ?: ".").let { dir ->
    if (File(dir, "app/src/main/AndroidManifest.xml").isFile) dir else dir.parentFile
  }

  @Test
  fun trustedRequestRequiresBothOriginAndTopFrameOnEngineOrigin() {
    val engineOnly: (String) -> Boolean = { url ->
      url.startsWith("http://127.0.0.1:4145") || url.startsWith("http://localhost:4145")
    }
    assertTrue(
      WebAudioPermissionPolicy.isTrustedRequest(
        "http://127.0.0.1:4145/",
        "http://127.0.0.1:4145/sessions/abc",
        engineOnly,
      ),
    )
    assertFalse(
      WebAudioPermissionPolicy.isTrustedRequest(
        "https://evil.example/",
        "http://127.0.0.1:4145/",
        engineOnly,
      ),
    )
    assertFalse(
      WebAudioPermissionPolicy.isTrustedRequest(
        "http://127.0.0.1:4145/",
        "https://evil.example/",
        engineOnly,
      ),
    )
    assertFalse(WebAudioPermissionPolicy.isTrustedRequest(null, "http://127.0.0.1:4145/", engineOnly))
    assertFalse(WebAudioPermissionPolicy.isTrustedRequest("http://127.0.0.1:4145/", "", engineOnly))
  }

  @Test
  fun grantableAudioResourcesAllowsOnlyAudioCapture() {
    val audioOnly = WebAudioPermissionPolicy.grantableAudioResources(
      arrayOf(WebAudioPermissionPolicy.RESOURCE_AUDIO_CAPTURE),
    )
    assertArrayEquals(arrayOf(WebAudioPermissionPolicy.RESOURCE_AUDIO_CAPTURE), audioOnly)

    assertNull(WebAudioPermissionPolicy.grantableAudioResources(null))
    assertNull(WebAudioPermissionPolicy.grantableAudioResources(emptyArray()))
    assertNull(
      WebAudioPermissionPolicy.grantableAudioResources(
        arrayOf("android.webkit.resource.VIDEO_CAPTURE"),
      ),
    )
    assertNull(
      WebAudioPermissionPolicy.grantableAudioResources(
        arrayOf(
          WebAudioPermissionPolicy.RESOURCE_AUDIO_CAPTURE,
          "android.webkit.resource.VIDEO_CAPTURE",
        ),
      ),
    )
  }

  @Test
  fun manifestAndMainActivityWireMicrophonePermissionWithoutExposingBrowserHost() {
    val manifest = File(repoRoot, "app/src/main/AndroidManifest.xml").readText()
    assertTrue("AndroidManifest must declare RECORD_AUDIO", manifest.contains("android.permission.RECORD_AUDIO"))
    assertTrue(
      "AndroidManifest must declare MODIFY_AUDIO_SETTINGS for Chromium WebView AudioManagerAndroid",
      manifest.contains("android.permission.MODIFY_AUDIO_SETTINGS"),
    )
    assertTrue(
      "AndroidManifest must declare optional microphone hardware feature",
      manifest.contains("android.hardware.microphone") && manifest.contains("android:required=\"false\""),
    )

    val mainActivity = File(repoRoot, "app/src/main/java/com/dsharnessmobile/shell/MainActivity.kt").readText()
    assertTrue(mainActivity.contains("webAudioPermissionController = WebAudioPermissionController(this)"))
    assertTrue(mainActivity.contains("override fun onPermissionRequest("))
    assertTrue(mainActivity.contains("webAudioPermissionController.release()"))

    val browserHost = File(repoRoot, "app/src/main/java/com/dsharnessmobile/shell/BrowserHost.kt").readText()
    assertFalse(
      "BrowserHost must never grant RESOURCE_AUDIO_CAPTURE to arbitrary web pages",
      browserHost.contains("RESOURCE_AUDIO_CAPTURE"),
    )
  }
}
