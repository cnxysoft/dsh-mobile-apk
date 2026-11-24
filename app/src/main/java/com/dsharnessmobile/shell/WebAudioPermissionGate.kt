package com.dsharnessmobile.shell

import android.Manifest
import android.content.pm.PackageManager
import android.webkit.PermissionRequest
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat

/**
 * Pure policy for WebView microphone permission requests (`getUserMedia({ audio: ... })`).
 *
 * Kept separate from [WebAudioPermissionController] so every origin and resource
 * boundary can be verified directly by JVM unit tests without Robolectric.
 */
internal object WebAudioPermissionPolicy {
  /** Mirrors `PermissionRequest.RESOURCE_AUDIO_CAPTURE` so JVM tests need no Android stub. */
  const val RESOURCE_AUDIO_CAPTURE = "android.webkit.resource.AUDIO_CAPTURE"

  /**
   * Allow audio capture only when both the requesting WebView origin and the
   * tracked top-level document belong to our local DSH engine origin.
   */
  fun isTrustedRequest(
    requestOrigin: String?,
    topFrameUrl: String?,
    originChecker: (String) -> Boolean = EngineProbe::isEngineOrigin,
  ): Boolean {
    val origin = requestOrigin?.trim().orEmpty()
    val top = topFrameUrl?.trim().orEmpty()
    if (origin.isEmpty() || top.isEmpty()) return false
    return originChecker(origin) && originChecker(top)
  }

  /**
   * Narrow the requested WebView resources strictly to audio capture.
   *
   * Returns `arrayOf(RESOURCE_AUDIO_CAPTURE)` only when the page asks for audio
   * capture and nothing else (never silently approves camera or protected media).
   */
  fun grantableAudioResources(requested: Array<String>?): Array<String>? {
    if (requested == null || requested.isEmpty()) return null
    val unique = requested.map { it.trim() }.filter { it.isNotEmpty() }.toSet()
    return if (unique.size == 1 && unique.contains(RESOURCE_AUDIO_CAPTURE)) {
      arrayOf(RESOURCE_AUDIO_CAPTURE)
    } else {
      null
    }
  }
}

/**
 * Activity-scoped controller for WebView `onPermissionRequest` audio capture.
 *
 * Must be constructed as a field on [MainActivity] so `registerForActivityResult`
 * runs before Lifecycle `STARTED`. Only ever attached to the engine WebView in
 * [MainActivity] (never [BrowserHost]).
 */
internal class WebAudioPermissionController(private val activity: MainActivity) {
  private var pendingRequest: PermissionRequest? = null
  private var pendingResources: Array<String>? = null

  private val micPermissionLauncher: ActivityResultLauncher<String> =
    activity.registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      val req = pendingRequest
      val resources = pendingResources
      pendingRequest = null
      pendingResources = null
      if (req == null) return@registerForActivityResult
      if (granted && resources != null && !activity.isFinishing && !activity.isDestroyed) {
        LogCollector.log("dsh-voice-perm", "runtime RECORD_AUDIO granted -> granting WebView audio capture")
        runCatching { req.grant(resources) }
      } else {
        LogCollector.log("dsh-voice-perm", "runtime RECORD_AUDIO denied or activity closing -> denying WebView audio capture")
        runCatching { req.deny() }
      }
    }

  fun onPermissionRequest(request: PermissionRequest) {
    if (activity.isFinishing || activity.isDestroyed) {
      runCatching { request.deny() }
      return
    }
    val origin = request.origin?.toString()
    val topUrl = activity.trustedTopFrameUrl
    if (!WebAudioPermissionPolicy.isTrustedRequest(origin, topUrl)) {
      LogCollector.log("dsh-voice-perm", "denied untrusted origin=" + (origin ?: "null"))
      runCatching { request.deny() }
      return
    }
    val grantable = WebAudioPermissionPolicy.grantableAudioResources(request.resources)
    if (grantable == null) {
      LogCollector.log("dsh-voice-perm", "denied non-audio resources=" + request.resources?.joinToString(","))
      runCatching { request.deny() }
      return
    }
    val hasRuntimeGrant = ContextCompat.checkSelfPermission(
      activity,
      Manifest.permission.RECORD_AUDIO,
    ) == PackageManager.PERMISSION_GRANTED
    if (hasRuntimeGrant) {
      runCatching { request.grant(grantable) }
      return
    }
    pendingRequest?.let { superseded ->
      runCatching { superseded.deny() }
    }
    pendingRequest = request
    pendingResources = grantable
    try {
      micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    } catch (t: Throwable) {
      pendingRequest = null
      pendingResources = null
      LogCollector.log("dsh-voice-perm", "launch RECORD_AUDIO failed: " + (t.message ?: t.javaClass.simpleName))
      runCatching { request.deny() }
    }
  }

  fun onPermissionRequestCanceled(request: PermissionRequest) {
    if (pendingRequest === request) {
      pendingRequest = null
      pendingResources = null
    }
  }

  fun release() {
    val req = pendingRequest
    pendingRequest = null
    pendingResources = null
    if (req != null) runCatching { req.deny() }
  }
}
