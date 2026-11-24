// Copyright (c)  2023  Xiaomi Corporation
// Apache-2.0 licensed Kotlin JNI bindings from k2-fsa/sherpa-onnx v1.13.8.
package com.k2fsa.sherpa.onnx

data class FeatureConfig(
    var sampleRate: Int = 16000,
    var featureDim: Int = 80,
    var dither: Float = 0.0f,
)
