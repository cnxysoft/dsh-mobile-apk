// Copyright (c)  2023  Xiaomi Corporation
// Apache-2.0 licensed Kotlin JNI bindings from k2-fsa/sherpa-onnx v1.13.8.
package com.k2fsa.sherpa.onnx

data class QnnConfig(
    var backendLib: String = "",
    var contextBinary: String = "",
    var systemLib: String = "",
)
