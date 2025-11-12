package com.example.fluencyscorer

import android.content.Context
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

data class ModelMetadata(
    val freq: Int,
    val frames: Int,
    val inputDim: Int,
    val sampleRate: Int,
    val clipSeconds: Int,
)

data class PredictionResult(
    val label: String,
    val confidence: Float,
    val logits: FloatArray,
    val latencyMs: Double,
)

class InferenceHelper(
    private val context: Context,
    private val logger: MetricsLogger,
) {
    private val modelMetadata: ModelMetadata = loadMetadata()
    private val labels: List<String> = loadLabels()
    private val interpreter = createInterpreter()
    private val featureExtractor = FeatureExtractor(modelMetadata.sampleRate)
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * modelMetadata.inputDim).order(ByteOrder.nativeOrder())
    private val outputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * labels.size).order(ByteOrder.nativeOrder())

    private fun loadMetadata(): ModelMetadata {
        val json = context.assets.open("metadata.json").use { stream ->
            stream.bufferedReader().readText()
        }
        val obj = JSONObject(json)
        return ModelMetadata(
            freq = obj.optInt("freq", 60),
            frames = obj.optInt("frames", 100),
            inputDim = obj.optInt("input_dim", 6000),
            sampleRate = obj.optInt("sample_rate", 16000),
            clipSeconds = obj.optInt("clip_seconds", 5),
        )
    }

    private fun loadLabels(): List<String> {
        val json = context.assets.open("label_map.json").use { it.bufferedReader().readText() }
        val obj = JSONObject(json)
        val pairs = mutableListOf<Pair<Int, String>>()
        val keys = obj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            pairs.add(obj.optInt(key) to key)
        }
        return pairs.sortedBy { it.first }.map { it.second }
    }

    private fun createInterpreter(): org.tensorflow.lite.Interpreter {
        val options = org.tensorflow.lite.Interpreter.Options().apply {
            setNumThreads(2)
        }
        return org.tensorflow.lite.Interpreter(loadModelFile(), options)
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("model_fluency.tflite")
        FileInputStream(fileDescriptor.fileDescriptor).use { input ->
            val channel = input.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        }
    }

    fun infer(pcm: FloatArray): Pair<PredictionResult, MetricsEntry> {
        val features = featureExtractor.extractFeatures(pcm, modelMetadata.frames)
        inputBuffer.rewind()
        inputBuffer.asFloatBuffer().put(features)
        outputBuffer.rewind()
        val start = System.nanoTime()
        interpreter.run(inputBuffer, outputBuffer)
        val latencyMs = (System.nanoTime() - start) / 1_000_000.0
        outputBuffer.rewind()
        val logits = FloatArray(labels.size)
        outputBuffer.asFloatBuffer().get(logits)
        val probs = softmax(logits)
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val prediction = PredictionResult(
            label = labels.getOrElse(maxIdx) { "unknown" },
            confidence = probs[maxIdx],
            logits = logits,
            latencyMs = latencyMs,
        )
        val metricsEntry = logger.snapshot(prediction.label, prediction.confidence, latencyMs)
        logger.persist(metricsEntry)
        return prediction to metricsEntry
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sum = expValues.sum().coerceAtLeast(1e-6f)
        return expValues.map { it / sum }.toFloatArray()
    }

    fun metadata(): ModelMetadata = modelMetadata
}
