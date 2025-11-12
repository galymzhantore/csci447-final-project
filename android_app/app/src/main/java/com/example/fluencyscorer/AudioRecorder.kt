package com.example.fluencyscorer

import android.content.ContentResolver
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.Uri
import java.io.BufferedInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min

class AudioRecorder(
    private val context: Context,
    private val sampleRate: Int,
    private val clipSeconds: Int = 5,
) {
    private val executor = Executors.newSingleThreadExecutor()
    private val isRecording = AtomicBoolean(false)
    private var audioRecord: AudioRecord? = null
    private val maxSamples = clipSeconds * sampleRate
    private val buffer = ArrayList<Float>(maxSamples)

    fun startRecording() {
        if (isRecording.get()) return
        buffer.clear()
        val minBuffer = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            max(minBuffer, maxSamples),
        )
        audioRecord?.startRecording()
        isRecording.set(true)
        executor.execute {
            val temp = ShortArray(minBuffer)
            while (isRecording.get()) {
                val read = audioRecord?.read(temp, 0, temp.size) ?: break
                if (read > 0) {
                    for (i in 0 until read) {
                        if (buffer.size >= maxSamples) {
                            isRecording.set(false)
                            break
                        }
                        buffer.add(temp[i] / 32768f)
                    }
                }
            }
        }
    }

    fun stopRecording(): FloatArray {
        if (!isRecording.getAndSet(false)) {
            return FloatArray(0)
        }
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        return normalizeLength(buffer.toFloatArray())
    }

    fun readFromUri(uri: Uri): FloatArray {
        val resolver: ContentResolver = context.contentResolver
        resolver.openInputStream(uri)?.use { stream ->
            val buffered = BufferedInputStream(stream)
            val header = ByteArray(44)
            buffered.read(header)
            val pcmBytes = buffered.readBytes()
            val shortBuffer = ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            val floats = FloatArray(shortBuffer.remaining())
            for (i in floats.indices) {
                floats[i] = shortBuffer.get().toFloat() / 32768f
            }
            return normalizeLength(floats)
        }
        return FloatArray(0)
    }

    private fun normalizeLength(samples: FloatArray): FloatArray {
        val out = FloatArray(maxSamples)
        val copyLen = min(samples.size, maxSamples)
        System.arraycopy(samples, 0, out, 0, copyLen)
        return out
    }
}
