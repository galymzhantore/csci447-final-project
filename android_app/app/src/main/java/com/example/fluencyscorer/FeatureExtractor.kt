package com.example.fluencyscorer

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

class FeatureExtractor(
    private val sampleRate: Int,
    private val numMfcc: Int = 20,
    private val frameLengthMs: Int = 25,
    private val frameShiftMs: Int = 10,
    private val numFilters: Int = 40,
    private val includeDeltas: Boolean = true,
) {
    private val preEmphasis = 0.97f
    private val frameLength = (sampleRate * frameLengthMs) / 1000
    private val frameShift = (sampleRate * frameShiftMs) / 1000
    private val fftSize = nextPowerOfTwo(frameLength)
    private val window = hammingWindow(frameLength)
    private val melFilterBank = buildMelFilterBank()
    private val dctMatrix = buildDctMatrix()
    private val deltaWindow = 2

    private fun nextPowerOfTwo(value: Int): Int {
        var n = 1
        while (n < value) n = n shl 1
        return n
    }

    private fun hammingWindow(length: Int): FloatArray =
        FloatArray(length) { idx -> (0.54 - 0.46 * cos((2 * PI * idx) / (length - 1))).toFloat() }

    private fun preEmphasize(signal: FloatArray): FloatArray {
        val emphasized = FloatArray(signal.size)
        for (i in signal.indices) {
            emphasized[i] = if (i == 0) signal[i] else signal[i] - preEmphasis * signal[i - 1]
        }
        return emphasized
    }

    private fun frame(signal: FloatArray): Array<FloatArray> {
        val numFrames = 1 + (signal.size - frameLength) / frameShift
        val frames = Array(max(1, numFrames)) { FloatArray(frameLength) }
        for (i in frames.indices) {
            val start = i * frameShift
            for (j in 0 until frameLength) {
                val idx = start + j
                frames[i][j] = if (idx < signal.size) signal[idx] * window[j] else 0f
            }
        }
        return frames
    }

    private fun magnitudeSpectrum(frame: FloatArray): FloatArray {
        val padded = FloatArray(fftSize)
        frame.copyInto(padded)
        val magnitudes = FloatArray(fftSize / 2 + 1)
        for (k in 0 until magnitudes.size) {
            var real = 0.0
            var imag = 0.0
            for (n in 0 until fftSize) {
                val angle = -2.0 * PI * k * n / fftSize
                val value = padded[n].toDouble()
                real += value * cos(angle)
                imag += value * sin(angle)
            }
            magnitudes[k] = (real * real + imag * imag).toFloat()
        }
        return magnitudes
    }

    private fun hzToMel(hz: Double) = 2595 * ln(1 + hz / 700)
    private fun melToHz(mel: Double) = 700 * (Math.exp(mel / 2595) - 1)

    private fun buildMelFilterBank(): Array<FloatArray> {
        val melMin = hzToMel(0.0)
        val melMax = hzToMel(sampleRate / 2.0)
        val melPoints = DoubleArray(numFilters + 2) { idx -> melMin + (melMax - melMin) * idx / (numFilters + 1) }
        val hzPoints = melPoints.map { melToHz(it) }
        val bins = hzPoints.map { floor((fftSize + 1) * it / sampleRate).toInt() }
        val filterBank = Array(numFilters) { FloatArray(fftSize / 2 + 1) }
        for (m in 1..numFilters) {
            val fMMinus = bins[m - 1]
            val fM = bins[m]
            val fMPlus = bins[m + 1]
            for (k in fMMinus until fM) {
                if (k in 0 until filterBank[m - 1].size) {
                    filterBank[m - 1][k] = ((k - fMMinus).toFloat() / max(1, fM - fMMinus))
                }
            }
            for (k in fM until fMPlus) {
                if (k in 0 until filterBank[m - 1].size) {
                    filterBank[m - 1][k] = ((fMPlus - k).toFloat() / max(1, fMPlus - fM))
                }
            }
        }
        return filterBank
    }

    private fun buildDctMatrix(): Array<FloatArray> {
        val matrix = Array(numMfcc) { FloatArray(numFilters) }
        for (i in 0 until numMfcc) {
            for (j in 0 until numFilters) {
                matrix[i][j] = cos(PI * i * (2 * j + 1) / (2 * numFilters)).toFloat()
            }
        }
        return matrix
    }

    private fun applyMelFilterBank(magnitude: FloatArray): FloatArray {
        val melEnergies = FloatArray(numFilters)
        for (m in 0 until numFilters) {
            var sum = 0f
            for (k in melFilterBank[m].indices) {
                sum += melFilterBank[m][k] * magnitude.getOrElse(k) { 0f }
            }
            melEnergies[m] = max(sum, 1e-8f)
        }
        return melEnergies
    }

    private fun dct(logMel: FloatArray): FloatArray {
        val coeffs = FloatArray(numMfcc)
        for (i in 0 until numMfcc) {
            var sum = 0f
            for (j in 0 until numFilters) {
                sum += dctMatrix[i][j] * logMel[j]
            }
            coeffs[i] = sum
        }
        return coeffs
    }

    private fun computeDeltas(base: Array<FloatArray>): Array<FloatArray> {
        val deltas = Array(base.size) { FloatArray(base[0].size) }
        for (t in base.indices) {
            for (c in base[0].indices) {
                var numerator = 0f
                var denominator = 0f
                for (n in 1..deltaWindow) {
                    val prev = base.getOrNull(t - n)?.get(c) ?: base.first()[c]
                    val next = base.getOrNull(t + n)?.get(c) ?: base.last()[c]
                    numerator += n * (next - prev)
                    denominator += (n * n)
                }
                deltas[t][c] = numerator / max(1f, 2f * denominator)
            }
        }
        return deltas
    }

    fun extractFeatures(pcm: FloatArray, targetFrames: Int): FloatArray {
        val emphasized = preEmphasize(pcm)
        val frames = frame(emphasized)
        val coeffFrames = Array(frames.size) { FloatArray(numMfcc) }
        for (idx in frames.indices) {
            val mag = magnitudeSpectrum(frames[idx])
            val mel = applyMelFilterBank(mag)
            val logMel = FloatArray(mel.size) { i -> ln(mel[i].toDouble()).toFloat() }
            coeffFrames[idx] = dct(logMel)
        }
        val finalFrames = if (includeDeltas) {
            val delta = computeDeltas(coeffFrames)
            val deltaDelta = computeDeltas(delta)
            combineFrames(coeffFrames, delta, deltaDelta)
        } else {
            coeffFrames
        }
        val freqBins = finalFrames[0].size
        val output = FloatArray(freqBins * targetFrames)
        for (t in 0 until targetFrames) {
            val src = finalFrames.getOrNull(t) ?: FloatArray(freqBins)
            src.copyInto(output, t * freqBins, endIndex = freqBins)
        }
        return output
    }

    private fun combineFrames(vararg groups: Array<FloatArray>): Array<FloatArray> {
        val numFrames = groups[0].size
        val combined = Array(numFrames) { FloatArray(groups.sumOf { it[0].size }) }
        for (t in 0 until numFrames) {
            var offset = 0
            for (group in groups) {
                val slice = group.getOrElse(t) { FloatArray(group[0].size) }
                slice.copyInto(combined[t], offset)
                offset += slice.size
            }
        }
        return combined
    }
}
