package com.example.fluencyscorer

import android.content.Context
import android.os.BatteryManager
import android.os.Debug
import org.json.JSONObject
import java.io.File

data class MetricsEntry(
    val timestamp: Long,
    val label: String,
    val confidence: Float,
    val latencyMs: Double,
    val memoryMb: Double,
    val batteryLevel: Int,
    val batteryMwh: Double?,
)

class MetricsLogger(private val context: Context) {
    private val logDir: File = File(context.filesDir, "metrics")
    private val batteryManager = context.getSystemService(BatteryManager::class.java)

    fun snapshot(label: String, confidence: Float, latencyMs: Double): MetricsEntry {
        val memMb = Debug.getNativeHeapAllocatedSize().toDouble() / (1024.0 * 1024.0)
        val level = batteryManager?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
        val energyNanoWh = batteryManager?.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER) ?: Long.MIN_VALUE
        val energyMwh = if (energyNanoWh > 0) energyNanoWh / 1_000_000.0 else null
        return MetricsEntry(
            timestamp = System.currentTimeMillis(),
            label = label,
            confidence = confidence,
            latencyMs = latencyMs,
            memoryMb = memMb,
            batteryLevel = level,
            batteryMwh = energyMwh,
        )
    }

    fun persist(entry: MetricsEntry) {
        logDir.mkdirs()
        val file = File(logDir, "latest.json")
        val payload = JSONObject()
            .put("timestamp", entry.timestamp)
            .put("label", entry.label)
            .put("confidence", entry.confidence)
            .put("latency_ms", entry.latencyMs)
            .put("memory_mb", entry.memoryMb)
            .put("battery_level", entry.batteryLevel)
            .put("battery_drop_mwh", entry.batteryMwh)
        file.writeText(payload.toString())
    }
}
