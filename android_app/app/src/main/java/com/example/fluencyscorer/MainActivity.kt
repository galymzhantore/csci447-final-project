package com.example.fluencyscorer

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val metricsLogger = MetricsLogger(this)
        val inferenceHelper = InferenceHelper(this, metricsLogger)
        val metadata = inferenceHelper.metadata()
        val audioRecorder = AudioRecorder(this, metadata.sampleRate, metadata.clipSeconds)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    FluencyScreen(recorder = audioRecorder, helper = inferenceHelper)
                }
            }
        }
    }
}

@Composable
fun FluencyScreen(recorder: AudioRecorder, helper: InferenceHelper) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val prediction = remember { mutableStateOf<PredictionResult?>(null) }
    val metrics = remember { mutableStateOf<MetricsEntry?>(null) }
    val status = remember { mutableStateOf("Ready") }
    val isRecording = remember { mutableStateOf(false) }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (!granted) {
            status.value = "Microphone permission denied"
        }
    }
    val audioPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            scope.launch {
                val pcm = withContext(Dispatchers.IO) { recorder.readFromUri(uri) }
                runInference(pcm, helper, prediction, metrics, status)
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(text = "Edge Fluency Profiler", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Text(text = status.value, style = MaterialTheme.typography.bodyMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = {
                val hasPermission = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                    android.content.pm.PackageManager.PERMISSION_GRANTED
                if (!hasPermission) {
                    permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    return@Button
                }
                if (!isRecording.value) {
                    recorder.startRecording()
                    isRecording.value = true
                    status.value = "Recording..."
                } else {
                    scope.launch {
                        val pcm = withContext(Dispatchers.IO) { recorder.stopRecording() }
                        isRecording.value = false
                        runInference(pcm, helper, prediction, metrics, status)
                    }
                }
            }) {
                Text(text = if (isRecording.value) "Stop & Infer" else "Record")
            }
            Button(onClick = { audioPicker.launch("audio/*") }) {
                Text(text = "Select Clip")
            }
        }
        prediction.value?.let { result ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(text = "Prediction", style = MaterialTheme.typography.titleMedium)
                    Text(text = "Label: ${result.label}", fontWeight = FontWeight.Bold)
                    Text(text = "Confidence: ${(result.confidence * 100).toInt()}%")
                    Text(text = "Latency: ${"%.2f".format(result.latencyMs)} ms")
                }
            }
        }
        metrics.value?.let { entry ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(text = "Telemetry", style = MaterialTheme.typography.titleMedium)
                    MetricRow(label = "Memory", value = "${"%.1f".format(entry.memoryMb)} MB")
                    MetricRow(label = "Battery", value = "${entry.batteryLevel}%")
                    entry.batteryMwh?.let {
                        MetricRow(label = "Battery (mWh)", value = "%.2f".format(it))
                    }
                }
            }
        }
    }
}

private suspend fun runInference(
    pcm: FloatArray,
    helper: InferenceHelper,
    prediction: androidx.compose.runtime.MutableState<PredictionResult?>,
    metrics: androidx.compose.runtime.MutableState<MetricsEntry?>,
    status: androidx.compose.runtime.MutableState<String>,
) {
    if (pcm.isEmpty()) {
        status.value = "No audio captured"
        return
    }
    status.value = "Running inference..."
    val (result, entry) = withContext(Dispatchers.Default) { helper.infer(pcm) }
    prediction.value = result
    metrics.value = entry
    status.value = "Last run: ${result.label}"
}

@Composable
fun MetricRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(text = label, style = MaterialTheme.typography.bodyMedium)
        Text(text = value, fontWeight = FontWeight.Bold)
    }
}
