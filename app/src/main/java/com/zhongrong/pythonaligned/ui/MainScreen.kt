package com.zhongrong.pythonaligned.ui

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.zhongrong.pythonaligned.model.BundledModel
import com.zhongrong.pythonaligned.viewmodel.MainViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(viewModel: MainViewModel) {
    val state by viewModel.uiState.collectAsState()

    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) viewModel.onImageSelected(uri)
    }
    val pickBatchInputDir = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocumentTree(),
    ) { uri ->
        if (uri != null) viewModel.onBatchInputDirSelected(uri)
    }
    val pickBatchOutputDir = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocumentTree(),
    ) { uri ->
        if (uri != null) viewModel.onBatchOutputDirSelected(uri)
    }

    val busy = state.isRunning || state.isBatchRunning

    Scaffold(
        topBar = { TopAppBar(title = { Text("Python 对齐 YOLO 推理") }) },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = if (state.modelReady) {
                    "模型已加载: ${state.selectedModel.label}\n${state.modelPath}\n${state.modelInfo}"
                } else {
                    "模型未加载${state.modelError?.let { ": $it" } ?: ""}"
                },
                color = if (state.modelReady) MaterialTheme.colorScheme.onSurface
                else MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium,
            )

            Text("内置模型", style = MaterialTheme.typography.titleSmall)
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .selectableGroup(),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                BundledModel.entries.forEach { model ->
                    Row(
                        modifier = Modifier
                            .selectable(
                                selected = state.selectedModel == model,
                                onClick = { viewModel.selectBundledModel(model) },
                                role = Role.RadioButton,
                            ),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        RadioButton(
                            selected = state.selectedModel == model,
                            onClick = null,
                        )
                        Text(model.label, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }

            Button(
                onClick = viewModel::tryLoadAssetModel,
                modifier = Modifier.fillMaxWidth(),
            ) { Text("重新加载 assets") }

            Text("推理参数", style = MaterialTheme.typography.titleSmall)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ThresholdField(
                    label = "conf",
                    value = state.confThreshold,
                    onValue = viewModel::updateConfThreshold,
                    modifier = Modifier.weight(1f),
                )
                ThresholdField(
                    label = "IoU",
                    value = state.iouThreshold,
                    onValue = viewModel::updateIouThreshold,
                    modifier = Modifier.weight(1f),
                )
                ThresholdField(
                    label = "高亮>",
                    value = state.highlightConfMin,
                    onValue = viewModel::updateHighlightConfMin,
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(
                    onClick = { pickImage.launch("image/*") },
                    modifier = Modifier.weight(1f),
                ) { Text("选择图片") }
                Button(
                    onClick = viewModel::runDetection,
                    enabled = !busy && state.previewBitmap != null && state.modelReady,
                    modifier = Modifier.weight(1f),
                ) { Text(if (state.isRunning) "推理中…" else "开始推理") }
            }

            Text("批量推理", style = MaterialTheme.typography.titleSmall)
            Text(
                text = buildString {
                    append("输入: ")
                    append(state.batchInputDirLabel ?: "未选择")
                    append("\n输出: ")
                    append(state.batchOutputDirLabel ?: "未选择")
                },
                style = MaterialTheme.typography.bodySmall,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(
                    onClick = { pickBatchInputDir.launch(null) },
                    enabled = !busy,
                    modifier = Modifier.weight(1f),
                ) { Text("输入目录") }
                Button(
                    onClick = { pickBatchOutputDir.launch(null) },
                    enabled = !busy,
                    modifier = Modifier.weight(1f),
                ) { Text("输出目录") }
            }
            Button(
                onClick = viewModel::runBatchInference,
                enabled = !busy &&
                    state.modelReady &&
                    state.batchInputDirUri != null &&
                    state.batchOutputDirUri != null,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (state.isBatchRunning) "批量推理中…" else "开始批量推理")
            }

            if (state.isBatchRunning && state.batchProgress != null) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CircularProgressIndicator(modifier = Modifier.padding(end = 8.dp))
                    Text(state.batchProgress!!)
                }
            }

            if (state.isRunning) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CircularProgressIndicator(modifier = Modifier.padding(end = 8.dp))
                    Text("Letterbox 推理中…")
                }
            }

            val showResult = state.resultBitmap != null
            val displayBitmap = state.resultBitmap ?: state.previewBitmap
            if (displayBitmap != null && !displayBitmap.isRecycled) {
                key(state.imageEpoch, showResult) {
                    PreviewCard(
                        bitmap = displayBitmap,
                        imageEpoch = state.imageEpoch,
                        title = if (showResult) {
                            "检测结果（NMS 框已映射回原图）"
                        } else {
                            "原图预览"
                        },
                    )
                }
            }

            state.errorMessage?.let {
                Text(it, color = MaterialTheme.colorScheme.error)
            }

            if (state.batchResultLines.isNotEmpty()) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("批量推理输出", style = MaterialTheme.typography.titleMedium)
                        Spacer(Modifier.height(8.dp))
                        state.batchResultLines.forEach { line ->
                            if (line.isEmpty()) Spacer(Modifier.height(4.dp))
                            else Text(line, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }

            if (state.resultLines.isNotEmpty()) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("模型输出", style = MaterialTheme.typography.titleMedium)
                        Spacer(Modifier.height(8.dp))
                        state.resultLines.forEach { line ->
                            if (line.isEmpty()) Spacer(Modifier.height(4.dp))
                            else Text(line, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ThresholdField(
    label: String,
    value: Float,
    onValue: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    OutlinedTextField(
        value = "%.2f".format(value),
        onValueChange = { text ->
            text.toFloatOrNull()?.let(onValue)
        },
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        modifier = modifier,
    )
}

@Composable
private fun PreviewCard(bitmap: Bitmap, imageEpoch: Long, title: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.padding(start = 16.dp, top = 12.dp, end = 16.dp),
            )
            val imageBitmap = remember(imageEpoch, bitmap) {
                if (bitmap.isRecycled) null else bitmap.asImageBitmap()
            }
            if (imageBitmap == null) return@Column
            Image(
                bitmap = imageBitmap,
                contentDescription = title,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(280.dp)
                    .padding(16.dp),
                contentScale = ContentScale.Fit,
            )
        }
    }
}
