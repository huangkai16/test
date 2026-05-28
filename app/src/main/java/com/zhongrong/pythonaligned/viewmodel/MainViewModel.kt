package com.zhongrong.pythonaligned.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.zhongrong.pythonaligned.PythonAlignedApplication
import com.zhongrong.pythonaligned.model.DetectConfig
import com.zhongrong.pythonaligned.model.DetectionOverlayDrawer
import com.zhongrong.pythonaligned.model.PythonAlignedDetectRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class MainUiState(
    val modelUri: Uri? = null,
    val modelPath: String? = null,
    val modelReady: Boolean = false,
    val modelInfo: String = "",
    val modelError: String? = null,
    val imageUri: Uri? = null,
    val imageLabel: String? = null,
    val previewBitmap: Bitmap? = null,
    val resultBitmap: Bitmap? = null,
    val confThreshold: Float = 0.3f,
    val iouThreshold: Float = 0.5f,
    val maxDetections: Int = 50,
    val highlightConfMin: Float = 0.8f,
    val isRunning: Boolean = false,
    val resultLines: List<String> = emptyList(),
    val errorMessage: String? = null,
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val runner: PythonAlignedDetectRunner =
        (application as PythonAlignedApplication).detectRunner

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch { loadBundledAssetsAndRun() }
    }

    fun onModelSelected(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { old ->
                old.resultBitmap?.recycle()
                old.copy(
                    modelUri = uri,
                    modelError = null,
                    modelReady = false,
                    resultLines = emptyList(),
                    resultBitmap = null,
                )
            }
            val error = withContext(Dispatchers.IO) { runner.loadModelFromUri(uri) }
            _uiState.update {
                it.copy(
                    modelReady = runner.modelReady,
                    modelPath = runner.modelPath,
                    modelInfo = if (runner.modelReady) runner.modelInfo() else "",
                    modelError = error,
                )
            }
        }
    }

    fun tryLoadAssetModel() {
        viewModelScope.launch { loadBundledAssetsAndRun() }
    }

    fun onImageSelected(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { old ->
                old.resultBitmap?.recycle()
                old.copy(imageUri = uri, errorMessage = null, resultLines = emptyList(), resultBitmap = null)
            }
            val bitmap = loadBitmap(uri)
            if (bitmap == null) {
                _uiState.update { it.copy(errorMessage = "无法解码所选图片") }
                return@launch
            }
            val label = uri.lastPathSegment ?: uri.toString()
            _uiState.update { old ->
                old.previewBitmap?.recycle()
                old.resultBitmap?.recycle()
                old.copy(previewBitmap = bitmap, imageLabel = label, resultBitmap = null)
            }
        }
    }

    fun updateConfThreshold(value: Float) {
        _uiState.update { it.copy(confThreshold = value.coerceIn(0.01f, 1f)) }
    }

    fun updateIouThreshold(value: Float) {
        _uiState.update { it.copy(iouThreshold = value.coerceIn(0.01f, 1f)) }
    }

    fun updateHighlightConfMin(value: Float) {
        _uiState.update { it.copy(highlightConfMin = value.coerceIn(0f, 1f)) }
    }

    fun runDetection() {
        val state = _uiState.value
        val bitmap = state.previewBitmap
        if (bitmap == null || bitmap.isRecycled) {
            _uiState.update { it.copy(errorMessage = "请先选择图片") }
            return
        }
        if (!state.modelReady) {
            _uiState.update { it.copy(errorMessage = state.modelError ?: "请先选择模型文件") }
            return
        }
        viewModelScope.launch { runDetectionInternal() }
    }

    private suspend fun loadBundledAssetsAndRun() {
        val modelError = withContext(Dispatchers.IO) { runner.loadDefaultAssetModel() }
        val (bitmap, imageError) = withContext(Dispatchers.IO) { runner.loadDefaultAssetImage() }

        _uiState.update { old ->
            old.previewBitmap?.recycle()
            old.resultBitmap?.recycle()
            old.copy(
                modelReady = runner.modelReady,
                modelPath = runner.modelPath,
                modelInfo = if (runner.modelReady) runner.modelInfo() else "",
                modelError = modelError,
                previewBitmap = bitmap,
                resultBitmap = null,
                imageLabel = PythonAlignedDetectRunner.DEFAULT_IMAGE_ASSET,
                errorMessage = imageError,
                resultLines = emptyList(),
            )
        }

        if (runner.modelReady && bitmap != null) {
            runDetectionInternal()
        }
    }

    private suspend fun runDetectionInternal() {
        val state = _uiState.value
        val bitmap = state.previewBitmap ?: return
        if (bitmap.isRecycled || !state.modelReady) return

        val config = DetectConfig(
            confThreshold = state.confThreshold,
            iouThreshold = state.iouThreshold,
            maxDetections = state.maxDetections,
            highlightConfMin = state.highlightConfMin,
        )

        _uiState.update { it.copy(isRunning = true, errorMessage = null, resultLines = emptyList(), resultBitmap = null) }
        try {
            val result = withContext(Dispatchers.IO) {
                runner.run(bitmap, state.imageLabel ?: "picked", config)
            }
            val annotated = withContext(Dispatchers.IO) {
                DetectionOverlayDrawer.draw(
                    source = bitmap,
                    detections = result.allDetections,
                    highlightConfMin = config.highlightConfMin,
                )
            }
            _uiState.update { old ->
                old.resultBitmap?.recycle()
                old.copy(
                    isRunning = false,
                    resultLines = result.toResultLines(),
                    resultBitmap = annotated,
                )
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(isRunning = false, errorMessage = "推理失败: ${e.message}")
            }
        }
    }

    private suspend fun loadBitmap(uri: Uri): Bitmap? = withContext(Dispatchers.IO) {
        getApplication<Application>().contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream)
        }
    }

    override fun onCleared() {
        _uiState.value.previewBitmap?.recycle()
        _uiState.value.resultBitmap?.recycle()
        super.onCleared()
    }
}
