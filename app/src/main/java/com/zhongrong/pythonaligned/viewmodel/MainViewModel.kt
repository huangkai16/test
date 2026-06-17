package com.zhongrong.pythonaligned.viewmodel

import android.app.Application
import android.content.Intent
import androidx.documentfile.provider.DocumentFile
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.zhongrong.pythonaligned.PythonAlignedApplication
import com.zhongrong.pythonaligned.model.BatchDirectoryInference
import com.zhongrong.pythonaligned.model.BundledModel
import com.zhongrong.pythonaligned.model.DetectConfig
import com.zhongrong.pythonaligned.model.DetectionOverlayDrawer
import com.zhongrong.pythonaligned.model.PythonAlignedDetectRunner
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class MainUiState(
    val selectedModel: BundledModel = PythonAlignedDetectRunner.DEFAULT_BUNDLED_MODEL,
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
    val isBatchRunning: Boolean = false,
    val batchInputDirUri: Uri? = null,
    val batchInputDirLabel: String? = null,
    val batchOutputDirUri: Uri? = null,
    val batchOutputDirLabel: String? = null,
    val batchProgress: String? = null,
    val batchResultLines: List<String> = emptyList(),
    val resultLines: List<String> = emptyList(),
    val errorMessage: String? = null,
    /** 预览/结果图变更时递增，供 Compose key 绑定，避免绘制已回收的 Bitmap。 */
    val imageEpoch: Long = 0L,
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val runner: PythonAlignedDetectRunner =
        (application as PythonAlignedApplication).detectRunner

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private var detectJob: Job? = null
    private var batchJob: Job? = null

    private val app get() = getApplication<Application>()

    init {
        viewModelScope.launch { loadBundledAssetsAndRun() }
    }

    private fun cancelDetection() {
        detectJob?.cancel()
        detectJob = null
    }

    private fun cancelBatch() {
        batchJob?.cancel()
        batchJob = null
    }

    private fun cancelAllJobs() {
        cancelDetection()
        cancelBatch()
    }

    fun onBatchInputDirSelected(uri: Uri) {
        persistTreeUri(uri, readOnly = true)
        val label = treeUriLabel(uri)
        _uiState.update {
            it.copy(
                batchInputDirUri = uri,
                batchInputDirLabel = label,
                batchResultLines = emptyList(),
                errorMessage = null,
            )
        }
    }

    fun onBatchOutputDirSelected(uri: Uri) {
        persistTreeUri(uri, readOnly = false)
        val label = treeUriLabel(uri)
        _uiState.update {
            it.copy(
                batchOutputDirUri = uri,
                batchOutputDirLabel = label,
                batchResultLines = emptyList(),
                errorMessage = null,
            )
        }
    }

    fun runBatchInference() {
        val state = _uiState.value
        if (!state.modelReady) {
            _uiState.update { it.copy(errorMessage = state.modelError ?: "请先加载模型") }
            return
        }
        val inputUri = state.batchInputDirUri
        val outputUri = state.batchOutputDirUri
        if (inputUri == null || outputUri == null) {
            _uiState.update { it.copy(errorMessage = "请先选择输入目录和输出目录") }
            return
        }
        cancelAllJobs()
        batchJob = viewModelScope.launch { runBatchInternal(inputUri, outputUri) }
    }

    fun selectBundledModel(model: BundledModel) {
        if (_uiState.value.selectedModel == model && runner.modelReady) return
        viewModelScope.launch { loadBundledModelAndMaybeRun(model) }
    }

    fun tryLoadAssetModel() {
        viewModelScope.launch { loadBundledAssetsAndRun() }
    }

    fun onImageSelected(uri: Uri) {
        viewModelScope.launch {
            cancelAllJobs()
            _uiState.update { old ->
                old.copy(
                    imageUri = uri,
                    errorMessage = null,
                    resultLines = emptyList(),
                    resultBitmap = null,
                    isRunning = false,
                )
            }
            val bitmap = loadBitmap(uri)
            if (bitmap == null) {
                _uiState.update { it.copy(errorMessage = "无法解码所选图片") }
                return@launch
            }
            val label = uri.lastPathSegment ?: uri.toString()
            _uiState.update { old ->
                old.copy(
                    previewBitmap = bitmap,
                    imageLabel = label,
                    resultBitmap = null,
                    imageEpoch = old.imageEpoch + 1,
                )
            }
            if (runner.modelReady) {
                detectJob = viewModelScope.launch { runDetectionInternal() }
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
        cancelDetection()
        detectJob = viewModelScope.launch { runDetectionInternal() }
    }

    private suspend fun runBatchInternal(inputUri: Uri, outputUri: Uri) {
        val state = _uiState.value
        val config = DetectConfig(
            confThreshold = state.confThreshold,
            iouThreshold = state.iouThreshold,
            maxDetections = state.maxDetections,
            highlightConfMin = state.highlightConfMin,
        )

        _uiState.update {
            it.copy(
                isBatchRunning = true,
                batchProgress = "准备中…",
                batchResultLines = emptyList(),
                errorMessage = null,
            )
        }

        try {
            val result = withContext(Dispatchers.IO) {
                BatchDirectoryInference.run(
                    context = app,
                    runner = runner,
                    inputTreeUri = inputUri,
                    outputTreeUri = outputUri,
                    config = config,
                    onProgress = { current, total, fileName ->
                        _uiState.update {
                            it.copy(batchProgress = "$current/$total  $fileName")
                        }
                    },
                )
            }
            _uiState.update {
                it.copy(
                    isBatchRunning = false,
                    batchProgress = null,
                    batchResultLines = result.toResultLines(),
                )
            }
        } catch (e: CancellationException) {
            _uiState.update {
                it.copy(isBatchRunning = false, batchProgress = null)
            }
            throw e
        } catch (e: Exception) {
            _uiState.update {
                it.copy(
                    isBatchRunning = false,
                    batchProgress = null,
                    errorMessage = "批量推理失败: ${e.message}",
                )
            }
        }
    }

    private fun persistTreeUri(uri: Uri, readOnly: Boolean) {
        val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or
            if (readOnly) 0 else Intent.FLAG_GRANT_WRITE_URI_PERMISSION
        try {
            app.contentResolver.takePersistableUriPermission(uri, flags)
        } catch (_: SecurityException) {
            // 部分系统/目录可能无法持久化，当次会话仍可用
        }
    }

    private fun treeUriLabel(uri: Uri): String {
        return DocumentFile.fromTreeUri(app, uri)?.name
            ?: uri.lastPathSegment
            ?: uri.toString()
    }

    private suspend fun loadBundledAssetsAndRun() {
        loadBundledModelAndMaybeRun(_uiState.value.selectedModel, reloadImage = true)
    }

    private suspend fun loadBundledModelAndMaybeRun(
        model: BundledModel,
        reloadImage: Boolean = false,
    ) {
        cancelAllJobs()
        val modelError = withContext(Dispatchers.IO) { runner.loadBundledAssetModel(model) }

        val imageResult = if (reloadImage) {
            withContext(Dispatchers.IO) { runner.loadDefaultAssetImage() }
        } else {
            null
        }

        _uiState.update { old ->
            val keepUserImage = old.imageUri != null
            val (bitmap, imageError) = if (reloadImage && !keepUserImage) {
                imageResult ?: (null to null)
            } else {
                old.previewBitmap to old.errorMessage
            }
            old.copy(
                selectedModel = model,
                modelReady = runner.modelReady,
                modelPath = runner.modelPath,
                modelInfo = if (runner.modelReady) runner.modelInfo() else "",
                modelError = modelError,
                previewBitmap = if (reloadImage && keepUserImage) old.previewBitmap else if (reloadImage) bitmap else old.previewBitmap,
                resultBitmap = if (reloadImage || model != old.selectedModel) null else old.resultBitmap,
                imageLabel = if (reloadImage && keepUserImage) old.imageLabel else if (reloadImage) PythonAlignedDetectRunner.DEFAULT_IMAGE_ASSET else old.imageLabel,
                errorMessage = if (reloadImage && keepUserImage) old.errorMessage else if (reloadImage) imageError else old.errorMessage,
                resultLines = if (model != old.selectedModel || reloadImage) emptyList() else old.resultLines,
                imageEpoch = when {
                    reloadImage && !keepUserImage -> old.imageEpoch + 1
                    model != old.selectedModel -> old.imageEpoch + 1
                    else -> old.imageEpoch
                },
            )
        }

        val state = _uiState.value
        if (runner.modelReady && state.previewBitmap != null && !state.isRunning) {
            detectJob = viewModelScope.launch { runDetectionInternal() }
        }
    }

    private suspend fun runDetectionInternal() {
        val state = _uiState.value
        val bitmap = state.previewBitmap ?: return
        if (bitmap.isRecycled || !state.modelReady) return
        val epoch = state.imageEpoch

        val snapshot = withContext(Dispatchers.IO) {
            if (bitmap.isRecycled) null else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
        if (snapshot == null) {
            _uiState.update { it.copy(errorMessage = "无法复制图片用于推理") }
            return
        }

        val config = DetectConfig(
            confThreshold = state.confThreshold,
            iouThreshold = state.iouThreshold,
            maxDetections = state.maxDetections,
            highlightConfMin = state.highlightConfMin,
        )

        _uiState.update { it.copy(isRunning = true, errorMessage = null, resultLines = emptyList(), resultBitmap = null) }
        try {
            val result = withContext(Dispatchers.IO) {
                ensureActive()
                runner.run(snapshot, state.imageLabel ?: "picked", config)
            }
            val annotated = withContext(Dispatchers.IO) {
                ensureActive()
                DetectionOverlayDrawer.draw(
                    source = snapshot,
                    detections = result.allDetections,
                    highlightConfMin = config.highlightConfMin,
                )
            }
            if (_uiState.value.imageEpoch != epoch) {
                _uiState.update {
                    it.copy(isRunning = false, errorMessage = "图片已更换，本次推理结果已丢弃")
                }
                return
            }
            _uiState.update { old ->
                old.copy(
                    isRunning = false,
                    resultLines = result.toResultLines(),
                    resultBitmap = annotated,
                    imageEpoch = old.imageEpoch + 1,
                )
            }
        } catch (e: CancellationException) {
            _uiState.update { it.copy(isRunning = false) }
            throw e
        } catch (e: Exception) {
            _uiState.update {
                it.copy(isRunning = false, errorMessage = "推理失败: ${e.message}")
            }
        }
    }

    private suspend fun loadBitmap(uri: Uri): Bitmap? = withContext(Dispatchers.IO) {
        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        getApplication<Application>().contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream, null, options)
        }
    }

    override fun onCleared() {
        cancelAllJobs()
        super.onCleared()
    }
}
