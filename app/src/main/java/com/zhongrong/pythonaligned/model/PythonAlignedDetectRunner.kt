package com.zhongrong.pythonaligned.model

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import java.io.File

data class DetectConfig(
    val confThreshold: Float = 0.3f,
    val iouThreshold: Float = 0.5f,
    val maxDetections: Int = 50,
    /** 结果列表里单独标出 conf 大于该值的框（与仪器测试 CONFIDENCE_PRINT_MIN 一致） */
    val highlightConfMin: Float = 0.8f,
)

data class DetectRunResult(
    val imageLabel: String,
    val imageSize: String,
    val modelPath: String,
    val modelInfo: String,
    val config: DetectConfig,
    val allDetections: List<YOLODetection>,
    val highlightedDetections: List<YOLODetection>,
    val preprocessNote: String = "Python 对齐: 拉伸 640×640 → RGB /255 → float HWC",
) {
    fun toResultLines(): List<String> = buildList {
        add("=== Python 对齐推理结果 ===")
        add("图片: $imageLabel ($imageSize)")
        add("模型: $modelPath")
        add("模型参数: $modelInfo")
        add("前处理: $preprocessNote")
        add("推理阈值: conf≥${config.confThreshold}, IoU=${config.iouThreshold}, max=${config.maxDetections}")
        add("")
        add("NMS 后全部检测 (${allDetections.size}):")
        if (allDetections.isEmpty()) {
            add("  （无）")
        } else {
            allDetections.forEachIndexed { i, d -> add("  ${d.formatLine(i)}") }
        }
        add("")
        add("conf > ${config.highlightConfMin} 高置信框 (${highlightedDetections.size}):")
        if (highlightedDetections.isEmpty()) {
            add("  （无）")
        } else {
            highlightedDetections.forEachIndexed { i, d ->
                add("  ${d.formatLine(i, "high")}")
            }
        }
    }
}

class PythonAlignedDetectRunner(private val context: Context) {

    companion object {
        const val DEFAULT_MODEL_ASSET = "chef.tflite"
        const val DEFAULT_IMAGE_ASSET = "test.jpg"
    }

    private val detector = YOLOv11TFLiteDetector()

    var modelReady: Boolean = false
        private set

    var modelPath: String? = null
        private set

    fun loadModelFromUri(uri: Uri): String? {
        val dest = copyUriToCache(uri, "model.tflite") ?: return "无法读取模型文件"
        return loadModelFromFile(dest)
    }

    fun loadModelFromFile(file: File): String? {
        modelReady = detector.initialize(file)
        modelPath = if (modelReady) file.absolutePath else null
        return when {
            modelReady -> null
            else -> "模型加载失败: ${file.name}"
        }
    }

    /** 从 assets 加载默认模型（需存在 [DEFAULT_MODEL_ASSET]）。 */
    fun loadDefaultAssetModel(assetName: String = DEFAULT_MODEL_ASSET): String? {
        return try {
            context.assets.open(assetName).use { input ->
                val dest = File(context.cacheDir, assetName)
                dest.outputStream().use { output -> input.copyTo(output) }
                loadModelFromFile(dest)
            }
        } catch (_: Exception) {
            "assets 中未找到 $assetName"
        }
    }

    /** 从 assets 加载默认测试图（需存在 [DEFAULT_IMAGE_ASSET]）。 */
    fun loadDefaultAssetImage(assetName: String = DEFAULT_IMAGE_ASSET): Pair<Bitmap?, String?> {
        return try {
            context.assets.open(assetName).use { stream ->
                BitmapFactory.decodeStream(stream) to null
            }
        } catch (_: Exception) {
            null to "assets 中未找到 $assetName"
        }
    }

    fun modelInfo(): String = detector.modelInfo()

    fun run(
        bitmap: Bitmap,
        imageLabel: String,
        config: DetectConfig = DetectConfig(),
    ): DetectRunResult {
        val all = detector.detect(
            bitmap = bitmap,
            confThreshold = config.confThreshold,
            iouThreshold = config.iouThreshold,
            maxDetections = config.maxDetections,
        )
        val high = all.filter { it.confidence > config.highlightConfMin }
        return DetectRunResult(
            imageLabel = imageLabel,
            imageSize = "${bitmap.width}×${bitmap.height}",
            modelPath = modelPath ?: "（未加载）",
            modelInfo = detector.modelInfo(),
            config = config,
            allDetections = all,
            highlightedDetections = high,
        )
    }

    private fun copyUriToCache(uri: Uri, defaultName: String): File? {
        val resolver = context.contentResolver
        val name = resolver.query(uri, null, null, null, null)?.use { cursor ->
            val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && cursor.moveToFirst()) cursor.getString(idx) else defaultName
        } ?: defaultName

        val safeName = name.replace(Regex("[^a-zA-Z0-9._-]"), "_")
        val dest = File(context.cacheDir, "picked_$safeName")
        return try {
            resolver.openInputStream(uri)?.use { input ->
                dest.outputStream().use { output -> input.copyTo(output) }
            }
            dest.takeIf { it.isFile && it.length() > 0 }
        } catch (_: Exception) {
            null
        }
    }

    fun copyImageUriToCache(uri: Uri): Pair<File?, String?> {
        val file = copyUriToCache(uri, "image.jpg") ?: return null to "无法读取图片"
        return file to null
    }
}
