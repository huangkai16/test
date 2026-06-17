package com.zhongrong.pythonaligned.model

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

data class BatchItemResult(
    val fileName: String,
    val detectionCount: Int,
    val saved: Boolean,
    val outputFileName: String? = null,
    val error: String? = null,
)

data class BatchRunResult(
    val inputDirLabel: String,
    val outputDirLabel: String,
    val totalImages: Int,
    val savedCount: Int,
    val skippedNoDetection: Int,
    val failedCount: Int,
    val items: List<BatchItemResult>,
) {
    fun toResultLines(): List<String> = buildList {
        add("=== 批量推理结果 ===")
        add("输入: $inputDirLabel")
        add("输出: $outputDirLabel")
        add("图片总数: $totalImages")
        add("已保存 NMS 标注图: $savedCount")
        add("无检测跳过: $skippedNoDetection")
        add("失败: $failedCount")
        add("")
        items.forEach { item ->
            when {
                item.error != null -> add("✗ ${item.fileName}: ${item.error}")
                item.saved -> add("✓ ${item.fileName} → ${item.outputFileName} (${item.detectionCount} 框)")
                else -> add("- ${item.fileName}: 无检测，未保存")
            }
        }
    }
}

object BatchDirectoryInference {

    private val IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png", "webp", "bmp")

    suspend fun run(
        context: Context,
        runner: PythonAlignedDetectRunner,
        inputTreeUri: Uri,
        outputTreeUri: Uri,
        config: DetectConfig,
        onProgress: suspend (current: Int, total: Int, fileName: String) -> Unit,
    ): BatchRunResult {
        val inputRoot = DocumentFile.fromTreeUri(context, inputTreeUri)
            ?: throw IllegalStateException("无法访问输入目录")
        val outputRoot = DocumentFile.fromTreeUri(context, outputTreeUri)
            ?: throw IllegalStateException("无法访问输出目录")

        val images = inputRoot.listFiles()
            .filter { it.isFile && isImageFile(it.name) }
            .sortedBy { it.name?.lowercase() }

        val items = mutableListOf<BatchItemResult>()
        var savedCount = 0
        var skippedNoDetection = 0
        var failedCount = 0

        images.forEachIndexed { index, doc ->
            coroutineContext.ensureActive()
            val fileName = doc.name ?: "unknown"
            onProgress(index + 1, images.size, fileName)

            val item = processOne(context, runner, doc, outputRoot, config)
            items.add(item)
            when {
                item.error != null -> failedCount++
                item.saved -> savedCount++
                else -> skippedNoDetection++
            }
        }

        return BatchRunResult(
            inputDirLabel = inputRoot.name ?: inputTreeUri.lastPathSegment ?: inputTreeUri.toString(),
            outputDirLabel = outputRoot.name ?: outputTreeUri.lastPathSegment ?: outputTreeUri.toString(),
            totalImages = images.size,
            savedCount = savedCount,
            skippedNoDetection = skippedNoDetection,
            failedCount = failedCount,
            items = items,
        )
    }

    private suspend fun processOne(
        context: Context,
        runner: PythonAlignedDetectRunner,
        doc: DocumentFile,
        outputRoot: DocumentFile,
        config: DetectConfig,
    ): BatchItemResult {
        val fileName = doc.name ?: return BatchItemResult(
            fileName = "unknown",
            detectionCount = 0,
            saved = false,
            error = "文件名为空",
        )

        return try {
            val bitmap = decodeBitmap(context, doc.uri)
                ?: return BatchItemResult(fileName, 0, false, error = "无法解码")

            val snapshot = bitmap.copy(Bitmap.Config.ARGB_8888, false)
            if (snapshot != bitmap && !bitmap.isRecycled) {
                bitmap.recycle()
            }

            val result = runner.run(snapshot, fileName, config)
            val detections = result.allDetections

            if (detections.isEmpty()) {
                if (!snapshot.isRecycled) snapshot.recycle()
                return BatchItemResult(fileName, 0, saved = false)
            }

            val annotated = DetectionOverlayDrawer.draw(
                source = snapshot,
                detections = detections,
                highlightConfMin = config.highlightConfMin,
            )
            if (!snapshot.isRecycled) snapshot.recycle()

            val outputFileName = buildOutputFileName(fileName)
            val saved = saveJpeg(context, outputRoot, outputFileName, annotated)
            if (!annotated.isRecycled) annotated.recycle()

            if (!saved) {
                BatchItemResult(fileName, detections.size, false, error = "写入输出目录失败")
            } else {
                BatchItemResult(
                    fileName = fileName,
                    detectionCount = detections.size,
                    saved = true,
                    outputFileName = outputFileName,
                )
            }
        } catch (e: Exception) {
            BatchItemResult(fileName, 0, false, error = e.message ?: "未知错误")
        }
    }

    private fun decodeBitmap(context: Context, uri: Uri): Bitmap? {
        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return context.contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream, null, options)
        }
    }

    private fun saveJpeg(context: Context, outputRoot: DocumentFile, fileName: String, bitmap: Bitmap): Boolean {
        val existing = outputRoot.findFile(fileName)
        existing?.delete()

        val baseName = fileName.removeSuffix(".jpg").removeSuffix(".jpeg")
        val outDoc = outputRoot.createFile("image/jpeg", baseName)
            ?: return false

        return try {
            context.contentResolver.openOutputStream(outDoc.uri)?.use { stream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
            } ?: false
        } catch (_: Exception) {
            false
        }
    }

    private fun buildOutputFileName(sourceName: String): String {
        val dot = sourceName.lastIndexOf('.')
        val base = if (dot > 0) sourceName.substring(0, dot) else sourceName
        return "${base}_nms.jpg"
    }

    private fun isImageFile(name: String?): Boolean {
        if (name.isNullOrBlank()) return false
        val ext = name.substringAfterLast('.', "").lowercase()
        return ext in IMAGE_EXTENSIONS
    }
}
