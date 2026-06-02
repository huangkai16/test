package com.zhongrong.pythonaligned.model

import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv11 TFLite 检测器，前处理与 PythonAlignedPreprocessInstrumentedTest 对齐。
 * 模型从本地 .tflite 文件加载（非 assets 固定路径）。
 */
class YOLOv11TFLiteDetector {

    private var interpreter: Interpreter? = null
    private var isInitialized = false
    private var loadedModelPath: String? = null

    private var inputSize = 640
    private var numClasses = 7
    private var numDetections = 8400

    fun initialize(modelFile: File): Boolean {
        if (!modelFile.isFile) return false
        return try {
            close()
            val options = Interpreter.Options().apply { setNumThreads(4) }
            val model = loadMappedFile(modelFile)
            interpreter = Interpreter(model, options)

            val intp = interpreter ?: return false
            val inputShape = intp.getInputTensor(0).shape()
            val outputShape = intp.getOutputTensor(0).shape()

            inputSize = inputShape.getOrElse(1) { 640 }
            numDetections = outputShape[2]
            numClasses = outputShape[1] - 4

            isInitialized = true
            loadedModelPath = modelFile.absolutePath
            true
        } catch (_: Exception) {
            close()
            false
        }
    }

    fun loadedModelPath(): String? = loadedModelPath

    fun modelInfo(): String {
        if (!isInitialized) return "模型未加载"
        return "input=${inputSize}×$inputSize, classes=$numClasses, anchors=$numDetections"
    }

    fun detect(
        bitmap: android.graphics.Bitmap,
        confThreshold: Float = 0.3f,
        iouThreshold: Float = 0.5f,
        maxDetections: Int = 50,
    ): YOLODetectOutcome {
        if (!isInitialized) return YOLODetectOutcome.EMPTY

        return try {
            val preprocessStartNs = System.nanoTime()
            val result = letterbox(bitmap, inputSize, inputSize)
            val inputBuffer = result.buffer
            val preprocessMs = elapsedMs(preprocessStartNs)

            val output = Array(1) {
                Array(4 + numClasses) { FloatArray(numDetections) }
            }

            val intp = interpreter ?: return YOLODetectOutcome.EMPTY
            inputBuffer.rewind()
            val inferenceStartNs = System.nanoTime()
            intp.run(inputBuffer, output)
            val inferenceMs = elapsedMs(inferenceStartNs)

            val postprocessStartNs = System.nanoTime()
            val detections = postprocess(
                output[0],
                result,
                confThreshold,
                iouThreshold,
                maxDetections,
            )
            val postprocessMs = elapsedMs(postprocessStartNs)

            YOLODetectOutcome(
                detections = detections,
                preprocessMs = preprocessMs,
                inferenceMs = inferenceMs,
                postprocessMs = postprocessMs,
            )
        } catch (e: Exception) {
            throw IllegalStateException("TFLite 推理异常: ${e.message}", e)
        }
    }

    private fun elapsedMs(startNs: Long): Long =
        (System.nanoTime() - startNs) / 1_000_000L

    private fun postprocess(
        output: Array<FloatArray>,
        info: LetterboxResult,
        confThreshold: Float,
        iouThreshold: Float,
        maxDetections: Int,
    ): List<YOLODetection> {
        val (w0, h0) = info.imgsz_ori
        val (x0, y0) = info.x0y0
        val scale = 1 / info.ratio

        val transposed = transposeOutput(output)

        val boxes = mutableListOf<FloatArray>()
        val scores = mutableListOf<Float>()
        val classIds = mutableListOf<Int>()

        for (detection in transposed) {
            var maxScore = 0f
            var maxClassId = 0
            for (c in 0 until numClasses) {
                val score = detection[4 + c]
                if (score > maxScore) {
                    maxScore = score
                    maxClassId = c
                }
            }
            if (maxScore >= confThreshold) {
                boxes.add(floatArrayOf(detection[0], detection[1], detection[2], detection[3]))
                scores.add(maxScore)
                classIds.add(maxClassId)
            }
        }

        if (boxes.isEmpty()) return emptyList()

        val maxCoord = boxes.maxOf { max(max(it[0], it[1]), max(it[2], it[3])) }
        val needsScale = maxCoord <= 1.5f

        val boxesXYXY = boxes.map { box ->
            var cx = box[0]
            var cy = box[1]
            var w = box[2]
            var h = box[3]
            if (needsScale) {
                cx *= inputSize
                cy *= inputSize
                w *= inputSize
                h *= inputSize
            }
            floatArrayOf(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)
        }

        val keepIndices = nms(
            boxesXYXY.map { it.toList().toFloatArray() }.toTypedArray(),
            scores.toFloatArray(),
            iouThreshold,
            topK = 300,
        )

        return keepIndices.take(maxDetections).map { idx ->
            var x1 = (boxesXYXY[idx][0] - x0) * scale
            var y1 = (boxesXYXY[idx][1] - y0) * scale
            var x2 = (boxesXYXY[idx][2] - x0) * scale
            var y2 = (boxesXYXY[idx][3] - y0) * scale

            x1 = x1.coerceIn(0f, w0.toFloat())
            y1 = y1.coerceIn(0f, h0.toFloat())
            x2 = x2.coerceIn(0f, w0.toFloat())
            y2 = y2.coerceIn(0f, h0.toFloat())

            val classId = classIds[idx]
            YOLODetection(
                classId = classId,
                className = ChefClassNames.nameEn(classId),
                chineseName = ChefClassNames.nameEn(classId),
                confidence = scores[idx],
                boundingBox = RectF(x1, y1, x2, y2),
            )
        }
    }

    private fun transposeOutput(output: Array<FloatArray>): Array<FloatArray> {
        val numFeatures = output.size
        val numDet = output[0].size
        return Array(numDet) { i ->
            FloatArray(numFeatures) { j -> output[j][i] }
        }
    }

    private fun nms(
        boxes: Array<FloatArray>,
        scores: FloatArray,
        iouThreshold: Float,
        topK: Int = 300,
    ): List<Int> {
        if (boxes.isEmpty()) return emptyList()

        val indices = scores.indices.sortedByDescending { scores[it] }
        val topIndices = indices.take(topK).toMutableList()
        val keep = mutableListOf<Int>()

        while (topIndices.isNotEmpty()) {
            val i = topIndices.removeAt(0)
            keep.add(i)
            val box1 = boxes[i]
            val area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])

            topIndices.removeIf { j ->
                val box2 = boxes[j]
                val area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                val xx1 = max(box1[0], box2[0])
                val yy1 = max(box1[1], box2[1])
                val xx2 = min(box1[2], box2[2])
                val yy2 = min(box1[3], box2[3])
                val inter = max(0f, xx2 - xx1) * max(0f, yy2 - yy1)
                val union = area1 + area2 - inter + 1e-6f
                inter / union > iouThreshold
            }
        }
        return keep
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        isInitialized = false
        loadedModelPath = null
    }

    private fun loadMappedFile(file: File): MappedByteBuffer {
        FileInputStream(file).use { inputStream ->
            val fileChannel = inputStream.channel
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
        }
    }
}

data class YOLODetectOutcome(
    val detections: List<YOLODetection>,
    val preprocessMs: Long,
    val inferenceMs: Long,
    val postprocessMs: Long,
) {
    companion object {
        val EMPTY = YOLODetectOutcome(emptyList(), 0L, 0L, 0L)
    }
}

data class YOLODetection(
    val classId: Int,
    val className: String,
    val chineseName: String,
    val confidence: Float,
    val boundingBox: RectF,
) {
    fun formatLine(index: Int, label: String = ""): String {
        val prefix = if (label.isEmpty()) "#$index" else "[$label#$index]"
        return "$prefix classId=$classId $className/$chineseName " +
            "conf=${"%.6f".format(confidence)} " +
            "box_lt_rb=${"%.2f".format(boundingBox.left)},${"%.2f".format(boundingBox.top)}," +
            "${"%.2f".format(boundingBox.right)},${"%.2f".format(boundingBox.bottom)}"
    }
}

/** 7 类 chef 模型标签；其他 classId 回退为 class_N。 */
object ChefClassNames {
    private val EN = arrayOf(
        "apron", "chef_hat", "incorrectly_mask", "no_face_protective",
        "with_mask", "without_apron", "without_cap",
    )
    private val CN = arrayOf(
        "穿围裙", "戴厨师帽", "口罩佩戴不正确", "无面部防护",
        "正确戴口罩", "未穿围裙", "未戴帽子",
    )

    fun nameEn(classId: Int): String = EN.getOrElse(classId) { "class_$classId" }
    fun nameCn(classId: Int): String = CN.getOrElse(classId) { "类别_$classId" }
}
