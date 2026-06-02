package com.zhongrong.pythonaligned.model

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 与 [com.zhongrong.moringdetect.PythonAlignedPreprocessInstrumentedTest.preprocessAsPythonScript] 一致：
 * 拉伸 640×640 → RGB /255 → float32 HWC。
 */
internal fun preprocessBitmapPythonAligned(
    bitmap: Bitmap,
    targetSize: Int,
): Triple<ByteBuffer, Pair<Int, Int>, Bitmap> {
    val w0 = bitmap.width
    val h0 = bitmap.height
    val scaled = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
    // createScaledBitmap 在尺寸相同时可能返回入参本身，切勿 recycle。

    val buffer = ByteBuffer.allocateDirect(4 * targetSize * targetSize * 3)
    buffer.order(ByteOrder.nativeOrder())

    val pixels = IntArray(targetSize * targetSize)
    scaled.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)

    for (argb in pixels) {
        val r = (argb shr 16) and 0xFF
        val g = (argb shr 8) and 0xFF
        val b = argb and 0xFF
        buffer.putFloat(r / 255.0f)
        buffer.putFloat(g / 255.0f)
        buffer.putFloat(b / 255.0f)
    }

    buffer.rewind()
    return Triple(buffer, Pair(w0, h0), scaled)
}

/** 仪器测试同款：仅生成 float buffer，不保留 scaled bitmap。 */
fun preprocessAsPythonScript(bitmap: Bitmap, targetSize: Int = 640): ByteBuffer {
    val (buffer, _, _) = preprocessBitmapPythonAligned(bitmap, targetSize)
    return buffer
}

data class LetterboxResult(
    val bitmap: Bitmap,
    val ratio: Float,
    val x0y0: Pair<Int, Int>,
    val buffer: ByteBuffer,
    val imgsz_ori: Pair<Int, Int> // w, h
)

fun letterbox(bitmap: Bitmap, newWidth: Int, newHeight: Int, paddingValue: Int = 114, scaleup: Boolean = true, center: Boolean = true): LetterboxResult {

    val w0 = bitmap.width
    val h0 = bitmap.height

    // 1. 计算缩放比例
    var r = min(newWidth.toFloat() / w0, newHeight.toFloat() / h0)

    // only scale down, do not scale up
    if (!scaleup) {
        r = min(r, 1.0f)
    }

    // 2. 计算缩放后的尺寸（不填充）
    val newUnpadWidth = (w0 * r).roundToInt()
    val newUnpadHeight = (h0 * r).roundToInt()

    // 3. 计算需要填充的像素
    var dw = (newWidth - newUnpadWidth).toDouble()
    var dh = (newHeight - newUnpadHeight).toDouble()

    if (center) {
        dw /= 2.0
        dh /= 2.0
    }

    // 计算四周的具体 padding 像素 (对应 Python 的 round 偏置)
    val top = if (center) (dh - 0.1).roundToInt() else 0
    val bottom = if (center) (dh + 0.1).roundToInt() else dh.roundToInt()
    val left = if (center) (dw - 0.1).roundToInt() else 0
    val right = if (center) (dw + 0.1).roundToInt() else dw.roundToInt()

    // 4. 缩放图像（保持宽高比）
    val scaledBitmap = Bitmap.createScaledBitmap(bitmap,newUnpadWidth,newUnpadHeight,true)

    // 5. 创建最终画布并填充背景

    val resultBitmap = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(resultBitmap)
    // 填充背景色
    val fillColor = Color.rgb(paddingValue, paddingValue, paddingValue)
    canvas.drawColor(fillColor)

    // 6. 将缩放后的图像居中绘制
    canvas.drawBitmap(scaledBitmap, left.toFloat(), top.toFloat(), null)

    val buffer = ByteBuffer.allocateDirect(4 * newWidth * newHeight * 3)
    buffer.order(ByteOrder.nativeOrder())

    val pixels = IntArray(newWidth * newHeight)
    resultBitmap.getPixels(pixels, 0, newWidth, 0, 0, newWidth, newHeight)

    for (argb in pixels) {
        val r = (argb shr 16) and 0xFF
        val g = (argb shr 8) and 0xFF
        val b = argb and 0xFF
        buffer.putFloat(r / 255.0f)
        buffer.putFloat(g / 255.0f)
        buffer.putFloat(b / 255.0f)
    }

    buffer.rewind()

    // 勿 recycle scaledBitmap：尺寸与源图相同时 createScaledBitmap 可能返回入参本身。
    return LetterboxResult(bitmap=resultBitmap, ratio=r, x0y0=Pair(left, top), buffer=buffer, imgsz_ori=Pair(w0, h0))
}
