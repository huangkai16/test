package com.zhongrong.pythonaligned.model

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

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
    val (buffer, _, scaled) = preprocessBitmapPythonAligned(bitmap, targetSize)
    scaled.recycle()
    return buffer
}
