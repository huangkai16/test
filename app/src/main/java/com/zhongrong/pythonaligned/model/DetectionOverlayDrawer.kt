package com.zhongrong.pythonaligned.model

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface

/** 将 NMS 后的检测框绘制回原图坐标系（与 [YOLOv11TFLiteDetector] 输出一致）。 */
object DetectionOverlayDrawer {

    fun draw(
        source: Bitmap,
        detections: List<YOLODetection>,
        highlightConfMin: Float,
    ): Bitmap {
        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        if (detections.isEmpty()) return output

        val canvas = Canvas(output)
        val normalStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#2196F3")
            style = Paint.Style.STROKE
            strokeWidth = maxOf(3f, source.width / 200f)
        }
        val highlightStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#4CAF50")
            style = Paint.Style.STROKE
            strokeWidth = maxOf(5f, source.width / 120f)
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = maxOf(24f, source.width / 28f)
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }

        detections.forEach { detection ->
            val highlighted = detection.confidence > highlightConfMin
            val stroke = if (highlighted) highlightStroke else normalStroke
            textBgPaint.color = if (highlighted) {
                Color.parseColor("#CC388E3C")
            } else {
                Color.parseColor("#CC1565C0")
            }

            val box = detection.boundingBox
            canvas.drawRect(box, stroke)
            drawLabel(
                canvas = canvas,
                text = "${detection.chineseName} ${"%.2f".format(detection.confidence)}",
                anchor = box,
                textPaint = textPaint,
                bgPaint = textBgPaint,
            )
        }

        return output
    }

    private fun drawLabel(
        canvas: Canvas,
        text: String,
        anchor: RectF,
        textPaint: Paint,
        bgPaint: Paint,
    ) {
        val bounds = Rect()
        textPaint.getTextBounds(text, 0, text.length, bounds)
        val padding = 6f
        val textWidth = bounds.width().toFloat() + padding * 2
        val textHeight = bounds.height().toFloat() + padding * 2

        var left = anchor.left
        var top = anchor.top - textHeight
        if (top < 0f) top = anchor.top
        if (left + textWidth > canvas.width) left = (canvas.width - textWidth).coerceAtLeast(0f)

        canvas.drawRect(left, top, left + textWidth, top + textHeight, bgPaint)
        canvas.drawText(text, left + padding, top + textHeight - padding, textPaint)
    }
}
