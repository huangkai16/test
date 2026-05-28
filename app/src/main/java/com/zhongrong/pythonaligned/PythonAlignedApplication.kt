package com.zhongrong.pythonaligned

import android.app.Application
import com.zhongrong.pythonaligned.model.PythonAlignedDetectRunner

class PythonAlignedApplication : Application() {

    lateinit var detectRunner: PythonAlignedDetectRunner
        private set

    override fun onCreate() {
        super.onCreate()
        detectRunner = PythonAlignedDetectRunner(this)
    }
}
