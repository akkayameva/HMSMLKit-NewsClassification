package com.ma.newscatecory.modelcreator

import android.content.Context
import android.util.Log
import com.huawei.hms.mlsdk.custom.MLModelExecutor
import kotlin.Throws
import android.widget.Toast
import com.huawei.hiai.modelcreatorsdk.textclassifier.TextClassifier
import com.huawei.hmf.tasks.OnSuccessListener
import com.huawei.hms.mlsdk.custom.MLModelOutputs
import com.huawei.hmf.tasks.OnFailureListener
import com.huawei.hms.mlsdk.custom.MLModelInputs
import com.huawei.hms.mlsdk.common.MLException
import com.huawei.hms.mlsdk.custom.MLModelInputOutputSettings
import com.huawei.hms.mlsdk.custom.MLModelDataType
import com.huawei.hms.mlsdk.custom.MLCustomLocalModel
import com.huawei.hms.mlsdk.custom.MLModelExecutorSettings
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.lang.Exception
import java.util.*

class ModelDetector(private val context: Context) {
    private val mModelName = "news_classification"
    private val mModelFullName = "$mModelName.ms"
    private var classifier: TextClassifier? = null
    private var modelExecutor: MLModelExecutor? = null
    private var loadMindsporeModelOk = false

    @Throws(Exception::class)
    fun loadFromAssets() {
        classifier = TextClassifier(context, MODEL_NAME)
        if (classifier!!.isHighAccMode) {
            val resName = "text_classifier.mc"

                if (!classifier!!.loadHighAccModelRes(context.assets.open(resName))) {
                    Log.e(TAG, "load high acc model res error")
                    return
                }
        }

        //load minspore model
        loadMindsporeModel()
    }

    val isReady: Boolean
        get() = classifier != null && classifier!!.isInitOk && loadMindsporeModelOk

    fun predict(
        textInput: String?,
        successCallback: OnSuccessListener<MLModelOutputs?>?,
        failureCallback: OnFailureListener?
    ) {
        if (!isReady) {
            Toast.makeText(context, "the model does not init ok", Toast.LENGTH_LONG).show()
            return
        }
        val input = classifier!!.getInput(textInput)
        Log.d(TAG, "interpret pre process")
        var inputs: MLModelInputs? = null
        try {
            inputs = MLModelInputs.Factory().add(input).create()
        } catch (e: MLException) {
            Log.e(TAG, "add inputs failed! " + e.message)
        }
        var inOutSettings: MLModelInputOutputSettings? = null
        try {
            val settingsFactory = MLModelInputOutputSettings.Factory()
            settingsFactory.setInputFormat(
                0,
                MLModelDataType.FLOAT32,
                intArrayOf(1, classifier!!.embSize, classifier!!.maxSeqLen)
            )
            val outputSettingsList = ArrayList<IntArray>()
            val outputShape = intArrayOf(1, classifier!!.labels.size)
            outputSettingsList.add(outputShape)
            for (i in outputSettingsList.indices) {
                settingsFactory.setOutputFormat(i, MLModelDataType.FLOAT32, outputSettingsList[i])
            }
            inOutSettings = settingsFactory.create()
        } catch (e: MLException) {
            Log.e(TAG, "set input output format failed! " + e.message)
        }
        Log.d(TAG, "interpret start")
        modelExecutor!!.exec(inputs, inOutSettings).addOnSuccessListener(successCallback)
            .addOnFailureListener(failureCallback)
    }

    fun getMaxProbLabel(probs: FloatArray): String {
        Log.d(TAG, Arrays.toString(probs))
        var maxLoc = -1
        var maxValue = Float.MIN_VALUE
        for (loc in probs.indices) {
            if (probs[loc] > maxValue) {
                maxLoc = loc
                maxValue = probs[loc]
            }
        }
        return classifier!!.labels[maxLoc]
    }

    fun softmax(logits: FloatArray): FloatArray {
        var maxValue = Float.MIN_VALUE
        for (i in logits.indices) {
            maxValue = Math.max(maxValue, logits[i])
        }
        val ans = FloatArray(logits.size)
        var sumExp = 0.0f
        for (i in ans.indices) {
            ans[i] = Math.exp((logits[i] - maxValue).toDouble()).toFloat()
            sumExp += ans[i]
        }
        for (i in ans.indices) {
            ans[i] = ans[i] / sumExp
        }
        return ans
    }

    @Throws(Exception::class)
    private fun loadMindsporeModel() {
        val resDir = context.filesDir
        val resFilePath = resDir.toString() + File.separator + "news_classification.ms"
        val fos = FileOutputStream(resFilePath)
        fos.write(classifier!!.msModelBytes, 0, classifier!!.msModelBytes.size)
        fos.close()
        val localModel =
            MLCustomLocalModel.Factory(mModelName).setLocalFullPathFile(resFilePath).create()
        val settings = MLModelExecutorSettings.Factory(localModel).create()
        try {
            modelExecutor = MLModelExecutor.getInstance(settings)
            loadMindsporeModelOk = true
        } catch (error: MLException) {
            error.printStackTrace()
        }
    }

    @Throws(IOException::class)
    private fun containFileInAssets(fileName: String): Boolean {
        for (tName in context.assets.list("")!!) {
            if (tName == fileName) {
                return true
            }
        }
        return false
    }

    companion object {
        private const val MODEL_NAME = "text_classifier.mc"
        private const val TAG = "HIAI_MINDSPORE"
        private fun sleep(ts: Int) {
            try {
                Thread.sleep(ts.toLong())
            } catch (e: InterruptedException) {
            }
        }
    }

    init {
        loadFromAssets()
    }
}