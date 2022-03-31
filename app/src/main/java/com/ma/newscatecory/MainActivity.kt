package com.ma.newscatecory

import android.widget.EditText
import android.widget.TextView
import com.ma.newscatecory.modelcreator.ModelDetector
import android.os.Bundle
import android.widget.Toast
import android.text.TextUtils
import android.util.Log
import android.view.View
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import java.lang.Exception

class MainActivity : AppCompatActivity() {
    private var inputEdt: EditText? = null
    private var identificationBtn: Button? = null
    private var clearBtn: Button? = null
    var labelTxt: TextView? = null
    private var detector: ModelDetector? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            detector = ModelDetector(this)
        } catch (e: Exception) {
            Log.e("hiai", e.message!!)
        }
        setContentView(R.layout.activity_main)
        initView()
        bindEvents()
    }

    private fun initView() {
        inputEdt = findViewById(R.id.input_data_edt)
        identificationBtn = findViewById(R.id.identification_btn)
        clearBtn = findViewById(R.id.clear_btn)
        labelTxt = findViewById(R.id.label_data_txt)
    }

    private fun bindEvents() {
        identificationBtn!!.setOnClickListener(View.OnClickListener {
            if (!detector!!.isReady) {
                Toast.makeText(this@MainActivity, "Model not ready", Toast.LENGTH_SHORT).show()
            } else {
                val inputData = inputEdt!!.text.toString()
                if (TextUtils.isEmpty(inputData)) {
                    Toast.makeText(this@MainActivity, "Please enter the text to be recognized ", Toast.LENGTH_SHORT).show()
                    return@OnClickListener
                }
                getLabelResult(inputData)
            }
        })
        clearBtn!!.setOnClickListener {
            inputEdt!!.setText("")
            labelTxt!!.visibility = View.GONE
        }
    }

    private fun getLabelResult(inputData: String) {
        val start = System.currentTimeMillis()
        detector!!.predict(inputData, { mlModelOutputs ->
            Log.i("hiai", "interpret get result")
            val result = mlModelOutputs!!.getOutput<Array<FloatArray>>(0)
            val probabilities = detector!!.softmax(result[0])
            val resultLabel = detector!!.getMaxProbLabel(probabilities)
            showResult(resultLabel)
            Toast.makeText(this@MainActivity, "success", Toast.LENGTH_SHORT).show()
            Log.i("hiai", "result: $resultLabel")
        }, { e ->
            e.printStackTrace()
            Log.e("hiai", "interpret failed, because " + e.message)
            Toast.makeText(this@MainActivity, "failed", Toast.LENGTH_SHORT).show()
        })
        Log.i("hiai", "time cost:" + (System.currentTimeMillis() - start))
        labelTxt!!.visibility = View.VISIBLE
    }


    fun showResult(result: String?) {
        labelTxt!!.post { labelTxt!!.text = result }
    }
}