package com.eros.clip;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
public class MobileHART {
    public static String TAG = ClipVision2.class.getSimpleName();
    protected Context context;
    protected ImageProcessor imageProcessor;
    public final int inputSize = 224;
    private Interpreter interpreter;
    private int OUTPUT_SIZE = 2;


    public MobileHART(Context context) {
        this.context = context;

        CompatibilityList compatList = new CompatibilityList();
        Interpreter.Options options;

        if (compatList.isDelegateSupportedOnThisDevice()) {
            Log.d(TAG, "This device is GPU Compatible ");

//                options = new Interpreter.Options();
//                options.addDelegate(new NnApiDelegate());
            //                options.setUseXNNPACK(true);
//                options.setUseNNAPI(true);
        }
        Log.d(TAG, "This device is GPU Incompatible ");
        options = new Interpreter.Options().setNumThreads(4);

        try {
            MappedByteBuffer buffer = TfLoader.loadModelFile(context, "model.tflite");
            Log.d(TAG, "This device is GPU Incompatible 0");

            interpreter = new Interpreter(buffer, options);
            interpreter.resizeInput(interpreter.getInputTensor(0).index(), new int[]{
                    1, 25, 6
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
                .add(new StandardizeOp())
                .build();

    }

    private TensorBuffer buildModelInput(Bitmap image) {
        return imageProcessor.process(TensorImage.fromBitmap(image)).getTensorBuffer();
//        TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(new int[]{1, 3, 224, 224}, DataType.FLOAT32);
        // bitmap = BitmapUtil.scaleCenterCrop(bitmap, 224, 224);
//        tensorBuffer.loadBuffer(BitmapUtil.coverF32ByteBuffer(image));
    }
    private List<Integer> costs = new ArrayList<>();
    public void encode(Bitmap image) {
        long start = System.currentTimeMillis();

        TensorBuffer buffer = TensorBufferFloat.createFixedSize(new int[]{1, 25, 6}, DataType.FLOAT32);
        Log.d(TAG, "preprocess cost " + (System.currentTimeMillis() - start));


        for (int i = 0; i < 5; i++) {
            start = System.currentTimeMillis();
            //imgData is input to our model
            Object[] inputArray = {buffer.getBuffer()};

            Map<Integer, Object> outputMap = new HashMap<>();
            float[][] embeddings = new float[1][OUTPUT_SIZE]; //output of model will be stored in this variable
            outputMap.put(0, embeddings);

            interpreter.runForMultipleInputsOutputs(inputArray, outputMap); //Run model
            int cost = (int) (System.currentTimeMillis() - start);
            costs.add(cost);
            Log.d(TAG, "mInterpreter cost " + cost);
            Log.d(TAG, Arrays.toString(embeddings[0]));

        }
        android.os.Handler handler = new android.os.Handler(Looper.getMainLooper());
        handler.post(() -> {
            Toast.makeText(context, costs.toString(), Toast.LENGTH_LONG).show();
        });

    }
}
