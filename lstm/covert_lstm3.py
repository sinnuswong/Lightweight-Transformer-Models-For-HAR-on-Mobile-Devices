import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import myutil

loaded_model = load_model('./lstm3_model.h5')

# 显示模型结构和摘要
loaded_model.summary()


run_model = tf.function(lambda x: loaded_model(x))
# This is important, let's fix the input size.

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 25, 6], loaded_model.inputs[0].dtype))

# model directory.
MODEL_DIR = "keras_lstm3"
loaded_model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
# Save the model.
with open('model_LSTM3.tflite', 'wb') as f:
    f.write(tflite_model)