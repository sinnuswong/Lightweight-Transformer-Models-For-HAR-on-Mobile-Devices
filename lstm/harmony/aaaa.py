import numpy
import tensorflow as tf
import numpy as np
from keras.layers import ConvLSTM2D
from tensorflow.keras.models import load_model
import myutil_huawei
from sklearn.model_selection import train_test_split
import coremltools as ct
from keras.callbacks import ModelCheckpoint
import os
from tcn import TCN

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
kill_data_path = '/Users/sinnus/Desktop/ActivityData/badminton/c130/1020left/kill_high_long_hit'
save_model_base_path = current_directory + os.sep + 'left_model'
model_name = 'left_tcn_kill'
save_model_path_no_extension = save_model_base_path + os.sep + model_name

l2 = tf.keras.regularizers.L2
Adam = tf.keras.optimizers.Adam

window_size = 130
feature_size = 6
num_classes = 3

hidden_size = 64
dense_size = 128
output_size = num_classes
input_shape = (window_size, feature_size)
batch_size = 32  # (128,50: 14,16), (32,50: 17,14),
epochs = 25  # (128,20: 20, 8) (32,20: 15, 16)
L2 = 0.000001

model = tf.keras.Sequential(name='sequential_1')
model.add(TCN(nb_filters=hidden_size,
              kernel_size=5,  # 卷积核大小
              dilations=[1, 2, 4, 8],  # 膨胀率
              return_sequences=False,
              kernel_initializer='orthogonal',
              name="TCN_1",
              input_shape=(window_size, feature_size)))

# model.add(tf.keras.layers.Flatten(name='Flatten'))
model.add(tf.keras.layers.Dense(dense_size, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
                                name="Dense_1"))
model.add(tf.keras.layers.Dense(dense_size, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
                                name="Dense_2"))
model.add(tf.keras.layers.Dense(output_size, activation='softmax', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
                                name="Output"))
model.summary()

# compile
LR = 0.0001
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LR), metrics=['accuracy'])


def save_mode():
    model.load_weights(save_model_path_no_extension + '.h5')
    #best_model = load_model(save_model_path_no_extension + '.h5')
    run_model = tf.function(lambda x: model(x))
    # This is important, let's fix the input size.

    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, window_size, feature_size], numpy.float32))

    # model directory.
    MODEL_DIR = save_model_path_no_extension
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    tflite_model = converter.convert()
    # Save the model.
    with open(save_model_path_no_extension + '.tflite', 'wb') as f:
        f.write(tflite_model)

    coreml_model = ct.convert([concrete_func])
    coreml_model.save(save_model_path_no_extension + '.mlmodel')


save_mode()