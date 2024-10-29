import tensorflow as tf
import numpy as np
from keras.layers import ConvLSTM2D
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import coremltools as ct
import os

import myutil

current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_directory = os.path.dirname(current_file_path)

l2 = tf.keras.regularizers.L2
Adam = tf.keras.optimizers.Adam

# configs
window_size = 25
feature_size = 6
num_classes = 3
hidden_size = 64
output_size = num_classes
input_shape = (window_size, feature_size)
batch_size = 32  # (128,50: 14,16), (32,50: 17,14),
epochs = 50  # (128,20: 20, 8) (32,20: 15, 16)
##### 这里64，128， 50 最好。
##### 这里64，32， 50 最好。
# (64,32,50: 14,15) (64,32,20: 9,21)
# (64,128,20: 0,23)

# (128,128, 50: 14,17)

# (128, 20: 10:12) , (128,128, 20: 10:12) ,
# (128, 32,20: 2,28) (128, 32,50: 7,17)

kill_data_path = '/Users/sinnus/Desktop/ActivityData/badminton/c25/右撇子所有/kill_high_long_hit'
save_model_base_path = current_directory + os.sep + 'right_model'
model_name = 'right_lstm_kill'
save_model_path_no_extension = save_model_base_path + os.sep + model_name

L2 = 0.000001

model = tf.keras.Sequential(name='sequential_1')

model.add(tf.keras.layers.LSTM(hidden_size, return_sequences=False, input_shape=(window_size, feature_size),
                               kernel_initializer='orthogonal', kernel_regularizer=l2(L2), recurrent_regularizer=l2(L2),
                               bias_regularizer=l2(L2), name="LSTM_1"))

# model.add(tf.keras.layers.Flatten(name='Flatten'))
model.add(tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
                                name="Dense_1"))
model.add(tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
                                name="Dense_2"))
model.add(tf.keras.layers.Dense(output_size, activation='softmax', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
                                name="Output"))
model.summary()

# compile
LR = 0.0001
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LR), metrics=['accuracy'])
# prepare callbacks
from keras.callbacks import ModelCheckpoint

callbacks = [
    ModelCheckpoint(save_model_path_no_extension + '.h5', save_weights_only=False, save_best_only=True,
                    verbose=1)]

data, labels = myutil.build_badminton_kill_data(kill_data_path=kill_data_path)
print(data.shape)

labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
print(labels.shape)
print(labels)

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                    callbacks=callbacks)

# model.fit(data, labels,
#           batch_size=batch_size, epochs=epochs,
#           validation_split=0.2, callbacks=callbacks)

model.save(save_model_path_no_extension + '.h5')


def save_mode():
    best_model = load_model(save_model_path_no_extension + '.h5')
    run_model = tf.function(lambda x: best_model(x))
    # This is important, let's fix the input size.

    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, window_size, feature_size], model.inputs[0].dtype))

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


# save_mode()
run_model = tf.function(lambda x: model(x))

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, window_size, feature_size], model.inputs[0].dtype))

# model directory.
MODEL_DIR = save_model_path_no_extension
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
# Save the model.
with open(save_model_path_no_extension + '.tflite', 'wb') as f:
    f.write(tflite_model)
coreml_model = ct.convert(save_model_path_no_extension + '.h5')
coreml_model.save(save_model_path_no_extension + '.mlmodel')
