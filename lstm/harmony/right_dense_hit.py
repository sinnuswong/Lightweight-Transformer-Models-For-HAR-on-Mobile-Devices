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
hit_data_path = '/Users/sinnus/Desktop/ActivityData/badminton/c130/0921right/hit'
save_model_base_path = current_directory + os.sep + 'right_model'
model_name = 'right_dense_hit'
save_model_path_no_extension = save_model_base_path + os.sep + model_name
# 获取当前文件所在的目录

l2 = tf.keras.regularizers.L2
Adam = tf.keras.optimizers.Adam

# configs
window_size = 130
feature_size = 6
num_classes = 2

hidden_size = 32
dense_size = 32
output_size = num_classes
input_shape = (window_size, feature_size)
batch_size = 32  # (128,50: 14,16), (32,50: 17,14),
epochs = 30  # (128,20: 20, 8) (32,20: 15, 16)

L2 = 0.000001

model = tf.keras.Sequential(name='sequential_1')

model.add(tf.keras.layers.Flatten(input_shape=(window_size, feature_size)))

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
# prepare callbacks

callbacks = [
    ModelCheckpoint(save_model_path_no_extension + '.h5', save_weights_only=False, save_best_only=True,
                    verbose=1)]

data, labels = myutil_huawei.build_badminton_hit_data(hit_data_path=hit_data_path)
print(data.shape)

labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
print(labels.shape)
print(labels)

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                    callbacks=callbacks)

# model.save(save_model_path_no_extension + '.h5')


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
# coreml_model = ct.convert(save_model_path_no_extension + '.h5')
# coreml_model.save(save_model_path_no_extension + '.mlmodel')