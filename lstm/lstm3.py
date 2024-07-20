import tensorflow as tf
import numpy as np
from keras.layers import ConvLSTM2D
from tensorflow.keras.models import load_model
import myutil

l2 = tf.keras.regularizers.L2
Adam = tf.keras.optimizers.Adam
# 256, 64batch
hidden_units = 128
window_size = 25
feature_size = 6
class_size = 2
L2 = 0.000001

model = tf.keras.Sequential(name='sequential_1')

model.add(tf.keras.layers.LSTM(hidden_units, return_sequences=True, input_shape=(window_size, feature_size),
                               kernel_initializer='orthogonal', kernel_regularizer=l2(L2), recurrent_regularizer=l2(L2),
                               bias_regularizer=l2(L2), name="LSTM_1"))

model.add(tf.keras.layers.Flatten(name='Flatten'))
model.add(tf.keras.layers.Dense(hidden_units, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
                                name="Dense_1"))
model.add(tf.keras.layers.Dense(hidden_units, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
                                name="Dense_2"))
model.add(tf.keras.layers.Dense(class_size, activation='softmax', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
                                name="Output"))
model.summary()

# compile
LR = 0.0001
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LR), metrics=['accuracy'])
# prepare callbacks
from keras.callbacks import ModelCheckpoint

callbacks = [ModelCheckpoint('lstm3_model.h5', save_weights_only=False, save_best_only=True, verbose=1)]

batch_size = 128
epochs = 20

data, labels = myutil.build_badminton_hit_data()
print(data.shape)

labels = tf.keras.utils.to_categorical(labels, num_classes=class_size)
print(labels.shape)
# 训练模型
# model.fit(data, labels, epochs=3, batch_size=64, validation_split=0.2)
# model.save('my_model.h5')
model.fit(data, labels,
          batch_size=batch_size, epochs=epochs,
          validation_split=0.2, callbacks=callbacks)

model.save('lstm3_latest.h5')

run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 25, 6], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "keras_lstm3"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
# Save the model.
with open('model_LSTM3.tflite', 'wb') as f:
    f.write(tflite_model)
