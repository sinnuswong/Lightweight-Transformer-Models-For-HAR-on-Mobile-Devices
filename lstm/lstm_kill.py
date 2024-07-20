import tensorflow as tf
import numpy as np
from keras.layers import ConvLSTM2D
from tensorflow.keras.models import load_model
import myutil
from sklearn.model_selection import train_test_split

l2 = tf.keras.regularizers.L2
Adam = tf.keras.optimizers.Adam
# 256, 64batch
hidden_units = 64
window_size = 25
feature_size = 6
class_size = 3
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

callbacks = [ModelCheckpoint('lstm3_model_kill.h5', save_weights_only=False, save_best_only=True, verbose=1)]

batch_size = 64
epochs = 20

data, labels = myutil.build_badminton_kill_data()
print(data.shape)

labels = tf.keras.utils.to_categorical(labels, num_classes=class_size)
print(labels.shape)
print(labels)

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

# model.fit(data, labels,
#           batch_size=batch_size, epochs=epochs,
#           validation_split=0.2, callbacks=callbacks)

model.save('lstm3_latest_kill.h5')

run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 25, 6], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "keras_lstm3_ios_kill"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
# Save the model.
with open('keras_lstm3_ios_kill.tflite', 'wb') as f:
    f.write(tflite_model)
