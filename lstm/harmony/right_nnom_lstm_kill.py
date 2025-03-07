import tensorflow as tf
import numpy as np
from keras.layers import ConvLSTM2D
from tensorflow.keras.models import load_model, save_model
import myutil_huawei
from sklearn.model_selection import train_test_split
import coremltools as ct
from keras.callbacks import ModelCheckpoint
import os
from nnom import *

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
kill_data_path = '/Users/sinnus/Desktop/ActivityData/badminton/c130/0921right/kill_high_long_hit'
save_model_base_path = current_directory + os.sep + 'right_model'
model_name = 'right_lstm_kill'
save_model_path_no_extension = save_model_base_path + os.sep + model_name
# 获取当前文件所在的目录

l2 = tf.keras.regularizers.L2
Adam = tf.keras.optimizers.Adam

# configs
window_size = 65
feature_size = 6
num_classes = 3

hidden_size = 32
dense_size = 32
output_size = num_classes
input_shape = (window_size, feature_size)
batch_size = 128  # (128,50: 14,16), (32,50: 17,14),
epochs = 60  # (128,20: 20, 8) (32,20: 15, 16)
max_acc = 8.0
max_gyro = 40.033665
L2 = 0.000001
# 32, 10, 128

data, labels = myutil_huawei.build_badminton_kill_data(kill_data_path=kill_data_path)
print(data.shape)

labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
print(labels.shape)
print(labels)


# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
#                     callbacks=callbacks)

def normalize(data):
    print("******")
    print(max(abs(np.max(data[:, :, 0:3])), abs(np.min(data[:, :, 0:3]))))
    print(max(abs(np.max(data[:, :, 3:6])), abs(np.min(data[:, :, 3:6]))))

    data[:, :, 0:3] = data[:, :, 0:3] / max(abs(np.max(data[:, :, 0:3])), abs(np.min(data[:, :, 0:3])))
    data[:, :, 3:6] = data[:, :, 3:6] / max(abs(np.max(data[:, :, 3:6])), abs(np.min(data[:, :, 3:6])))
    # data[:, :, 6:9] = data[:, :, 6:9] / max(abs(np.max(data[:, :, 6:9])), abs(np.min(data[:, :, 6:9])))

    return data


def train(x_train, y_train, x_test, y_test, batch_size=64, epochs=100):
    inputs = Input(shape=x_train.shape[1:])
    x = inputs
    # x = Conv1D(6, kernel_size=(6), strides=(3), padding='same')(inputs)
    # x = BatchNormalization()(x)

    # you can use either of the format below.
    # x = RNN(SimpleRNNCell(16), return_sequences=True)(x)
    # x = SimpleRNN(16, return_sequences=True)(x)

    # x2 = RNN(LSTMCell(32), return_sequences=True)(x)
    # x1 = LSTM(32, return_sequences=True, go_backwards=True)(x)
    # x = concatenate([x1, x2], axis=-1)
    #
    # # Bidirectional with concatenate. (not working yet)
    # x1 = RNN(GRUCell(16), return_sequences=True)(x)
    # x2 = GRU(16, return_sequences=True, go_backwards=True)(x)
    # x = concatenate([x1, x2], axis=-1)

    # Bidirectional with concatenate. (not working yet)

    x = LSTM(32, return_sequences=False)(x)
    # x = LSTM(32, return_sequences=True, go_backwards=True)(x)
    # x = add([x1, x2])
    # x = LSTM(64, return_sequences=False)(x)
    # x = Conv1D(6, kernel_size=(6), strides=(3), padding='same')(inputs)
    # x = tf.keras.layers.MaxPooling2D(pool_size=(5, 1), strides=(5, 1))(x)
    # x = MaxPooling1D(pool_size=3, strides=3)(x)
    # x = Flatten()(x)

    # x = MaxPooling1D()
    x = Dense(32)(x)
    x = Dense(32)(x)
    x = Dense(num_classes)(x)
    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint(save_model_path_no_extension + '.h5', save_weights_only=False, save_best_only=True,
                        verbose=1)]

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks)

    # free the session to avoid nesting naming while we load the best model after.
    save_model(model, model_name)
    del model
    tf.keras.backend.clear_session()
    return history


x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# normolized each sensor, to range -1~1
x_train = normalize(x_train)
x_val = normalize(x_val)

# test, ranges
print("train acc range", np.max(x_train[:, :, 0:3]), np.min(x_train[:, :, 0:3]))
print("train gyro range", np.max(x_train[:, :, 3:6]), np.min(x_train[:, :, 3:6]))
print("test acc range", np.max(x_val[:, :, 0:3]), np.min(x_val[:, :, 0:3]))
print("test gyro range", np.max(x_val[:, :, 3:6]), np.min(x_val[:, :, 3:6]))

# generate binary test data, convert range to [-128 127] for mcu
x_test_bin = np.clip(x_val * 128, -128, 127)
x_train_bin = np.clip(x_train * 128, -128, 127)
generate_test_bin(x_test_bin, y_val, name='test_data.bin')
generate_test_bin(x_train_bin, y_train, name='train_data.bin')

# train model
history = train(x_train, y_train, x_val, y_val, batch_size=batch_size, epochs=epochs)

# get best model
model = load_model(model_name)

# evaluate
scores = evaluate_model(model, x_val, y_val)
print("evaluate scores " + str(scores))

# save weight
generate_model(model, x_val[:200], name="model_"+model_name + '.h',quantize_method='kld')
