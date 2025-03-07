'''
    Copyright (c) 2018-2020
    Jianjia Ma
    majianjia@live.com
    SPDX-License-Identifier: Apache-2.0
    Change Logs:
    Date           Author       Notes
    2019-02-12     Jianjia Ma   The first version
'''

import matplotlib.pyplot as plt
import os

from tensorflow.keras import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model, save_model
import tensorflow as tf
import numpy as np
from nnom import *

model_name = 'model.h5'


def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


def one_hot(y_, n_classes=6):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def plot_raw(data):
    list = range(0, data.shape[0], int(data.shape[0] / 10))
    for i in list:
        dat = data[int(i), :, :]
        print(dat.shape)
        plt.plot(dat)
        plt.title('raw')
        plt.show()


def normalize(data):
    data[:, :, 0:3] = data[:, :, 0:3] / max(abs(np.max(data[:, :, 0:3])), abs(np.min(data[:, :, 0:3])))
    data[:, :, 3:6] = data[:, :, 3:6] / max(abs(np.max(data[:, :, 3:6])), abs(np.min(data[:, :, 3:6])))
    data[:, :, 6:9] = data[:, :, 6:9] / max(abs(np.max(data[:, :, 6:9])), abs(np.min(data[:, :, 6:9])))
    return data


def train(x_train, y_train, x_test, y_test, batch_size=64, epochs=100):
    inputs = Input(shape=x_train.shape[1:])
    # x=inputs
    x = Conv1D(9, kernel_size=(9), strides=(3), padding='same')(inputs)
    x = BatchNormalization()(x)

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

    # x1 = LSTM(32, return_sequences=True)(x)
    # x2 = LSTM(32, return_sequences=True, go_backwards=True)(x)
    # x = add([x1, x2])
    x = LSTM(32, return_sequences=False)(x)
    # x1 = GRU(32, return_sequences=True)(x)
    # x2 = GRU(32, return_sequences=True, go_backwards=True)(x)
    # x = concatenate([x1, x2], axis=-1)

    # x = GRU(32, return_sequences=True)(x)
    # x = GRU(32, return_sequences=True)(x)

    # x = Flatten()(x)
    x = Dense(32)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(6)(x)
    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                        verbose=2, validation_data=(x_test, y_test))

    # free the session to avoid nesting naming while we load the best model after.
    save_model(model, model_name)
    del model
    tf.keras.backend.clear_session()
    return history


def main():
    ## cpu normally run fast in 1-D data
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if (not os.path.exists('data/UCI HAR Dataset')):
        raise Exception('Please download the dataset and unzip into "data/UCI HAR Dataset/"')

    epochs = 10

    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    # Output classes to learn how to classify
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    DATA_PATH = "data/"
    MODEL_PATH = 'best_model.h5'

    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)

    TRAIN = "train/"
    TEST = "test/"

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]

    x_train = load_X(X_train_signals_paths)
    x_test = load_X(X_test_signals_paths)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"



    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)
    y_test_original = y_test  # save for CI test
    print(x_train.shape)
    print(y_train)
    y_train = one_hot(y_train, 6)
    y_test = one_hot(y_test, 6)

    # normolized each sensor, to range -1~1
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # test, ranges
    print("train acc range", np.max(x_train[:, :, 0:3]), np.min(x_train[:, :, 0:3]))
    print("train acc2 range", np.max(x_train[:, :, 6:9]), np.min(x_train[:, :, 6:9]))
    print("train gyro range", np.max(x_train[:, :, 3:6]), np.min(x_train[:, :, 3:6]))
    print("test acc range", np.max(x_test[:, :, 0:3]), np.min(x_test[:, :, 0:3]))
    print("test acc2 range", np.max(x_test[:, :, 6:9]), np.min(x_test[:, :, 6:9]))
    print("test gyro range", np.max(x_test[:, :, 3:6]), np.min(x_test[:, :, 3:6]))

    # cut test
    # only use 1000 for test
    # x_test = x_test[:1000]
    # y_test = y_test[:1000]

    # generate binary test data, convert range to [-128 127] for mcu
    x_test_bin = np.clip(x_test * 128, -128, 127)
    x_train_bin = np.clip(x_train * 128, -128, 127)
    generate_test_bin(x_test_bin, y_test, name='test_data.bin')
    generate_test_bin(x_train_bin, y_train, name='train_data.bin')

    # train model
    history = train(x_train,y_train, x_test, y_test, batch_size=128, epochs=epochs)

    # get best model
    model = load_model(model_name)

    # evaluate
    scores = evaluate_model(model, x_test, y_test)

    # save weight
    generate_model(model, x_test[:200], name='weights.h')


if __name__ == "__main__":
    main()