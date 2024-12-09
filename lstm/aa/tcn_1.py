import tensorflow as tf
import numpy as np
from keras.layers import ConvLSTM2D
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint
import os
from tcn import TCN

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
hit_data_path = '/Users/sinnus/Desktop/ActivityData/badminton/c130/1020left/hit'
save_model_base_path = current_directory + os.sep + 'left_model'
model_name = 'left_tcn_hit'
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
epochs = 9  # (128,20: 20, 8) (32,20: 15, 16)

L2 = 0.000001

model = tf.keras.Sequential(name='sequential_1')
model.add(TCN(nb_filters=hidden_size,
              kernel_size=6,  # 卷积核大小
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
