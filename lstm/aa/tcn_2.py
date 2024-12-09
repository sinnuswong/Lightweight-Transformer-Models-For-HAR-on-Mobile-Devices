# coding=utf-8

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tcn import TCN  # 假设使用tcn库

# 定义参数
window_size = 130  # 时间窗口长度
feature_size = 6   # 特征数量
hidden_size = 32  # TCN层的滤波器数量
dense_size = 64    # Dense层的神经元数量
output_size = 2    # 输出类别数量
L2 = 0.01          # L2正则化参数

# 创建模型
model = Sequential()

model.add(TCN(nb_filters=hidden_size,
               kernel_size=5,
               dilations=[1, 2, 4, 8, 16],
               return_sequences=False,
               kernel_initializer='orthogonal',
               input_shape=(window_size, feature_size)))

# 全连接层
model.add(Dense(dense_size, activation='relu', kernel_regularizer=l2(L2), name="Dense_1"))
model.add(Dense(dense_size, activation='relu', kernel_regularizer=l2(L2), name="Dense_2"))
model.add(Dense(output_size, activation='softmax', kernel_regularizer=l2(L2), name="Output"))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 输出模型摘要
model.summary()