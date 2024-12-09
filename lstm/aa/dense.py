# coding=utf-8

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import l2

# 定义参数
window_size = 130  # 时间窗口长度
feature_size = 6   # 特征数量
dense_size = 32    # Dense层的神经元数量
output_size = 2    # 输出类别数量
L2 = 0.01          # L2正则化参数

# 创建模型
model = Sequential()

# 将输入展平为一维
model.add(Flatten(input_shape=(window_size, feature_size)))

# 添加全连接层
model.add(Dense(dense_size, activation='relu', kernel_regularizer=l2(L2), name="Dense_1"))
model.add(Dense(dense_size, activation='relu', kernel_regularizer=l2(L2), name="Dense_2"))
model.add(Dense(output_size, activation='softmax', kernel_regularizer=l2(L2), name="Output"))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 输出模型摘要
model.summary()
