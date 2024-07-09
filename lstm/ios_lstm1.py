import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LSTM, Dense, ReLU, Input, Flatten
from tensorflow.keras.models import Model

# 定义模型
input_shape = (6, 1, 25)
inputs = Input(shape=input_shape)

# 卷积层
conv = Conv2D(filters=64, kernel_size=(1, 25), strides=(1, 25), padding='valid')(inputs)
conv = ReLU()(conv)

# 调整维度以匹配 LSTM 输入格式
conv = tf.keras.layers.Reshape(target_shape=(64,))(conv)

# LSTM 层
lstm_out, hidden_state, cell_state = LSTM(units=200, return_state=True)(conv)

# 全连接层1
dense1 = Dense(128, activation='relu')(lstm_out)

# 全连接层2
outputs = Dense(2, activation='softmax')(dense1)

# 创建模型
model = Model(inputs=inputs, outputs=outputs)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型总结
model.summary()
