import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import myutil
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, LSTM, Dense, Dropout, BatchNormalization
# Define your model using Sequential or Functional API
from keras.regularizers import l2 #regularization function
from tensorflow.keras.optimizers import Adam  #perform gradient descent
from tensorflow.math import reduce_prod
import myutil

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(25, 6),
               kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), name='lstm_input_1'))

model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), name='Dense_2'))
model.add(tf.keras.layers.BatchNormalization())

model.add(Flatten(name='Flatten_3'))

model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), name='Dense_4'))
model.add(BatchNormalization())

model.add(Dense(2, activation='softmax',
                kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), name='output'))

# compiling the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
len(model.trainable_weights)

model.summary()

# 构造虚拟数据
num_samples = 1000
timesteps = 25
features = 6
batch_size = 64

data, labels = myutil.build_badminton_hit_data()
print(data.shape)
print(labels.shape)
# 转换标签为sparse categorical
labels = tf.keras.utils.to_categorical(labels, num_classes=2)
print(labels.shape)
# 训练模型
model.fit(data, labels, epochs=20, batch_size=64, validation_split=0.2)
model.save('my_model.h5')

# 加载模型
loaded_model = load_model('my_model.h5')

# 显示模型结构和摘要
loaded_model.summary()

# 转换为TFLite模型
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_new_converter = True
tflite_model = converter.convert()

# 保存TFLite模型
with open('../my_model.tflite', 'wb') as f:
    f.write(tflite_model)

