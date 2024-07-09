import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import myutil

# Define your model using Sequential or Functional API
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256, return_sequences=True, input_shape=(25, 6)),
    tf.keras.layers.Dropout(0.5),  # 添加 Dropout 层
    tf.keras.layers.LSTM(units=256, return_sequences=False),
    tf.keras.layers.Dropout(0.5),  # 添加 Dropout 层
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 构造虚拟数据
num_samples = 1000
timesteps = 25
features = 6
batch_size = 64

data, labels = myutil.build_badminton_hit_data()
print(data.shape)
print(labels.shape)
# 转换标签为sparse categorical
# labels = tf.keras.utils.to_categorical(labels, num_classes=2)

# 训练模型
model.fit(data, labels, epochs=3, batch_size=64, validation_split=0.2)
model.save('my_model.h5')

# 加载模型
loaded_model = load_model('my_model.h5')

# 显示模型结构和摘要
loaded_model.summary()

# 转换为TFLite模型
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter._experimental_lower_tensor_list_ops = False
# converter.experimental_new_converter = True
tflite_model = converter.convert()
converter._experimental_lower_tensor_list_ops = False
# 保存TFLite模型
with open('../my_model.tflite', 'wb') as f:
    f.write(tflite_model)
#
# Epoch 49/50
# 28/28 [==============================] - 3s 96ms/step - loss: 0.0107 - accuracy: 0.9977 - val_loss: 0.1292 - val_accuracy: 0.9839
# Epoch 50/50
# 28/28 [==============================] - 3s 97ms/step - loss: 0.0128 - accuracy: 0.9965 - val_loss: 0.1688 - val_accuracy: 0.9816
