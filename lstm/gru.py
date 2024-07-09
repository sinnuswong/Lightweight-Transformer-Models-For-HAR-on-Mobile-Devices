import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import myutil


class GRUModel(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define GRU layers
        self.gru_layers = [
            tf.keras.layers.GRU(units=hidden_size, return_sequences=True, return_state=True, dropout=dropout)
            for _ in range(num_layers - 1)
        ]
        self.gru_layers.append(
            tf.keras.layers.GRU(units=hidden_size, return_sequences=False, return_state=True, dropout=dropout))

        # Define Dense layer for classification
        self.dense = tf.keras.layers.Dense(units=output_size, activation='softmax')

    def call(self, inputs, states=None):
        x = inputs
        new_states = []
        for i, gru_layer in enumerate(self.gru_layers):
            if i == 0:
                if states is not None:
                    x, state = gru_layer(x, initial_state=states[i])
                else:
                    x, state = gru_layer(x)
            else:
                if states is not None:
                    x, state = gru_layer(x, initial_state=[states[i]])
                else:
                    x, state = gru_layer(x)
            new_states.append(state)
        x = self.dense(x)
        return x, new_states

    def init_states(self, batch_size):
        return [tf.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]


# 参数设置
features_size = 6
hidden_size = 256
num_layers = 1
output_size = 2
dropout = 0.2

# 初始化模型
model = GRUModel(features_size, hidden_size, num_layers, output_size, dropout)
dummy_input = tf.random.normal((1, 25, features_size))  # Batch size of 1, sequence length of 25, input size of 6
states = tf.random.normal((256,))
_ = model(dummy_input) # call model to build
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

timesteps = 25
features = 6
batch_size = 64

data, labels = myutil.build_badminton_hit_data()
print(data.shape)
print(labels.shape)
# 转换标签为sparse categorical
# labels = tf.keras.utils.to_categorical(labels, num_classes=2)
# 训练模型
states = model.init_states(batch_size)
model.summary()

history = model.fit(data, labels, epochs=2, batch_size=batch_size, validation_split=0.2)
# model.save('my_gru.h5')
#
# # 加载模型
# loaded_model = load_model('my_gru.h5')

# 显示模型结构和摘要

# 转换为TFLite模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_new_converter = True
tflite_model = converter.convert()

# 保存TFLite模型
with open('../my_gru.tflite', 'wb') as f:
    f.write(tflite_model)
#
# Epoch 49/50
# 28/28 [==============================] - 3s 96ms/step - loss: 0.0107 - accuracy: 0.9977 - val_loss: 0.1292 - val_accuracy: 0.9839
# Epoch 50/50
# 28/28 [==============================] - 3s 97ms/step - loss: 0.0128 - accuracy: 0.9965 - val_loss: 0.1688 - val_accuracy: 0.9816
