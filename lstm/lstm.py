import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint


class LSTMModel(tf.keras.Model):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(units=128, return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=2, activation='softmax')

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.dense3(x)

        return output


# 构建模型
model = LSTMModel()

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 构造虚拟数据
num_samples = 1000
window_size = 25
feature_size = 6
batch_size = 64

data = np.random.rand(1000, 25, 6)  # (samples, timesteps, features)
aa1 = [1 for i in range(500)]
aa2 = [0 for i in range(500)]
a1 = np.array(aa1 + aa2)

labels = a1
print(data.shape)
print(labels.shape)
# 转换标签为sparse categorical
# labels = tf.keras.utils.to_categorical(labels, num_classes=2)


callbacks = [
    ModelCheckpoint("__best_my_model_test.h5", save_weights_only=False, save_best_only=True,
                    verbose=1)]
# 训练模型
model.fit(data, labels, epochs=4, batch_size=64, validation_split=0.2, callbacks=callbacks)
model.save('__my_model_test.h5', save_format='tf')

# 加载模型
# loaded_model = load_model('my_model.h5')

# 显示模型结构和摘要
model.summary()

run_model = tf.function(lambda x: model(x))

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, window_size, feature_size], model.inputs[0].dtype))

# model directory.
MODEL_DIR = '__my_model_test'
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
# Save the model.
with open(MODEL_DIR + '.tflite', 'wb') as f:
    f.write(tflite_model)
