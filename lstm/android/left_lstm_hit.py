import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import coremltools as ct
from keras.models import load_model
import numpy as np
import myutil_huawei
import os

current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_directory = os.path.dirname(current_file_path)

l2 = tf.keras.regularizers.L2
L2 = 0.000001


class LSTMModel(tf.keras.Model):
    def __init__(self, hidden_size, class_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True,
                                         kernel_initializer='orthogonal', kernel_regularizer=l2(L2),
                                         recurrent_regularizer=l2(L2),
                                         bias_regularizer=l2(L2), name="LSTM_1")
        self.flatten = tf.keras.layers.Flatten(name='Flatten')

        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=l2(L2),
                                            bias_regularizer=l2(L2),
                                            name="Dense_1")
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=l2(L2),
                                            bias_regularizer=l2(L2),
                                            name="Dense_2")
        self.Output = tf.keras.layers.Dense(class_size, activation='softmax', kernel_regularizer=l2(L2),
                                            bias_regularizer=l2(L2),
                                            name="Output")

    def call(self, inputs, states):
        # inputs: (batch_size, time_steps, feature_size)
        # states: a tuple of (hidden_state, cell_state)
        x, hidden_state, cell_state = self.lstm(inputs, initial_state=states)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.Output(x)
        return outputs, hidden_state, cell_state

    def init_states(self, batch_size):
        return [tf.zeros((batch_size, self.hidden_size)), tf.zeros((batch_size, self.hidden_size))]


# config
window_size = 130
feature_size = 6
num_classes = 2

hidden_size = 128
output_size = num_classes
input_shape = (window_size, feature_size)
batch_size = 128
num_epochs = 20
hit_data_path = '/Users/sinnus/Desktop/ActivityData/badminton/c130/1020left/hit'
save_model_base_path = current_directory + os.sep + 'left_model'
model_name = 'left_lstm_hit'
save_model_path_no_extension = save_model_base_path + os.sep + model_name

# 创建模型
inputs = Input(shape=input_shape)
hidden_state_input = Input(shape=(hidden_size,))
cell_state_input = Input(shape=(hidden_size,))

lstm_model = LSTMModel(hidden_size, output_size)
outputs, hidden_state_output, cell_state_output = lstm_model(inputs, (hidden_state_input, cell_state_input))

model = Model(inputs=[inputs, hidden_state_input, cell_state_input],
              outputs=[outputs, hidden_state_output, cell_state_output])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型总结
model.summary()

data, labels = myutil_huawei.build_badminton_hit_data(hit_data_path=hit_data_path)
print(data.shape)

labels = to_categorical(labels, num_classes=num_classes)
print(labels.shape)

train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# 自定义训练循环

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')

    # 初始化状态
    initial_hidden_state = np.zeros((batch_size, hidden_size), dtype=np.float32)
    initial_cell_state = np.zeros((batch_size, hidden_size), dtype=np.float32)

    # 训练集
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]

        # 确保批量大小一致
        if len(batch_data) < batch_size:
            continue

        # 训练一步
        with tf.GradientTape() as tape:
            predictions, _, _ = model([batch_data, initial_hidden_state, initial_cell_state], training=True)
            loss = tf.keras.losses.categorical_crossentropy(batch_labels, predictions)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f'Batch {i // batch_size + 1}/{len(train_data) // batch_size}, Loss: {loss.numpy():.4f}')

    # 验证集
    val_loss = 0
    val_accuracy = 0
    val_steps = 0

    for i in range(0, len(val_data), batch_size):
        batch_data = val_data[i:i + batch_size]
        batch_labels = val_labels[i:i + batch_size]

        if len(batch_data) < batch_size:
            continue

        predictions, _, _ = model([batch_data, initial_hidden_state, initial_cell_state], training=False)
        loss = tf.keras.losses.categorical_crossentropy(batch_labels, predictions)
        loss = tf.reduce_mean(loss)

        accuracy = tf.keras.metrics.categorical_accuracy(batch_labels, predictions)
        accuracy = tf.reduce_mean(accuracy)

        val_loss += loss.numpy()
        val_accuracy += accuracy.numpy()
        val_steps += 1

    val_loss /= val_steps
    val_accuracy /= val_steps

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


# 保存模型的函数
def save_tflite(model, window_size, feature_size, hidden_size):
    # 用 tf.function 包装模型
    @tf.function
    def serve(inputs, hidden_state_input, cell_state_input):
        return model([inputs, hidden_state_input, cell_state_input])

    # 获取具体函数
    concrete_func = serve.get_concrete_function(
        tf.TensorSpec([1, window_size, feature_size], model.inputs[0].dtype),
        tf.TensorSpec([1, hidden_size], model.inputs[1].dtype),
        tf.TensorSpec([1, hidden_size], model.inputs[2].dtype))

    # 保存模型
    model.save(save_model_path_no_extension + '.keras')

    MODEL_DIR = save_model_path_no_extension
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

    # 转换为TFLite模型
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    tflite_model = converter.convert()

    # 保存TFLite模型
    with open(save_model_path_no_extension + '.tflite', 'wb') as f:
        f.write(tflite_model)
    coreml_model = ct.convert([concrete_func])
    coreml_model.save(save_model_path_no_extension + '.mlmodel')


# 示例调用
save_tflite(model, window_size, feature_size, hidden_size)