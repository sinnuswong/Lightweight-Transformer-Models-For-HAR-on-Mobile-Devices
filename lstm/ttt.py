import tensorflow as tf
from keras import Input
from tensorflow.keras.models import Model

l2 = tf.keras.regularizers.L2
Adam = tf.keras.optimizers.Adam

hidden_size = 128
window_size = 25
feature_size = 6
class_size = 2
L2 = 0.000001


# model = tf.keras.Sequential(name='sequential_1')
#
# model.add(tf.keras.layers.LSTM(hidden_units, return_sequences=True, input_shape=(window_size, feature_size),
#                                kernel_initializer='orthogonal', kernel_regularizer=l2(L2), recurrent_regularizer=l2(L2),
#                                bias_regularizer=l2(L2), name="LSTM_1"))
#
# model.add(tf.keras.layers.Flatten(name='Flatten'))
# model.add(tf.keras.layers.Dense(hidden_units, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
#                                 name="Dense_1"))
# model.add(tf.keras.layers.Dense(hidden_units, activation='relu', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
#                                 name="Dense_2"))
# model.add(tf.keras.layers.Dense(class_size, activation='softmax', kernel_regularizer=l2(L2), bias_regularizer=l2(L2),
#                                 name="Output"))
# model.summary()


class CustomLSTMModel(tf.keras.Model):
    def __init__(self, hidden_size, class_size):
        super(CustomLSTMModel, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True, kernel_initializer='orthogonal',
                                          kernel_regularizer=l2, recurrent_regularizer=l2, bias_regularizer=l2,
                                          name="LSTM_1")
        self.flatten = tf.keras.layers.Flatten(name='Flatten')
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=l2, bias_regularizer=l2,
                                            name="Dense_1")
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=l2, bias_regularizer=l2,
                                            name="Dense_2")
        self.output_layer = tf.keras.layers.Dense(class_size, activation='softmax', kernel_regularizer=l2,
                                                  bias_regularizer=l2, name="Output")

    def call(self, inputs, states):
        x = self.lstm1(inputs, initial_state=states[0])
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.output_layer(x)
        return outputs, states


input_shape = (25, 6)

inputs = Input(shape=input_shape)
hidden_state_input = Input(shape=(hidden_size,))
cell_state_input = Input(shape=(hidden_size,))

lstm_model = CustomLSTMModel(hidden_size, class_size)
outputs, hidden_state_output, cell_state_output = lstm_model(inputs, (hidden_state_input, cell_state_input))

model = Model(inputs=[inputs, hidden_state_input, cell_state_input],
              outputs=[outputs, hidden_state_output, cell_state_output])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型总结
model.summary()
