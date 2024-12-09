import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dropout, Activation, Dense
from tensorflow.keras.models import Model

def tcn_layer(inputs, filters, kernel_size, dilation_rate):
    x = Conv1D(filters=filters, kernel_size=kernel_size,
               dilation_rate=dilation_rate, padding='causal')(inputs)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    return x

def build_tcn_model(input_shape):
    inputs = Input(shape=input_shape)

    # TCN Block
    x = tcn_layer(inputs, filters=32, kernel_size=3, dilation_rate=1)
    x = tcn_layer(x, filters=32, kernel_size=3, dilation_rate=2)
    x = tcn_layer(x, filters=32, kernel_size=3, dilation_rate=4)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Fully Connected Layer
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output Layer
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

# Define input shape (n, 130, 6)
input_shape = (130, 6)

# Build model
model = build_tcn_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Assuming you have your training data in x_train and y_train
# x_train.shape should be (n, 130, 6) and y_train.shape should be (n,)
# The following line would be used to train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32)
