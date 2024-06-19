import model
import tensorflow as tf
import coremltools as ct

input_shape = (128, 6)  # The shape of your input data
activityCount = 6  # Number of classification heads

HART = model.HART(input_shape, activityCount)
MobileHART = model.mobileHART_XS(input_shape, activityCount)
print(MobileHART)
print(MobileHART is tf.keras.Model)
model_path = 'F:\\Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices\\HART_Results\Tests\MobileHART_16frameLength_16TimeStep_192ProjectionSize_0.005LR\\MotionSense\\bestTrain.h5'
MobileHART.load_weights(model_path)




class MobileHART_XS(tf.keras.Model):
    def __init__(self, input_shape, activityCount, projectionDims=[96, 120, 144],
                 filterCount=[8, 16, 24, 32, 40, 48, 192], expansion_factor=4, mlp_head_units=[1024], dropout_rate=0.3):
        super(MobileHART_XS, self).__init__()

        # Initial conv-stem -> MV2 block.
        self.conv_block_accX = self.conv_block(filters=filterCount[0])
        self.conv_block_gyroX = self.conv_block(filters=filterCount[0])
        self.mv2Block_accX = self.mv2Block(expansion_factor, filterCount)
        self.mv2Block_gyroX = self.mv2Block(expansion_factor, filterCount)
        self.sensorWiseHART = self.sensorWiseHART(num_blocks=2, projection_dim=projectionDims[0])

        self.dense_projection = layers.Dense(projectionDims[0], activation=tf.nn.swish)
        self.dropout = layers.Dropout(dropout_rate)

        # Second MV2 -> MobileViT block.
        self.inverted_residual_block1 = self.inverted_residual_block(
            expanded_channels=projectionDims[0] * expansion_factor,
            output_channels=filterCount[4], strides=2)
        self.mobilevit_block1 = self.mobilevit_block(num_blocks=4, projection_dim=projectionDims[1])

        # Third MV2 -> MobileViT block.
        self.inverted_residual_block2 = self.inverted_residual_block(
            expanded_channels=projectionDims[1] * expansion_factor,
            output_channels=filterCount[5], strides=2)
        self.mobilevit_block2 = self.mobilevit_block(num_blocks=3, projection_dim=projectionDims[2])

        self.conv_block_final = self.conv_block(filters=filterCount[6], kernel_size=1, strides=1)

        # Classification head.
        self.global_avg_pool = layers.GlobalAvgPool1D(name="GAP")
        self.mlp = self.mlp(hidden_units=mlp_head_units, dropout_rate=dropout_rate)

        self.dense_output = layers.Dense(activityCount, activation="softmax")

    def call(self, inputs):
        accX = self.conv_block_accX(inputs[:, :, :3])
        gyroX = self.conv_block_gyroX(inputs[:, :, 3:])
        accX = self.mv2Block_accX(accX)
        gyroX = self.mv2Block_gyroX(gyroX)
        accX, gyroX = self.sensorWiseHART(accX, gyroX)
        x = tf.concat((accX, gyroX), axis=2)
        x = self.dense_projection(x)
        x = self.dropout(x)

        x = self.inverted_residual_block1(x)
        x = self.mobilevit_block1(x)

        x = self.inverted_residual_block2(x)
        x = self.mobilevit_block2(x)

        x = self.conv_block_final(x)

        x = self.global_avg_pool(x)
        x = self.mlp(x)

        outputs = self.dense_output(x)
        return outputs

    def conv_block(self, filters, kernel_size=3, strides=1, padding="same"):
        return layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

    def mv2Block(self, expansion_factor, filterCount):
        return None  # Implement your mv2Block function

    def sensorWiseHART(self, num_blocks, projection_dim):
        return None  # Implement your sensorWiseHART function

    def inverted_residual_block(self, expanded_channels, output_channels, strides):
        return None  # Implement your inverted_residual_block function

    def mobilevit_block(self, num_blocks, projection_dim):
        return None  # Implement your mobilevit_block function

    def mlp(self, hidden_units, dropout_rate):
        return None  # Implement your mlp function


converter = tf.lite.TFLiteConverter.from_keras_model(MobileHART)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    f.flush()
    f.close()

mlmodel = ct.convert(MobileHART, source='tensorflow', convert_to="mlprogram")
mlmodel.save("mobileHAR.mlmodel")
