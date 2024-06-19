import model
import tensorflow as tf
import coremltools as ct

input_shape = (128, 6)  # The shape of your input data
activityCount = 6  # Number of classification heads

HART = model.HART(input_shape, activityCount)
MobileHART = model.mobileHART_XS(input_shape, activityCount)
print(MobileHART)
print(MobileHART is tf.keras.Model)
model_path = './HART_Results/Tests/MobileHART_16frameLength_16TimeStep_192ProjectionSize_0.005LR/MotionSense/bestTrain.h5'
MobileHART.load_weights(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(MobileHART)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    f.flush()
    f.close()

mlmodel = ct.convert(MobileHART, source='tensorflow')
mlmodel.save("mobileHAR.mlmodel")
