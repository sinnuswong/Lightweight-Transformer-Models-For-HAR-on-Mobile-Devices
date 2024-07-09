import model
import tensorflow as tf
import coremltools as ct
import myutil

data = myutil.build_badminton_data()
data = data[0]
input_shape = (25, 6)  # The shape of your input data
activityCount = 2  # Number of classification heads

HART = model.HART(input_shape, activityCount)
MobileHART = model.mobileHART_XS(input_shape, activityCount)
print(MobileHART)
print(MobileHART is tf.keras.Model)
model_path = './HART_Results/MobileHART_16frameLength_16TimeStep_192ProjectionSize_0.005LR/train_hit_c25/bestValcheckpoint.h5'
MobileHART.load_weights(model_path)

output = MobileHART.predict(data)
print("*********")
print(output)
print(output.shape)

converter = tf.lite.TFLiteConverter.from_keras_model(MobileHART)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('mobile_har_xs_bestval.tflite', 'wb') as f:
    f.write(tflite_model)
    f.flush()
    f.close()

# mlmodel = ct.convert(MobileHART, source='tensorflow')
# mlmodel.save("mobileHAR.mlmodel")
