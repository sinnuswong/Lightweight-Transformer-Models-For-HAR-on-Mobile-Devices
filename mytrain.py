#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import csv
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import time
import hickle as hkl
import random
import math
import logging
import shutil
import gc
import sys
import sklearn.manifold
import seaborn as sns
import argparse
import matplotlib.gridspec as gridspec

import model
import utils
from keras_flops import get_flops
import myutil

randomSeed = 1
os.environ['PYTHONHASHSEED'] = str(randomSeed)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set The Default Hyperparameters Here
architecture = "HART"  # MobileHART, HART

dataSetName = 'train_hit_c25'
# BALANCED, UNBALANCED
dataConfig = "BALANCED"

# Show training verbose: 0,1
showTrainVerbose = 1

# input window size
segment_size = 25

# input channel count
num_input_channels = 6

learningRate = 1e-3

# model drop out rate
dropout_rate = 0.3

# local epoch
localEpoch = 50
# or 4
frameLength = 16

timeStep = 16

positionDevice = ''
# ['chest','forearm','head','shin','thigh','upperarm','waist']
# ['nexus4', 'lgwatch','s3', 's3mini','gear','samsungold']
tokenBased = False

measureEnergy = False

# hyperparameter for the model

batch_size = 32
projection_dim = 192
filterAttentionHead = 4

# To adjust the number of blocks in for HART, Add or remove the conv kernel size here
# each conv kernel length is for one HART block
convKernels = [3, 7, 15, 31, 31, 31]


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size of the training')
    parser.add_argument('--localEpoch', type=int, default=localEpoch,
                        help='Number of epochs for training')
    parser.add_argument('--architecture', type=str, default=architecture,
                        help='Choose between HART or MobileHART')
    parser.add_argument('--projection_dim', type=int, default=projection_dim,
                        help='Size of the projection dimensions')
    parser.add_argument('--frame_length', type=int, default=frameLength,
                        help='Patch Size')
    parser.add_argument('--time_step', type=int, default=timeStep,
                        help='Stride Size')
    parser.add_argument('--dataset', type=str, default=dataSetName,
                        help='Dataset')
    parser.add_argument('--tokenBased', type=bool, default=tokenBased,
                        help='Use Token or Global Average Pooling')
    parser.add_argument('--positionDevice', type=str, default=positionDevice,
                        help='Test is done other position not in training, if empty, uses a 70/10/20 train/dev/test ratio ')
    args = parser.parse_args()
    return args


args = add_fit_args(argparse.ArgumentParser(description='Human Activity Recognition Transformer'))
localEpoch = args.localEpoch
batch_size = args.batch_size
architecture = args.architecture
projection_dim = args.projection_dim
frameLength = args.frame_length
timeStep = args.time_step
dataSetName = args.dataset
tokenBased = args.tokenBased
positionDevice = args.positionDevice

input_shape = (segment_size, num_input_channels)
projectionHalf = projection_dim // 2
projectionQuarter = projection_dim // 4

transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers

R = projectionHalf // filterAttentionHead
assert R * filterAttentionHead == projectionHalf

segmentTime = [x for x in range(0, segment_size - frameLength + timeStep, timeStep)]
print("segment Time ")
print(segmentTime)
assert R * filterAttentionHead == projectionHalf

# specifying activities and where the results will be stored
if dataSetName == 'train_hit_c25':
    ACTIVITY_LABEL = ['none', 'yes']

activityCount = 2

# 这里没看懂
architectureType = str(architecture) + '_' + str(int(frameLength)) + 'frameLength_' + str(timeStep) + 'TimeStep_' + str(
    projection_dim) + "ProjectionSize_" + str(learningRate) + 'LR'
if (tokenBased):
    architectureType = architectureType + "_tokenBased"

if (positionDevice != ''):
    architectureType = architectureType + "_PositionWise_" + str(positionDevice)
mainDir = './'
# ************************ config file path.
if (localEpoch < 20):
    architectureType = "Tests/" + str(architectureType)
filepath = mainDir + 'HART_Results/' + architectureType + '/' + dataSetName + '/'
os.makedirs(filepath, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

attentionPath = filepath + "attentionImages/"
os.makedirs(attentionPath, exist_ok=True)

bestModelPath = filepath + 'bestModels/'
os.makedirs(bestModelPath, exist_ok=True)

currentModelPath = filepath + 'currentModels/'
os.makedirs(currentModelPath, exist_ok=True)
# ************************

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
np.random.seed(randomSeed)
tf.keras.utils.set_random_seed(randomSeed)
tf.random.set_seed(randomSeed)
random.seed(randomSeed)

# ************************ data prepare

centralTrainData, centralTrainLabel = myutil.build_badminton_data()
print(centralTrainData.shape)
print(centralTrainLabel.shape)#     using a 70 10 20 ratio
centralTrainData, centralDevData, centralTrainLabel, centralDevLabel = train_test_split(centralTrainData,
                                                                                        centralTrainLabel,
                                                                                        test_size=0.125,
                                                                                        random_state=randomSeed)
print(centralTrainData.shape)
print(centralTrainLabel.shape)
print(centralDevData.shape)
print(centralDevLabel.shape)

_, centralTestData, _, centralTestLabel = train_test_split(centralTrainData,
                                                                                        centralTrainLabel,
                                                                                        test_size=0.1,
                                                                                        random_state=randomSeed)
print("***************fuck")
print(centralTrainData[0])
print(centralTrainLabel[0])
print(centralDevData[0])
print(centralDevLabel[0])
# ************************ data prepare

# One Hot of labels
centralTrainLabel = tf.one_hot(
    centralTrainLabel,
    activityCount,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
centralTestLabel = tf.one_hot(
    centralTestLabel,
    activityCount,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
centralDevLabel = tf.one_hot(
    centralDevLabel,
    activityCount,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)

optimizer = tf.keras.optimizers.Adam(learningRate)

if (architecture == "HART"):
    model_classifier = model.HART(input_shape, activityCount)
else:
    model_classifier = model.mobileHART_XS(input_shape, activityCount)
model_classifier.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)
# model_classifier.summary()
# ********************************************************************************************************************************************

checkpoint_filepath = filepath + "bestValcheckpoint.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

start_time = time.time()
# ********************************************** training start
print("********************** start train")
print(centralTrainData.shape)
print(centralTrainLabel.shape)
print(centralDevData.shape)
print(centralDevLabel.shape)
print(centralTrainLabel[0])
# (15069, 128, 6)
# (15069, 6)
# (2153, 128, 6)
# (2153, 6)
# tf.Tensor([0. 0. 1. 0. 0. 0.], shape=(6,), dtype=float32)

history = model_classifier.fit(
    x=centralTrainData,
    y=centralTrainLabel,
    validation_data=(centralDevData, centralDevLabel),
    batch_size=batch_size,
    epochs=localEpoch,
    verbose=showTrainVerbose,
    callbacks=[checkpoint_callback],
)
end_time = time.time() - start_time

model_classifier.save_weights(filepath + 'bestTrain.h5')
model_classifier.load_weights(checkpoint_filepath)
_, accuracy = model_classifier.evaluate(centralTestData, centralTestLabel)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

hkl.dump(history.history, filepath + 'history.hkl')
# Convert the model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model_classifier)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(filepath + architecture + '.tflite', 'wb') as f:
    f.write(tflite_model)


def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx


if (architecture == "HART"):
    finalAccMHAIndex = getLayerIndexByName(model_classifier, "AccMHA_" + str(len(convKernels) - 1))
    finalGyroMHAIndex = getLayerIndexByName(model_classifier, "GyroMHA_" + str(len(convKernels) - 1))
    finalInputsIndex = getLayerIndexByName(model_classifier, "normalizedInputs_" + str(len(convKernels) - 1))
    totalLayer = len(model_classifier.layers)
    classTokenIndex = totalLayer - 4
else:
    finalAccMHAIndex = getLayerIndexByName(model_classifier, "AccMHA_1")
    finalGyroMHAIndex = getLayerIndexByName(model_classifier, "GyroMHA_1")
    finalInputsIndex = getLayerIndexByName(model_classifier, "normalizedInputs_1")
    classTokenIndex = getLayerIndexByName(model_classifier, "GAP")

y_pred = np.argmax(model_classifier.predict(centralTestData), axis=-1)

y_test = np.argmax(centralTestLabel, axis=-1)
weightVal_f1 = f1_score(y_test, y_pred, average='weighted')
microVal_f1 = f1_score(y_test, y_pred, average='micro')
macroVal_f1 = f1_score(y_test, y_pred, average='macro')

modelStatistics = {
    "Results on server model on ALL testsets": '',
    "\nTrain:": utils.roundNumber(np.max(history.history['accuracy'])),
    "\nValidation:": utils.roundNumber(np.max(history.history['val_accuracy'])),
    "\nTest weighted f1:": utils.roundNumber(weightVal_f1),
    "\nTest micro f1:": utils.roundNumber(microVal_f1),
    "\nTest macro f1:": utils.roundNumber(macroVal_f1),
}
with open(filepath + 'GlobalACC.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(modelStatistics.items())

xLabel = np.arange(0, 2.58, 0.02)
idx = np.round(np.linspace(0, len(xLabel) - 1, 8)).astype(int)
s = pd.Series(y_test)
indices = [v.values[2] for k, v in s.groupby(s).groups.items()]
segmentTime = [x for x in range(0, segment_size - frameLength + timeStep, timeStep)]
inputModel = tf.keras.Model(inputs=model_classifier.inputs, outputs=model_classifier.layers[finalInputsIndex].output)

for index, classLoc in enumerate(indices):
    inputsToAttention = inputModel(np.expand_dims(centralTestData[classLoc], 0))
    _, attentionAccWeights = model_classifier.layers[finalAccMHAIndex](inputsToAttention, return_attention_scores=True)
    _, attentionGyroWeights = model_classifier.layers[finalGyroMHAIndex](inputsToAttention,
                                                                         return_attention_scores=True)
    if (tokenBased):
        attentionScores = np.mean(attentionAccWeights[0], axis=0)[0, 1:]
    else:
        attentionScores = np.mean(attentionAccWeights[0], axis=0)[0, :]
    attentionScoresNorm = ((attentionScores - min(attentionScores)) / (
            max(attentionScores) - min(attentionScores)) - 1) * - 0.5
    gs = gridspec.GridSpec(2, 1)
    fig = plt.figure()
    plt.title("Attention Map for " + ACTIVITY_LABEL[index] + " ", size=16)
    plt.margins(x=0)
    plt.tick_params(
        axis='both',
        which='both',
        labelleft=False,
        left=False,
        bottom=False,
        top=False,
        labelbottom=False)

    ax = fig.add_subplot(gs[0])
    ax.margins(x=0)

    ax.plot(centralTestData[classLoc][:, 0], label="x-axis")
    ax.plot(centralTestData[classLoc][:, 1], label="y-axis")
    ax.plot(centralTestData[classLoc][:, 2], label="z-axis")
    for barIndex, starTime in enumerate(segmentTime):
        ax.axvspan(starTime, starTime + frameLength, facecolor='black', alpha=float(attentionScoresNorm[barIndex]),
                   zorder=4)

    ax.set_ylabel(r'Acc ($m/s^2$)', size=16)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        labelbottom=False)
    plt.legend(loc='upper right', framealpha=0.7)

    if (tokenBased):
        attentionScores = np.mean(attentionGyroWeights[0], axis=0)[0, 1:]
    else:
        attentionScores = np.mean(attentionGyroWeights[0], axis=0)[0, :]
    attentionScoresNorm = ((attentionScores - min(attentionScores)) / (
            max(attentionScores) - min(attentionScores)) - 1) * - 0.5

    ax = fig.add_subplot(gs[1], sharex=ax)
    ax.margins(x=0)

    ax.plot(centralTestData[classLoc][:, 3], label="x-axis")
    ax.plot(centralTestData[classLoc][:, 4], label="y-axis")
    ax.plot(centralTestData[classLoc][:, 5], label="z-axis")

    for barIndex, starTime in enumerate(segmentTime):
        ax.axvspan(starTime, starTime + frameLength, facecolor='black', alpha=float(attentionScoresNorm[barIndex]),
                   zorder=99)

    ax.set_ylabel(r'Gyro (rad/s)', size=16)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plt.xticks([0, 32, 64, 96, 128])
    fig.get_axes()[2].set_xticklabels([0, 0.64, 1.28, 1.9, 2.56])
    plt.xlabel("Time (s)", size=16)
    plt.margins(x=0)

    plt.savefig(attentionPath + ACTIVITY_LABEL[index] + "MeanHeadAttention.png", bbox_inches="tight")
    plt.show()
    plt.clf()

utils.plot_learningCurve(history, localEpoch, filepath)

totalLayer = len(model_classifier.layers)
classTokenIndex = totalLayer - 4
intermediateModel = utils.extract_intermediate_model_from_base_model(model_classifier, classTokenIndex)

perplexity = 30.0
embeddings = intermediateModel.predict(centralTestData, batch_size=batch_size)
del intermediateModel
tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=showTrainVerbose, random_state=randomSeed)
tsne_projections = tsne_model.fit_transform(embeddings)
labels_argmax = np.argmax(centralTestLabel, axis=-1)
unique_labels = np.unique(labels_argmax)

if ((dataSetName == 'RealWorld' or dataSetName == 'HHAR') and positionDevice == ''):
    utils.projectTSNEWithPosition(dataSetName, architecture + "_TSNE_Embeds", filepath, ACTIVITY_LABEL, labels_argmax,
                                  orientationsNames, clientOrientationTest, tsne_projections, unique_labels)
else:
    utils.projectTSNE(architecture + "_TSNE_Embeds", filepath, ACTIVITY_LABEL, labels_argmax, tsne_projections,
                      unique_labels)

results = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(results, index=[i for i in ACTIVITY_LABEL],
                     columns=[i for i in ACTIVITY_LABEL])
plt.figure(figsize=(14, 14))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, cbar=False)
plt.ylabel('Prediction')
plt.xlabel('Ground Truth')
plt.savefig(filepath + 'HeatMap.png')

# Saves the training time per round
flops = get_flops(model_classifier, batch_size=1)
modelStatistics = {
    "NumberOfParams:": str(model_classifier.count_params()),
    "flops": str(flops),
    "TrainingTime": str(end_time / 360),
}
with open(filepath + 'numberOfParams.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(modelStatistics.items())

print("Total Number of Parapmeters : " + str(model_classifier.count_params()))

print("Training Done!")
