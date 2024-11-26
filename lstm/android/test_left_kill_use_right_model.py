# coding: utf-8
import shutil
import os
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import myutil


class KillRecognizer1:

    def __init__(self, model_file_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_file_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        print("kill self.interpreter.get_input_details()")
        print(self.interpreter.get_input_details())
        print("##############")
        print(self.interpreter.get_output_details())

        self.output_details = self.interpreter.get_output_details()

    def predict(self, window_frame, window_size, feature_size):  # 1,25,6
        if window_frame.shape[0] != 25:
            return 0
        window_frame = window_frame.reshape(1, window_size, feature_size)
        self.interpreter.set_tensor(self.input_details[0]["index"], window_frame.astype(np.float32))

        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]["index"])
        print("saaaa " + str(result))
        # self.interpreter.reset_all_variables()
        return result[0]


cur_kill_recognizer = KillRecognizer1(model_file_path='./left_model/left_lstm_kill.tflite')

hit_data_path = '/Users/sinnus/Desktop/ActivityData/badminton/c25/左撇子所有 0426/左撇子hit/yes'
data_save_path = '/Users/sinnus/Desktop/ActivityData/badminton/c25/左撇子处理/'
forehand_pingchou = data_save_path + 'forehand_pingchou'
high_long_hit = data_save_path + 'high_long_hit'
kill = data_save_path + 'kill'

data = []
# 遍历目录中的每个CSV文件
for filename in os.listdir(hit_data_path):
    if filename.endswith(".csv"):
        p = hit_data_path + "/" + filename
        print(p)
        window_frame_data = myutil.load_data_from_csv(p)
        # print(window_frame_data.shape)
        # print(window_frame_data)
        window_frame_data[:, 0] = -window_frame_data[:, 0]
        window_frame_data[:, 3] = -window_frame_data[:, 3]
        # print(window_frame_data)
        kill_res = cur_kill_recognizer.predict(window_frame_data, window_size=25,
                                               feature_size=6)
        print(kill_res)
        # if kill_res[0] > kill_res[1] and kill_res[0] > kill_res[2] and kill_res[0] > 0.5:
        #     shutil.copy2(p, forehand_pingchou)
        # elif kill_res[1] > kill_res[0] and kill_res[1] > kill_res[2] and kill_res[1] > 0.5:
        #     shutil.copy2(p, high_long_hit)
        # elif kill_res[2] > kill_res[0] and kill_res[2] > kill_res[1] :
        #     print(kill_res[2])
        #     shutil.copy2(p, kill)
