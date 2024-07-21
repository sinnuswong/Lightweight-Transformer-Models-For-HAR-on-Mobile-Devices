import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import myutil


# data, labels = myutil.build_badminton_hit_data()


def read_floats_from_binary(file_path):
    # 读取二进制文件中的所有 float 数据
    data = np.fromfile(file_path, dtype=np.float32)
    delete_indices = np.arange(0, len(data), 7)
    # 删除指定索引的元素
    new_data = np.delete(data, delete_indices)
    return new_data


def reshape_data_to_frames(data, feature_size=6, window_size=25):
    # 计算总的帧数
    total_frames = data.shape[0] // feature_size

    # 计算完整组的数量
    num_groups = total_frames // window_size

    # 截断不完整的数据
    truncated_data = data[:num_groups * window_size * feature_size]

    # 重新形状为 (num_groups, group_size, frame_size)
    reshaped_data = truncated_data.reshape((num_groups, window_size, feature_size))
    return reshaped_data


def reshape_data_to_motion_frame(data, feature_size=6):
    total_frames = data.shape[0] // feature_size
    # 重新形状为 (num_groups, group_size, frame_size)
    reshaped_data = data.reshape((total_frames, feature_size))
    return reshaped_data


aaaaaq = "./error910A7242__1713273736716__CN__Watch7,3__right__right__2024_04_16_21_22_16.ad"
# 高远球0CABAC36__1720177032222__CN__S7_41mm__right__right__2024_07_05_18_57_12.ad
file_path = '../aatest/F954FC8A__1718433170022__CN__S8_45mm__right__right__2024_06_15_14_32_50.ad'  # './C13D0A02__1718283752630__CN__Watch7,2__right__right__2024_06_13_21_02_32.ad'

# 读取二进制文件中的 float 数据
float_data = read_floats_from_binary(file_path)

motion_frame = reshape_data_to_motion_frame(float_data, feature_size=6)
print(motion_frame.shape)

# 将数据重新形状为 (x, 25, 6)
reshaped_data = reshape_data_to_frames(float_data, feature_size=6, window_size=25)
print(reshaped_data.shape)


def test_sequence():
    hits = []
    hit = 0
    for i in range(3545):
        interpreter.set_tensor(input_details[0]["index"], reshaped_data[i:i + 1, :, :].astype(np.float32))
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]["index"])
        print(result[0])
        # print(reshaped_data[i:i + 1, :, :].astype(np.float32))
        hits.append(result[0][1])
        # interpreter.reset_all_variables()

    for i in range(len(hits)):
        if hits[i] > 0.9:
            hit = hit + 1
    print(hit)


class HitRecognizer:

    def __init__(self, model_file_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_file_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        print("self.interpreter.get_input_details()")
        print(self.interpreter.get_input_details())
        print("##############")
        print(self.interpreter.get_output_details())

        self.output_details = self.interpreter.get_output_details()
        self.state1 = np.zeros((1, 128), dtype=np.float32)
        self.state2 = np.zeros((1, 128), dtype=np.float32)

    def predict(self, window_frame, window_size, feature_size):  # 1,25,6
        if window_frame.shape[0] != 25:
            return 0
        window_frame = window_frame.reshape(1, window_size, feature_size)
        # self.interpreter.set_tensor(self.input_details[0]["index"], window_frame.astype(np.float32))
        self.interpreter.set_tensor(self.input_details[0]["index"], self.state1)
        self.interpreter.set_tensor(self.input_details[1]["index"], self.state2)
        self.interpreter.set_tensor(self.input_details[2]["index"], window_frame.astype(np.float32))

        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[1]["index"])
        self.state1 = self.interpreter.get_tensor(self.output_details[0]["index"])
        self.state2 = self.interpreter.get_tensor(self.output_details[2]["index"])

        ###
        self.interpreter.reset_all_variables()
        # self.interpreter.set_tensor(self.input_details[0]["index"], np.zeros((1, 128), dtype=np.float32))
        # self.interpreter.set_tensor(self.input_details[1]["index"], np.zeros((1, 128), dtype=np.float32))
        # self.interpreter.set_tensor(self.input_details[2]["index"], window_frame.astype(np.float32))
        #
        # self.interpreter.invoke()
        # self.state1 = self.interpreter.get_tensor(self.output_details[0]["index"])
        # self.state2 = self.interpreter.get_tensor(self.output_details[2]["index"])
        #
        # print(result[0])
        # if result[0][1] > 0.7:
        #     self.state1 = np.zeros((1, 128), dtype=np.float32)
        #     self.state2 = np.zeros((1, 128), dtype=np.float32)
        return result[0][1]


class KillRecognizer:

    def __init__(self, model_file_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_file_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        print("self.interpreter.get_input_details()")
        print(self.interpreter.get_input_details())
        print("##############")
        print(self.interpreter.get_output_details())

        self.output_details = self.interpreter.get_output_details()
        self.state1 = np.zeros((1, 128), dtype=np.float32)
        self.state2 = np.zeros((1, 128), dtype=np.float32)

    def predict(self, window_frame, window_size, feature_size):  # 1,25,6
        if window_frame.shape[0] != 25:
            return 0
        window_frame = window_frame.reshape(1, window_size, feature_size)
        # self.interpreter.set_tensor(self.input_details[0]["index"], window_frame.astype(np.float32))
        self.interpreter.set_tensor(self.input_details[0]["index"], self.state1)
        self.interpreter.set_tensor(self.input_details[1]["index"], self.state2)
        self.interpreter.set_tensor(self.input_details[2]["index"], window_frame.astype(np.float32))

        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[1]["index"])
        # self.state1 = self.interpreter.get_tensor(self.output_details[0]["index"])
        # self.state2 = self.interpreter.get_tensor(self.output_details[2]["index"])

        return result[0]


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


def get_max_acc_position(motion_frame_window):
    accelerations = motion_frame_window[:, -3:]
    # 计算每个位置的加速度的范数
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

    # 找到加速度最大的位置
    max_acceleration_index = np.argmax(acceleration_magnitudes)
    return max_acceleration_index


def test_with_padding(motion_frame, feature_size, window_size, padding=10):
    motion_frame_count = motion_frame.shape[0] // window_size
    motion_frame = motion_frame[:motion_frame_count * window_size]
    hits = []
    hit = 0
    kill = 0
    kill_ress = []
    high_long = 0
    ping_chou = 0
    cur_hit_recognizer = HitRecognizer(model_file_path='./ios/right_model/right_lstm_hit.tflite')
    prev_hit_recognizer = HitRecognizer(model_file_path='./ios/right_model/right_lstm_hit.tflite')
    next_hit_recognizer = HitRecognizer(model_file_path='./ios/right_model/right_lstm_hit.tflite')

    cur_kill_recognizer = KillRecognizer1(model_file_path='./ios/right_model/right_lstm_kill.tflite')

    # a = cur_kill_recognizer.predict(motion_frame[6865:6890, :], window_size=window_size, feature_size=feature_size)
    # print(get_max_acc_position(motion_frame[6865:6890, :]))
    # b = cur_kill_recognizer.predict(motion_frame[6870:6895, :], window_size=window_size, feature_size=feature_size)
    # print(get_max_acc_position(motion_frame[6870:6895, :]))
    #
    # c = cur_kill_recognizer.predict(motion_frame[6875:6900, :], window_size=window_size, feature_size=feature_size)
    # print(get_max_acc_position(motion_frame[6875:6900, :]))
    #
    # print("hahaha")
    # print(a)
    # print(b)
    # print(c)
    # print(cur_kill_recognizer.predict(motion_frame[6860:6885, :], window_size=window_size, feature_size=feature_size))
    # print(cur_kill_recognizer.predict(motion_frame[6855:6880, :], window_size=window_size, feature_size=feature_size))

    threshold = 0.78
    last_flag = -1
    last_index = 0
    for i in range(0, len(motion_frame) - padding - padding - window_size, window_size):
        prev_window_frame = motion_frame[i:i + window_size, :]
        cur_window_frame = motion_frame[i + padding:i + window_size + padding, :]
        next_window_frame = motion_frame[i + padding + padding:i + window_size + padding + padding, :]
        # print(cur_window_frame.shape)
        cur_res = cur_hit_recognizer.predict(cur_window_frame, window_size=window_size, feature_size=feature_size)
        prev_res = prev_hit_recognizer.predict(prev_window_frame, window_size=window_size, feature_size=feature_size)
        next_res = next_hit_recognizer.predict(next_window_frame, window_size=window_size, feature_size=feature_size)

        best_ranges_string = ["[" + str(i - padding) + ":" + str(i + window_size - padding) + "]   " + str(prev_res),
                              "[" + str(i) + ":" + str(i + window_size) + "]   " + str(cur_res),
                              "[" + str(i + padding) + ":" + str(i + window_size + padding) + "]   " + str(next_res)]
        best_ranges = [[i - padding, i + window_size - padding], [i, i + window_size],
                       [i + padding, i + window_size + padding]]
        print(best_ranges_string[0])
        print(best_ranges_string[1])
        print(best_ranges_string[2])
        print("*****************")
        flag = -1
        new_index = 0
        max_res = 0
        if prev_res > cur_res and prev_res > next_res:
            flag = 0
            new_index = i
            max_res = prev_res
        elif cur_res > prev_res and cur_res > next_res:
            flag = 1
            new_index = i + padding
            max_res = cur_res
        elif next_res > prev_res and next_res > cur_res:
            flag = 2
            new_index = i + padding + padding
            max_res = next_res
        if max_res > threshold:
            if last_flag == 2 and flag == 0 or last_index + 25 > new_index:
                continue
            else:
                best_range = best_ranges[flag]
                best_center = get_max_acc_position(motion_frame[best_range[0]:best_range[1], :])
                print(best_center)
                best_frame = motion_frame[best_range[0] - (12 - best_center):best_range[1] - (12 - best_center), :]
                best_kill_res = cur_kill_recognizer.predict(best_frame, window_size=window_size,
                                                            feature_size=feature_size)
                print(best_kill_res)

                last_index = new_index
                hit = hit + 1
                last_flag = flag
                if best_kill_res[2] > best_kill_res[0] and best_kill_res[2] > best_kill_res[1]:
                    kill = kill + 1
                elif best_kill_res[1] > best_kill_res[0] and best_kill_res[1] > best_kill_res[2]:
                    high_long = high_long + 1
                elif best_kill_res[0] > best_kill_res[1] and best_kill_res[0] > best_kill_res[2]:
                    ping_chou = ping_chou + 1

    print("hit res is " + str(hit))
    print("kill res is " + str(kill))
    print("highlong res is " + str(high_long))
    print("ping_chou res is " + str(ping_chou))

    for r in kill_ress:
        print(r)


test_with_padding(motion_frame=motion_frame, feature_size=6, window_size=25)

# 0.8 836
# 0.8 27
