import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import myutil

data, labels = myutil.build_badminton_hit_data()


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


file_path = './test.ad'

# 读取二进制文件中的 float 数据
float_data = read_floats_from_binary(file_path)

motion_frame = reshape_data_to_motion_frame(float_data, feature_size=6)
print(motion_frame.shape)

# 将数据重新形状为 (x, 25, 6)
reshaped_data = reshape_data_to_frames(float_data, feature_size=6, window_size=25)
print(reshaped_data.shape)


def test_sequence(motion_frame):
    hits = []
    hit = 0
    # Run the model with TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path='./mobile_har_xs_bestval.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(reshaped_data)):
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
        self.output_details = self.interpreter.get_output_details()

    def predict(self, window_frame, window_size, feature_size):  # 1,25,6
        if window_frame.shape[0] != 25:
            return 0
        window_frame = window_frame.reshape(1, window_size, feature_size)
        self.interpreter.set_tensor(self.input_details[0]["index"], window_frame.astype(np.float32))
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]["index"])

        # print(result[0])
        # print(reshaped_data[i:i + 1, :, :].astype(np.float32))
        # self.interpreter.reset_all_variables()
        return result[0][1]


def test_with_padding(motion_frame, feature_size, window_size, padding=10):
    motion_frame_count = motion_frame.shape[0] // window_size
    motion_frame = motion_frame[:motion_frame_count * window_size]
    hits = []
    hit = 0
    cur_hit_recognizer = HitRecognizer(model_file_path='./mobile_har_xs_bestval.tflite')
    prev_hit_recognizer = HitRecognizer(model_file_path='./mobile_har_xs_bestval.tflite')
    next_hit_recognizer = HitRecognizer(model_file_path='./mobile_har_xs_bestval.tflite')
    threshold = 0.86
    last_flag = -1
    for i in range(10, len(motion_frame - padding - window_size), window_size):
        cur_window_frame = motion_frame[i:i + window_size, :]
        prev_window_frame = motion_frame[i - padding:i + window_size - padding, :]
        next_window_frame = motion_frame[i + padding:i + window_size + padding, :]
        # print(cur_window_frame.shape)
        cur_res = cur_hit_recognizer.predict(cur_window_frame, window_size=window_size, feature_size=feature_size)
        prev_res = prev_hit_recognizer.predict(prev_window_frame, window_size=window_size, feature_size=feature_size)
        next_res = next_hit_recognizer.predict(next_window_frame, window_size=window_size, feature_size=feature_size)

        best_ranges = ["[" + str(i - padding) + ":" + str(i + window_size - padding) + "]   " + str(prev_res),
                       "[" + str(i) + ":" + str(i + window_size) + "]   " + str(cur_res),
                       "[" + str(i + padding) + ":" + str(i + window_size + padding) + "]   " + str(next_res)]
        # print("[" + str(i - padding) + ":" + str(i + window_size - padding) + "]   " + str(prev_res))
        # print("[" + str(i) + ":" + str(i + window_size) + "]   " + str(cur_res))
        # print("[" + str(i + padding) + ":" + str(i + window_size + padding) + "]   " + str(next_res))
        print("*****************")
        flag = -1
        max_res = 0
        best_range = ""
        if prev_res > cur_res and prev_res > next_res:
            flag = 0
            max_res = prev_res
            best_range = best_ranges[0]
        elif cur_res > prev_res and cur_res > next_res:
            flag = 1
            max_res = cur_res
            best_range = best_ranges[1]
        elif next_res > prev_res and next_res > cur_res:
            flag = 2
            max_res = next_res
            best_range = best_ranges[2]
        if max_res > threshold:
            if last_flag == 2 and flag == 0:
                continue
            else:
                hit = hit + 1
                last_flag = flag
                print(best_range)
    print("hit res is " + str(hit))


def test_with(motion_frame, feature_size, window_size):
    motion_frame_count = motion_frame.shape[0] // window_size
    motion_frame = motion_frame[:motion_frame_count * window_size]
    hits = []
    hit = 0
    cur_hit_recognizer = HitRecognizer(model_file_path='./model_LSTM3.tflite')

    threshold = 0.9
    for i in range(0, len(motion_frame), window_size):
        cur_window_frame = motion_frame[i:i + window_size, :]

        cur_res = cur_hit_recognizer.predict(cur_window_frame, window_size=window_size, feature_size=feature_size)

        # print("[" + str(i - padding) + ":" + str(i + window_size - padding) + "]   " + str(prev_res))
        # print("[" + str(i) + ":" + str(i + window_size) + "]   " + str(cur_res))
        # print("[" + str(i + padding) + ":" + str(i + window_size + padding) + "]   " + str(next_res))
        print("*****************")
        flag = -1
        max_res = cur_res
        if max_res > threshold:
            hit = hit + 1

    print("hit res is " + str(hit))


test_with_padding(motion_frame=motion_frame, feature_size=6, window_size=25)

# test_sequence(motion_frame=motion_frame, )
# 0.8 836
# 0.8 27
