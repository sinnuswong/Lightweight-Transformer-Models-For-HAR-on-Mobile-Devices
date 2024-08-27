import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
from ad_reader import read_floats_from_ad, read_motion_floats_from_ad, reshape_time_frames_to_floats, \
    reshape_floats_to_time_frames, reshape_motion_floats_to_motion_frames, reshape_motion_floats_to_window_frames, \
    save_time_frames,save_window_frames_to_ad
import os

# 假设 timesteps = 25，features = 6
timesteps = 25
features = 6

X_train = [[[1] * 6] * 25] * 100
Y_train = [[[2] * 6] * 25] * 100
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape)

print(Y_train.shape)


# (none, 25, 6), (none, 25, 6),
def train(from_window_activity_data, to_window_activity_data, model_name):
    # 构建一个LSTM模型
    model = Sequential()
    model.add(LSTM(256, input_shape=(25, 6), return_sequences=True))
    # model.add(LSTM(64, return_sequences=True))  # 保持时间序列的输出
    model.add(Dense(features))  # 输出的特征数量与输入的特征数量相同

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 假设我们已经有了训练数据 X_train 和 Y_train
    # X_train: 形状为 (样本数, 25, 6) 的华为手表的传感器数据
    # Y_train: 形状为 (样本数, 25, 6) 的对应苹果手表的传感器数据
    model.fit(from_window_activity_data, to_window_activity_data, epochs=200, batch_size=32, validation_split=0.2)

    # 保存模型
    model.save(model_name)


# train(X_train, Y_train, 'sensor_conversion_model.h5')
#
# # 加载训练好的模型
# model = tf.keras.models.load_model('sensor_conversion_model.h5')
#
# # 将华为手表的新数据转换为苹果手表的数据
# # 假设 new_huawei_data 的形状是 (样本数, 25, 6)
# new_huawei_data = [[[1] * 6] * 25]  # 新的华为手表数据
# converted_apple_data = model.predict(new_huawei_data)
# print(converted_apple_data)
# # 转换后的苹果手表数据的形状也应该是 (样本数, 25, 6)


aaaaaq = "./error910A7242__1713273736716__CN__Watch7,3__right__right__2024_04_16_21_22_16.ad"

# # 高远球0CABAC36__1720177032222__CN__S7_41mm__right__right__2024_07_05_18_57_12.ad
from_file_path = '../aatest/F954FC8A__1718433170022__CN__S8_45mm__right__right__2024_06_15_14_32_50.ad'
to_file_path = './C13D0A02__1718283752630__CN__Watch7,2__right__right__2024_06_13_21_02_32.ad'


#
# # 读取二进制文件中的 float 数据
# float_data = read_floats_from_binary(file_path)
#
# motion_frame = reshape_data_to_motion_frame(float_data, feature_size=6)
# print(motion_frame.shape)
#
# # 将数据重新形状为 (x, 25, 6)
# reshaped_data = reshape_data_to_frames(float_data, feature_size=6, window_size=25)
# print(reshaped_data.shape)

def get_timestamp_from_ad_filename(ad_file_name):
    strs = ad_file_name.split("__")
    print(strs)
    return int(strs[1])


def read_ios_android_ad_fix(from_ad_file_path, to_ad_file_path):
    from_file_name = os.path.basename(from_ad_file_path)
    to_file_name = os.path.basename(to_ad_file_path)
    print("file names " + from_file_name + ",,,, " + to_file_name)
    from_start_time = get_timestamp_from_ad_filename(from_file_name)
    to_start_time = get_timestamp_from_ad_filename(to_file_name)
    print("file start times " + str(from_start_time) + " ,,," + str(to_start_time))

    from_time_frames = reshape_floats_to_time_frames(read_floats_from_ad(from_ad_file_path))  # none,7
    to_time_frames = reshape_floats_to_time_frames(read_floats_from_ad(to_ad_file_path))  # none,7
    # 用时间信息进行偏移比较，把两段ad文件取公共部分之后，保存，确保两个文件的长度大小一致。
    if from_start_time < to_start_time:
        dt = to_start_time - from_start_time
        dn = int(dt / 50)-15
        print("dn " + str(dn))
        # 删掉from time frame dn 个
        from_time_frames = from_time_frames[dn:]
        from_l = from_time_frames.shape[0]
        to_l = to_time_frames.shape[0]
        print("file l " + str(from_l) + " ,,," + str(to_l))

        l = min(from_l, to_l)
        from_time_frames = from_time_frames[: l]
        to_time_frames = to_time_frames[: l]
    elif from_start_time >= to_start_time:
        dt = from_start_time - to_start_time
        dn = int(dt / 50)-23
        print("dn " + str(dn))
        # to time frame dn 个
        to_time_frames = to_time_frames[dn:]

        from_l = from_time_frames.shape[0]
        to_l = to_time_frames.shape[0]
        print("file l " + str(from_l) + " ,,," + str(to_l))

        l = min(from_l, to_l)
        from_time_frames = from_time_frames[: l]
        to_time_frames = to_time_frames[: l]
    print(from_time_frames.shape[0], to_time_frames.shape[0])
    save_time_frames(from_time_frames, os.path.dirname(from_ad_file_path) + os.path.sep + 'from' + from_file_name)
    save_time_frames(to_time_frames, os.path.dirname(to_ad_file_path) + os.path.sep + 'to' + to_file_name)

test_from = '/Users/sinnus/Downloads/haoqiu/fromc7230e5b__1724742395337__CN1__HUAWEIWATCHGT4C41__right__right__2024_08_27_15_06_35.ad'
test_to = '/Users/sinnus/Downloads/haoqiu/toA436FCC9__1724742398182__CN__S8_45mm__right__right__2024_08_27_15_06_38.ad'

# read_ios_android_ad_fix(test_from, test_to)

# from_window_frames = reshape_motion_floats_to_window_frames(read_motion_floats_from_ad(test_from))
# to_window_frames = reshape_motion_floats_to_window_frames(read_motion_floats_from_ad(test_to))
# train(from_window_frames, to_window_frames, "test_covert_ios_android_model.h5")

model = tf.keras.models.load_model('test_covert_ios_android_model.h5')

# 将华为手表的新数据转换为苹果手表的数据
# 假设 new_huawei_data 的形状是 (样本数, 25, 6)
huawei_data = reshape_motion_floats_to_window_frames(read_motion_floats_from_ad('huawei_issue.ad'))
converted_huawei_data = model.predict(huawei_data)
print(converted_huawei_data.shape)
save_window_frames_to_ad(converted_huawei_data, 'huawei_issue_converted.ad')