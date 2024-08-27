import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import myutil


# 返回所有的floats
def read_floats_from_ad(file_path):
    # 读取二进制文件中的所有 float 数据
    data = np.fromfile(file_path, dtype=np.float32)
    return data


# 返回删掉时间信息的floats，就是只剩下传感器floats了
def read_motion_floats_from_ad(file_path):
    # 读取二进制文件中的所有 float 数据
    data = np.fromfile(file_path, dtype=np.float32)
    delete_indices = np.arange(0, len(data), 7)
    # 删除指定索引的元素
    new_data = np.delete(data, delete_indices)
    return new_data


# 这里是window frame return (none, 25, 6)
def reshape_motion_floats_to_window_frames(data, feature_size=6, window_size=25):
    # 计算总的帧数
    total_frames = data.shape[0] // feature_size

    # 计算完整组的数量
    num_groups = total_frames // window_size

    # 截断不完整的数据
    truncated_data = data[:num_groups * window_size * feature_size]

    # 重新形状为 (num_groups, group_size, frame_size)
    reshaped_data = truncated_data.reshape((num_groups, window_size, feature_size))
    return reshaped_data


# 将motion floats，注意只有6个传感器数字，没有时间信息。return (none, 6)
def reshape_motion_floats_to_motion_frames(data, feature_size=6):
    total_frames = data.shape[0] // feature_size
    # 重新形状为 (num_groups, group_size, frame_size)
    reshaped_data = data.reshape((total_frames, feature_size))
    return reshaped_data


# return (none, 7)
def reshape_floats_to_time_frames(data, feature_size=7):
    total_frames = data.shape[0] // feature_size
    reshaped_data = data.reshape((total_frames, feature_size))
    return reshaped_data


# return (none,)
def reshape_time_frames_to_floats(time_frames, feature_size=7):
    total_size = time_frames.shape[0] * feature_size
    reshaped_data = time_frames.reshape((total_size,))
    return reshaped_data


def save_time_frames(time_frames, file_name):
    floats = reshape_time_frames_to_floats(time_frames)
    floats.astype(np.float32).tofile(file_name)


def save_window_frames_to_ad(window_frames, ad_file_name):
    bba = []
    data = window_frames.flatten()
    for i in range(len(data)):
        if i % 6 == 0:
            bba.append(0)
        bba.append(data[i])
    print(bba)
    nn = np.array(bba)
    print(nn.shape)
    nn.astype(np.float32).tofile(ad_file_name)

# huawei_issue_window_ad = [-1.0995574, 0.2406809, -2.4581218, -0.23754883, 0.27905273, -0.5048828, -2.149024, 0.47769663, -2.919936, -0.25634766, 0.2434082, -0.4091797, -2.502104, 0.3359759, -4.107458, -0.47583008, 0.28393555, -0.25732422, -2.4007003, 0.31398472, -4.9712214, -0.8371582, 0.66308594, -0.1430664, -2.8576276, 0.41538838, -5.2681017, -1.1518555, 0.5649414, -0.0012207031, -3.5662313, 0.6169739, -5.1202726, -1.5864258, 0.8190918, 0.1850586, -4.408004, 0.77213365, -5.0029864, -1.706543, 0.49169922, 0.29785156, -5.385388, 0.9896017, -4.5387287, -1.9660645, 0.73779297, 0.37451172, -4.918687, 1.1520919, -3.794695, -1.7543945, 0.578125, 0.38427734, -3.7641516, 0.91507614, -3.0860913, -1.5900879, 0.58740234, 0.33642578, -2.5082128, 0.8894198, -2.4459045, -1.3989258, 0.54003906, 0.31054688, -2.2308798, 0.8356637, -1.6749926, -1.1713867, 0.39208984, 0.26293945, -2.9150488, 0.61575216, -0.87964594, -1.0043945, 0.359375, 0.17407227, -3.3499851, 0.41538838, -0.25900686, -0.83813477, 0.39135742, 0.1394043, -2.918714, 0.19792034, 0.061086524, -0.82128906, 0.46362305, 0.11279297, -1.709201, 0.036651913, 0.087964594, -0.8466797, 0.52856445, 0.09008789, -0.50823987, -0.33108896, 0.13072516, -0.8088379, 0.5046387, 0.022705078, 0.61208695, -0.7867944, 0.3323107, -0.89990234, 0.501709, -0.017333984, 1.6872098, -1.047023, 0.5852089, -1.0141602, 0.44995117, -0.013427734, 2.179567, -1.097114, 1.0616838, -1.0715332, 0.33984375, -2.4414062E-4, 2.1087067, -1.1496484, 1.5344934, -1.1552734, 0.30810547, -0.067871094, 2.1099286, -1.0897835, 1.8912388, -1.1696777, 0.2277832, -0.0012207031, 1.6639969, -0.9492846, 1.9816469, -1.1520996, 0.20605469, -0.026855469, 1.4966198, -0.906524, 1.7776178, -1.0610352, 0.20141602, -0.056640625, 1.5063937, -0.7635816, 1.359786, -0.9863281, 0.19018555, -0.020263672]
# bba = []
# for i in range(len(huawei_issue_window_ad)):
#     if i%6 == 0:
#         bba.append(0)
#     bba.append(huawei_issue_window_ad[i])
# print(bba)
# nn = np.array(bba)
# print(nn.shape)
# nn.astype(np.float32).tofile('./huawei_issue.ad')

# aaaaaq = "./error910A7242__1713273736716__CN__Watch7,3__right__right__2024_04_16_21_22_16.ad"
# # 高远球0CABAC36__1720177032222__CN__S7_41mm__right__right__2024_07_05_18_57_12.ad
# file_path = '../aatest/F954FC8A__1718433170022__CN__S8_45mm__right__right__2024_06_15_14_32_50.ad'
# './C13D0A02__1718283752630__CN__Watch7,2__right__right__2024_06_13_21_02_32.ad'
#
# # 读取二进制文件中的 float 数据
# float_data = read_floats_from_ad('../aatest/F954FC8A__1718433170022__CN__S8_45mm__right__right__2024_06_15_14_32_50.ad')
# print(float_data.shape)
# aa = reshape_floats_to_time_frames(float_data)
# save_time_frames(aa, './test_save_time_frames.ad')

#
# motion_frame = reshape_data_to_motion_frame(float_data, feature_size=6)
# print(motion_frame.shape)
#
# # 将数据重新形状为 (x, 25, 6)
# reshaped_data = reshape_data_to_frames(float_data, feature_size=6, window_size=25)
# print(reshaped_data.shape)
