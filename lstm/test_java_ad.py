import tensorflow as tf
import numpy as np


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


file_path = './F954FC8A__1718433170022__CN__S8_45mm__right__right__2024_06_15_14_32_50.ad'

# 读取二进制文件中的 float 数据
float_data = read_floats_from_binary(file_path)

motion_frame = reshape_data_to_motion_frame(float_data, feature_size=6)
print(motion_frame.shape)
print(motion_frame)
