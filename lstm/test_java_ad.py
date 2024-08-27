import tensorflow as tf
import numpy as np
from ad_reader import read_motion_floats_from_ad, reshape_motion_floats_to_window_frames, reshape_motion_floats_to_motion_frames


file_path = './C13D0A02__1718283752630__CN__Watch7,2__right__right__2024_06_13_21_02_32.ad'

# 读取二进制文件中的 float 数据
float_data = read_motion_floats_from_ad(file_path)

motion_frame = reshape_motion_floats_to_motion_frames(float_data, feature_size=6)
print(motion_frame.shape)
print(motion_frame)
