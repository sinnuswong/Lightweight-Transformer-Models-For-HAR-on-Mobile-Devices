# coding: utf-8

import os
import numpy as np
import pandas as pd
import csv


def load_data_from_directory(directory):
    data = []
    # 遍历目录中的每个CSV文件
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            # # 读取CSV文件
            # df = pd.read_csv(filepath, encoding='utf-8')
            # # 丢弃第一列（时间列），保留后面的6列数据
            # df = df.iloc[:, 1:]
            # # 将数据添加到列表中
            # data.append(df.values)
            lines = open(filepath).readlines()
            # print(lines)
            lines = lines[1:]
            dd = []
            for l in lines:
                sn = l.split(',')
                float_data = [float(x) for x in sn][1:]
                if len(float_data) == 6:
                    dd.append(np.array(float_data))

            # print(len(dd))
            if len(dd) == 25:
                data.append(np.array(dd))

    return np.stack(data)


def build_badminton_data():
    # 加载第一个目录的数据
    category1_data = load_data_from_directory('./datasets/train_hit_c25/none')

    # 加载第二个目录的数据
    category2_data = load_data_from_directory('./datasets/train_hit_c25/yes')
    print(category1_data.shape)
    # 将数据转换为NumPy数组，并整合到一个大的数据数组中
    data = np.concatenate((category1_data, category2_data), axis=0)

    # 创建对应的标签数组，其中category1对应标签0，category2对应标签1
    # 创建一个形状为 (2173, 2) 的数组
    dd = [0 for i in range(len(category1_data))] + [1 for i in range(len(category2_data))]

    labels = np.array(dd)

    print(labels)
    # 打印数据和标签的形状，确保正确加载和整理
    print("Data shape:", data.shape)  # 应为 (2000, 25, 6)
    print("Labels shape:", labels.shape)  # 应为 (2000,)
    return data, labels

# build_badminton_data()
#
# dd = [[1, 0] for i in range(5)]
# dd = np.array(dd)
# print(dd.shape)
# array1 = dd
#
# # 创建后面的行的数组，第二列为 [0, 1]
# dd = [[0, 1] for i in range(5)]
# dd = np.array(dd)
# print(dd.shape)
# array2 = dd
# # 使用 np.vstack() 合并数组
# labels = np.vstack((array1, array2))
#
# print(labels)
