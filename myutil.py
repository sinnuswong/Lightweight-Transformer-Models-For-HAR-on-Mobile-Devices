# coding: utf-8

import os
import numpy as np
import pandas as pd
import csv


def load_data_from_csv(file_path):
    lines = open(file_path).readlines()
    # print(lines)
    lines = lines[1:]
    dd = []
    for l in lines:
        sn = l.split(',')
        # print(sn)
        float_data = [float(x.strip('"\n')) for x in sn][1:]
        if len(float_data) == 6:
            dd.append(np.array(float_data))

    # print(len(dd))
    if len(dd) == 25:
        return np.array(dd)


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
            # print(filepath)
            lines = open(filepath).readlines()
            # print(lines)
            lines = lines[1:]
            dd = []
            for l in lines:
                sn = l.split(',')
                # print(sn)
                float_data = [float(x.strip('"\n')) for x in sn][1:]
                if len(float_data) == 6:
                    dd.append(np.array(float_data))

            # print(len(dd))
            if len(dd) == 25:
                data.append(np.array(dd))

    return np.stack(data)


def build_badminton_hit_data(hit_data_path):
    # 加载第一个目录的数据
    category1_data = load_data_from_directory(hit_data_path + '/none')

    # 加载第二个目录的数据
    category2_data = load_data_from_directory(hit_data_path + '/yes')
    print(category1_data.shape)
    # 将数据转换为NumPy数组，并整合到一个大的数据数组中
    data = np.concatenate((category1_data, category2_data), axis=0)

    # 创建对应的标签数组，其中category1对应标签0，category2对应标签1
    # 创建一个形状为 (2173, 2) 的数组
    aa1 = [0 for i in range(len(category1_data))]
    aa2 = [1 for i in range(len(category2_data))]
    a1 = np.array(aa1 + aa2)

    labels = a1

    print(labels)
    # 打印数据和标签的形状，确保正确加载和整理
    print("Data shape:", data.shape)  # 应为 (2000, 25, 6)
    print("Labels shape:", labels.shape)  # 应为 (2000,)
    return data, labels


def build_badminton_kill_data(kill_data_path):
    # 加载第一个目录的数据
    category1_data = load_data_from_directory(kill_data_path + '/forehand_pingchou')

    # 加载第二个目录的数据
    category2_data = load_data_from_directory(kill_data_path + '/high_long_shot')
    category3_data = load_data_from_directory(kill_data_path + '/kill_shot')

    print(category1_data.shape)
    print(category2_data.shape)
    print(category3_data.shape)

    # 将数据转换为NumPy数组，并整合到一个大的数据数组中
    data = np.concatenate((category1_data, category2_data, category3_data), axis=0)

    # 创建对应的标签数组，其中category1对应标签0，category2对应标签1
    # 创建一个形状为 (2173, 2) 的数组
    aa1 = [0 for i in range(len(category1_data))]
    aa2 = [1 for i in range(len(category2_data))]
    aa3 = [2 for i in range(len(category3_data))]

    a1 = np.array(aa1 + aa2 + aa3)

    labels = a1

    print(labels)
    # 打印数据和标签的形状，确保正确加载和整理
    print("Data shape:", data.shape)  # 应为 (2000, 25, 6)
    print("Labels shape:", labels.shape)  # 应为 (2000,)
    return data, labels

# build_badminton_kill_data()