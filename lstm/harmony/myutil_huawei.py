# coding: utf-8

import os
import numpy as np
import pandas as pd
import csv

window = 130
features = 6
arguement_data = True  # 数据增强，


def load_data_from_csv(file_path):
    lines = open(file_path).readlines()
    # print(lines)
    lines = lines[1:]
    dd = []
    for l in lines:
        sn = l.split(',')
        # print(sn)
        float_data = [float(x.strip('"\n')) for x in sn][1:]
        if len(float_data) == features:
            dd.append(np.array(float_data))
        else:
            dd.append(np.array(float_data[:features]))

    # print(len(dd))
    if len(dd) == window:
        return np.array(dd)


def load_data_from_directory1(directory):
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
                if len(float_data) == features:
                    dd.append(np.array(float_data))
                else:
                    dd.append(np.array(float_data[:features]))

            # print(len(dd))
            if len(dd) == window:
                abc = np.array(dd)
                data.append(abc[::2, :]) #切片 130-> 65

    return np.stack(data)

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
            arg_dd = []  # 增强的dd

            for l in lines:
                sn = l.split(',')
                # print(sn)
                float_data = [float(x.strip('"\n')) for x in sn][1:]  #去掉时间
                if len(float_data) == features:
                    dd.append(np.array(float_data))
                    if arguement_data:
                        arg = [-float_data[0], -float_data[1], float_data[2], -float_data[3], -float_data[4],
                               float_data[5]]
                        arg_dd.append(np.array(arg))
                else:
                    dd.append(np.array(float_data[:features]))
                    if arguement_data:
                        arg = [-float_data[0], -float_data[1], float_data[2], -float_data[3], -float_data[4],
                               float_data[5]]
                        arg_dd.append(np.array(arg))

            # print(len(dd))
            if len(dd) == window:
                abc = np.array(dd)
                data.append(abc[::2, :])  # 切片 130-> 65
            if len(arg_dd) == window:
                arg_abc = np.array(arg_dd)
                data.append(arg_abc[::2, :])  # 切片 130-> 65

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
    print("Data shape:", data.shape)  # 应为 (2000, 130, 9)
    print("Labels shape:", labels.shape)  # 应为 (2000,)
    return data, labels


def build_badminton_fbou_data(fbou_data_path):
    # 加载第一个目录的数据
    category1_data = load_data_from_directory(fbou_data_path + os.sep + 'forehand_overhand')

    # 加载第二个目录的数据
    category2_data = load_data_from_directory(fbou_data_path + os.sep + 'forehand_underhand')
    category3_data = load_data_from_directory(fbou_data_path + os.sep + 'backhand_overhand')
    category4_data = load_data_from_directory(fbou_data_path + os.sep + 'backhand_underhand')

    print(category1_data.shape)
    print(category2_data.shape)
    print(category3_data.shape)
    print(category4_data.shape)

    # 将数据转换为NumPy数组，并整合到一个大的数据数组中
    data = np.concatenate((category1_data, category2_data, category3_data, category4_data), axis=0)

    # 创建对应的标签数组，其中category1对应标签0，category2对应标签1
    # 创建一个形状为 (2173, 2) 的数组
    aa1 = [0 for i in range(len(category1_data))]
    aa2 = [1 for i in range(len(category2_data))]
    aa3 = [2 for i in range(len(category3_data))]
    aa4 = [3 for i in range(len(category4_data))]

    a1 = np.array(aa1 + aa2 + aa3 + aa4)

    labels = a1

    print(labels)
    # 打印数据和标签的形状，确保正确加载和整理
    print("Data shape:", data.shape)  # 应为 (2000, 130, 9)
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
    print("Data shape:", data.shape)  # 应为 (2000, 130, 9)
    print("Labels shape:", labels.shape)  # 应为 (2000,)
    return data, labels


# build_badminton_kill_data()

print("my utils huawei loading")
