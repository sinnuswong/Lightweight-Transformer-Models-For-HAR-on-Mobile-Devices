# 把模型组合起来，
# 左手模型放一起，右手模型放一起
# 1. hit, 2. fbou, 3. 杀球高远球模型。

import os
import myutil_huawei

import os


def merge_tflite_files_to_binary(output_file, input_files, config_file):
    """
    将多个文件的二进制内容写入到一个新文件中，并记录每个文件的起始和结束位置到配置文件中。

    :param output_file: 要生成的二进制文件的路径
    :param input_files: 输入文件列表
    :param config_file: 记录文件偏移信息的配置文件路径
    """
    file_info = []  # 存储每个文件的开始和结束位置

    with open(output_file, 'wb') as outfile:
        current_position = 0
        for input_file in input_files:
            start_position = current_position  # 记录起始位置
            len_content = 0
            # 读取文件的二进制内容并写入到输出文件中
            with open(input_file, 'rb') as infile:
                content = infile.read()
                outfile.write(content)
                len_content = len(content)
                current_position += len_content  # 更新当前写入位置

            file_info.append((input_file, start_position, len_content))

    # 将每个文件的开始和结束位置写入到配置文件中
    with open(config_file, 'w') as config:
        config_ints = []
        for input_file, start, l in file_info:
            config_ints.append(start)
            config_ints.append(l)

        ss = ''
        for i in config_ints:
            if ss == '':
                ss = ss + str(i)
            else:
                ss = ss + ', ' + str(i)

        config.write(ss)


right_input_files = ['right_model/right_lstm_hit.tflite', 'right_model/right_lstm_fbou.tflite',
                     'right_model/right_lstm_kill.tflite']  # 要合并的文件
merge_tflite_files_to_binary('right_model/right.model', right_input_files, 'right_model/right_model_file_config.txt')

left_input_files = ['left_model/left_lstm_hit.tflite', 'left_model/left_lstm_fbou.tflite',
                     'left_model/left_lstm_kill.tflite']  # 要合并的文件
merge_tflite_files_to_binary('left_model/left.model', left_input_files, 'left_model/left_model_file_config.txt')
