import struct
import numpy as np

# 读取文本文件并解析为 float32 数组
def read_float32_from_txt(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # 按逗号分隔并转换为 float32
        float_array = [float(x) for x in content.split(',')]
    return float_array

# 将 float32 数组以二进制形式写入文件
def write_float32_to_binary(float_array, output_file_path, endian='<'):
    with open(output_file_path, 'wb') as file:
        for value in float_array:
            # 将 float32 转换为二进制数据
            # '<f' 表示小端，'>f' 表示大端
            binary_data = struct.pack(f'{endian}f', value)
            file.write(binary_data)

# 示例文件路径
input_file_path = 'had1.txt'  # 输入的文本文件
output_file_path = '正手上手单框架.had'  # 输出的二进制文件

# 读取文本文件
float_array = read_float32_from_txt(input_file_path)
print("读取的 float32 数组:", len(float_array))

nn = np.array(float_array)[0:282820]
print(nn.shape)
nn.astype(np.float32).tofile(output_file_path)

# 写入二进制文件（小端）
# write_float32_to_binary(float_array, output_file_path, endian='<')
# print(f"二进制文件已写入: {output_file_path}")