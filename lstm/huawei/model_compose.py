# 把模型组合起来，
# 左手模型放一起，右手模型放一起
# 1. hit, 2. forehand backhand 3. fou, 4. bou, 5 杀球高远球模型。

import os
import myutil_huawei
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_directory = os.path.dirname(current_file_path)

print("Current file path:", current_file_path)
print("Current directory:", current_directory)
print(current_directory + os.sep + 'left_model')
