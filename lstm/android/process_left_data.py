# coding: utf-8
import shutil
import os
import numpy as np
import pandas as pd
import csv

directory = '/Users/sinnus/Desktop/server_activity_data'
data = []
# 遍历目录中的每个CSV文件
for filename in os.listdir(directory):
    if filename.endswith(".ad") and filename.__contains__('left'):
        filepath = os.path.join(directory, filename)
        data.append(filepath)
print(data)
print(len(data))
for d in data:
    shutil.copy2(d, '/Users/sinnus/Desktop/server_temp')