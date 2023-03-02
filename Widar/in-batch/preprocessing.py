import numpy as np
import os
from wifilib import *
import random
import json
path_to_data = "/export/shiyd/CSI-gesture/pushpull"
train_file = 'train.txt'
test_file = 'test.txt'
valid_file_list = []
sample_count = -1
fraction_for_test = 0.2
# False: 随机选择一个人为测试集
# True: 随机选择fraction_for_test个样本为测试集
random_split = True
# 从人的编号到valid_file_list中对应的序号的映射
# eg. 'user9' -> [1, 3, 5, ..]
person_to_valid_file = dict()
# 遍历所有文件，暂不支持二级文件夹
for data_root, data_dirs, data_files in os.walk(path_to_data):
    data_files.sort()  # 方便寻找同一个数据的其他天线
    sample_name = ''
    temp = []
    temp.clear()
    valid = True  # 该sample是否合法
    for data_file_name in data_files:
        file_path = os.path.join(data_root, data_file_name)
        try:
            # 拆分文件名
            name = data_file_name.rsplit('-', 1)[0]
            if sample_name != name:
                # 下一个sample
                if valid and sample_name != '':
                    valid_file_list.append(temp[:])
                    sample_count += 1
                    user_name = sample_name.split('-')[3]
                    if user_name not in person_to_valid_file.keys():
                        person_to_valid_file[user_name] = []
                    person_to_valid_file[user_name].append(
                        len(valid_file_list)-1)
                temp.clear()
                sample_name = name
                valid = True

            temp.append(file_path)
            bf = read_bf_file(file_path)
            csi_list = list(map(get_scale_csi, bf))
            csi_np = (np.array(csi_list))
            csi_amp = np.abs(csi_np)
            isnan = np.isnan(csi_amp)
            if(True in isnan):
                # print(file_path)
                valid = False

        except Exception as e:
            valid = False
            continue
        
np.save("valid_file_list.npy", valid_file_list)
with open('person_to_valid_file.json', 'w') as f:
    json.dump(person_to_valid_file, f)
    
valid_file_list = np.load("valid_file_list.npy").tolist()
with open('person_to_valid_file.json', 'r') as f:
    person_to_valid_file = json.load(f)

if random_split:
    test_set_list = random.sample(range(0, len(valid_file_list)), int(
        len(valid_file_list) * fraction_for_test))
    test_set_list.sort()
    train_set = open(train_file, 'w')
    test_set = open(test_file, 'w')

    for i in range(0, len(valid_file_list)):
        if i in test_set_list:
            for line in valid_file_list[i]:
                test_set.write(line + '\n')
        else:
            for line in valid_file_list[i]:
                train_set.write(line + '\n')
else:
    keys = list(person_to_valid_file.keys())
    temp = random.sample(range(0, len(keys)), 1)
    test_set_list = []
    for i in temp:
        test_set_list.extend(person_to_valid_file[keys[i]])
    
    train_set = open(train_file, 'w')
    test_set = open(test_file, 'w')

    for i in range(0, len(valid_file_list)):
        if i in test_set_list:
            for line in valid_file_list[i]:
                test_set.write(line + '\n')
        else:
            for line in valid_file_list[i]:
                train_set.write(line + '\n')
