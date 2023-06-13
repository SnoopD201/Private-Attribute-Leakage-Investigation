# predict data from label

from __future__ import print_function

import os
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import scipy.io as scio
import tensorflow as tf
import keras
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, \
    TimeDistributed
from keras.models import Model, Sequential
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import confusion_matrix, mean_squared_error
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from wifilib import *


# Parameters
use_existing_model = False
fraction_for_test = 0.2
data_dir = '/achive/220301/shiyd/Wiar-validuser/10'
ALL_MOTION = [1, 2]
# ALL_MOTION = [1,2,3,4]
# N_MOTION = len(ALL_MOTION)
N_MOTION = len(ALL_MOTION)  # 2
# 10 531    9 531
T_MAX = 3645
n_epochs = 30
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_batch_size = 32  # 32
f_learning_rate = 0.001


# data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]
def normalize_data(data_1):
    # print("??")
    data_1_max = np.concatenate(
        (data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate(
        (data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        # print(">0")
        return data_1
    # print(data_1_max.shape,data_1_min.shape)
    data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
    data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)

    # print("data_1_norm")
    return data_1_norm

# data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
# 零填充


def zero_padding(data, T_MAX):
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(
            data[i], ((0, 0), (0, 0), (T_MAX - t, 0)), 'constant', constant_values=0).tolist())
    return np.array(data_pad)


count1 = 0
data = []
label_test = []
for data_root, data_dirs, data_files in os.walk(data_dir):  # 遍历所有文件
    for data_file_name in data_files:
        count1 += 1
        file_path = os.path.join(data_root, data_file_name)
        try:
            weight = int(data_file_name.split('-')[2])
            # 拆分文件名
            bf = read_bf_file(file_path)
            csi_list = list(map(get_scale_csi, bf))
            csi_np = (np.array(csi_list))
            csi_amp = np.abs(csi_np)
            isnan = np.isnan(csi_amp)
            if(True in isnan):
                print(data_file_name)

                continue
            print(weight, count1)
            # if(height not in motion_sel):
            #     print(height)
            #     continue

            # print(gender,count1)
            # if(count1>=500):
            #     break
            col = csi_amp.shape[0]
            data_1 = csi_amp.reshape(col, 3, 30)
            data_1 = data_1.transpose(2, 1, 0)
            # print("data1shape:",data_1.shape)
            # Normalization
            data_normed_1 = normalize_data(data_1)

        except Exception:
            continue

        data.append(data_normed_1.tolist())  # 转换成列表
        # print(np.array(data).shape)
        # print(height,'1')
        label_test.append(weight)  # 追加label 实际上所有label都一样
        # print(label)
# print(np.array(data).shape)
data = zero_padding(data, T_MAX)
# data = data.transpose(3,0,1,2)

data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
# print(data.shape)
data = np.expand_dims(data, axis=-1)

label_test = np.array(label_test)
# print(label)
# print(data)
print(data.shape)
print(len(label_test))
label_new = list(set(label_test))  # 列出该测试集中的身高的个数(因为不一定是所有的)
# label_new=list(set(label_test))#列出该测试集中的身高的个数(因为不一定是所有的)
print('Testing...')
big = 88-54
small = 54

new_model = load_model('model_widar3_trained.h5')
label_test_pred = new_model.predict(data)

# heightlist=np.array(label_new)#转化成array
# print("heightlist=",heightlist)
# predtotal=(heightlist-small)/(big-small)#heightlist标准化
# usernum=len(predtotal)#其中身高的个数
# heightdiff=np.zeros(usernum)#初始化身高差值数组和用户个数统计数组
# eachusernum=np.zeros(usernum)
actsum = 0
predsum = 0
testlen = len(data)
print(testlen)
for n in range(0, testlen):  # 遍历所有测试集的样本
    pred = label_test_pred[n]*big+small
    act = label_test[n]  # 还原预测和实际身高值
    predsum += pred
    print(pred, act)

predsum = predsum/testlen

print("predict:", predsum)
print("actual:", 72)
