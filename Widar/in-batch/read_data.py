from __future__ import print_function

import os
import threading

import time
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from wifilib import *
from keras.utils import Sequence

ALL_MOTION = [178, 161, 170, 160, 180, 172, 168,
              175, 165, 170, 175, 176, 155, 158, 175, 186]
maxheight = 186
minheight = 155
tmax = 3879


def normalize_data(data_1):
    data_1_max = np.concatenate(
        (data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate(
        (data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    # print(data_1_max.shape,data_1_min.shape)
    data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
    data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)

    return data_1_norm

# data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
# 零填充


def zero_padding(data, tmax):
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        if tmax-t>=0:
            data_pad.append(np.pad(
                data[i], ((0, 0), (0, 0), (tmax - t, 0)), 'constant', constant_values=0).tolist())
        else:
            tmp=np.asarray(data[i])
            data_pad.append(np.array(tmp[:,:,0:tmax]))
    return np.array(data_pad)

# One-Hot 编码
# label(list)=>_label(ndarray): [N,]=>[N,num_class]


def onehot_encoding(label, num_class):
    label = np.array(label).astype('int32')
    label = np.squeeze(label)
    _label = np.eye(num_class)[label - 1]  # from label to onehot
    return _label


def load_data(file_list, motion_sel, label_name='height', cal_distance=True, min_value=0, max_value=0, T_MAX=0):
    data = []
    label = []
    if cal_distance:
        min_value = 100000
        max_value = -100000
        T_MAX = 0
    for file_path in file_list:
        if file_path == '':
            continue
        data_file_name = file_path.rsplit('/', 1)[-1]
        # TODO
        if label_name == 'height':
            label_tmp = int(data_file_name.split('-')[1])
        elif label_name == 'weight':
            label_tmp = int(data_file_name.split('-')[2])
        elif label_name == 'gender':
            label_tmp = int(data_file_name.split('-')[0])
        elif label_name == 'actitity':
            label_tmp = int(data_file_name.split('-')[4])
        # 拆分文件名
        bf = read_bf_file(file_path)
        csi_list = list(map(get_scale_csi, bf))
        csi_np = (np.array(csi_list))
        csi_amp = np.abs(csi_np)
        col = csi_amp.shape[0]
        data_1 = csi_amp.reshape(col, 3, 30)
        data_1 = data_1.transpose(2, 1, 0)
        # print("data1shape:",data_1.shape)
        # Normalization
        data_normed_1 = normalize_data(data_1)

        data.append(data_normed_1.tolist())  # 转换成列表
        # print(np.array(data).shape)
        # print(height,'1')
        label.append(label_tmp)  # 追加label

        if cal_distance:
            min_value = min(min_value, label_tmp)
            max_value = max(max_value, label_tmp)
            # Update T_MAX
            if T_MAX < np.array(data_1).shape[2]:
                T_MAX = np.array(data_1).shape[2]
            # print(T_MAX)
        # print(label)

        # data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
        # {'__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Sun Nov 11 21:52:08 2018',
        # '__version__': '1.0', '__globals__': [], 'velocity_spectrum_ro'[data....]
        # user1-1-1-1-1-1-1e-07-100-20-100000-L0.mat

        # label_1 = int(data_file_name.split('-')[4])#1 gesture type
        # location = int(data_file_name.split('-')[5])#1 torso location
        # orientation = int(data_file_name.split('-')[6])#1 face orientation
        # repetition = int(data_file_name.split('-')[7])#repetition number
        # receiverid = data_file_name.split('-')[8]
        # print(height,label_1,location,orientation,repetition,receiverid)
        # Each file is a 20*20*T matrix, where the first dimension represents the
        # velocity along x axis ranging between [-2,+2] m/s, the second dimension represents
        # the velocity along y axis ranging between [-2,+2] m/s and the third dimension
        # represents the timestamps with 10Hz sampling rate.

        # Select Motion
        # if (label_1 not in motion_sel):
        #     continue
        # #Select BMI
        # if(BMIindex not in motion_sel):
        #     print(BMIindex)
        #     continue
        # Select Location
        # if (location not in [1,2,3,5]):
        #     continue

        # if(label_tmp not in motion_sel):
        #     print(label_tmp)
        #     continue
        # label,output
        # if(receiverid!="r4.dat"):
        #     print(height,"no")
        #     continue

        # Select Orientation
        # if (orientation not in [1,2,4,5]):
        #     continue

    # print(np.array(data).shape)
    data = zero_padding(data, T_MAX)

    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
    # print(data.shape)
    data = np.expand_dims(data, axis=-1)

    label = np.array(label)
    # print(data)
    # print(label.shape)
    # print("data shape")
    # print(data.shape)

    if max_value-min_value > 0:
        label_norm = (label - min_value) / (max_value - min_value)
    else:
        label_norm = label
    # print("TMAX: "+str(T_MAX))

    return data, label_norm, min_value, max_value, T_MAX


# all_files: 文件需要读取的文件名列表。
# batch_size：普通的batch_size
class DataGenerator(Sequence):
    def __init__(self, path, batch_size):
        self.batch_size = batch_size
        self.all_files = []
        for data_root, data_dirs, data_files in os.walk(path):  # 遍历所有文件
            for data_file_name in data_files:
                self.all_files.append(os.path.join(data_root, data_file_name))
        print(len(self.all_files))
        self.all_files = self.all_files

    def __len__(self):
        return int(np.ceil(len(self.all_files) / self.batch_size))

    def __getitem__(self, idx):
        print('  wf>>>>>>>>>>>>>>>>>>>>>generator yielded a batch %d  PID:%d  ident:%d' % (
            idx, os.getpid(), threading.currentThread().ident))
        data = []
        label = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            if(i >= len(self.all_files)):
                break
            file_path = self.all_files[i]
            try:
                data_file_name = file_path.split('/')[-1]
                height = int(data_file_name.split('-')[1])
                # 拆分文件名
                bf = read_bf_file(file_path)
                csi_list = list(map(get_scale_csi, bf))
                csi_np = (np.array(csi_list))
                csi_amp = np.abs(csi_np)
                receiverid = data_file_name.split('-')[8]

                if(height not in ALL_MOTION):
                    print(height)
                    continue
                # print(receiverid,height,count)
                col = csi_amp.shape[0]
                data_1 = csi_amp.reshape(col, 3, 30)
                data_1 = data_1.transpose(2, 1, 0)
                data_normed_1 = normalize_data(data_1)
            except Exception:
                continue
            data.append(data_normed_1.tolist())
            label.append((height-minheight)/(maxheight-minheight))

        data = zero_padding(data, tmax)
        data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
        data = np.expand_dims(data, axis=-1)
        label = np.array(label)
        # print(data.shape)
        # print(label.shape)
        return data, label
