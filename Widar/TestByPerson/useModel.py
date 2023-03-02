from __future__ import print_function

import os, sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import scipy.io as scio
import tensorflow as tf
import keras
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, \
    TimeDistributed
from keras.models import Model, Sequential
import keras.backend as K
from sklearn.metrics import confusion_matrix,mean_squared_error
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from wifilib import *
import h5py
use_existing_model = False
fraction_for_test = 0.2
data_dir = '/export/shiyd/CSI-/r4-test'
# data_dir = '/export/shiyd/CSI-resize/r4'
ALL_MOTION =[178,161,170,160,180,172,168,175,165,170,175,176,155,158,175,186]
# ALL_MOTION = [1,2,3,4]
# N_MOTION = len(ALL_MOTION)
N_MOTION=1
T_MAX = 2743
n_epochs = 30
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_batch_size = 32 # 32
f_learning_rate = 0.001

# 活动识别
# 隐私预测(性别,身高,体重-baseline)

#标准化
def normalize_data(data_1):
    # print("??")
    data_1_max = np.concatenate((data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
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
#零填充
def zero_padding(data, T_MAX):
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0, 0), (0, 0), (T_MAX - t, 0)), 'constant', constant_values=0).tolist())
    return np.array(data_pad)

def load_data(path_to_data, motion_sel):
    global T_MAX
    # global count1
    data = []
    label = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):#遍历所有文件
        for data_file_name in data_files:
            # count1=count1+1
            file_path = os.path.join(data_root, data_file_name)
            try:
                height=int(data_file_name.split('-')[1])
                username=data_file_name.split('-')[3]
                #拆分文件名
                if(height not in motion_sel):
                    print(height)
                    continue
                if(username!="user6"):
                    continue
                bf = read_bf_file(file_path)
                csi_list = list(map(get_scale_csi, bf))
                csi_np = (np.array(csi_list))
                csi_amp = np.abs(csi_np)
                
                # data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
                #{'__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Sun Nov 11 21:52:08 2018',
                # '__version__': '1.0', '__globals__': [], 'velocity_spectrum_ro'[data....]
                #user1-1-1-1-1-1-1e-07-100-20-100000-L0.mat
                
                # label_1 = int(data_file_name.split('-')[4])#1 gesture type
                # location = int(data_file_name.split('-')[5])#1 torso location
                # orientation = int(data_file_name.split('-')[6])#1 face orientation
                # repetition = int(data_file_name.split('-')[7])#repetition number
                receiverid=data_file_name.split('-')[8]
                # print(height,label_1,location,orientation,repetition,receiverid)
                #Each file is a 20*20*T matrix, where the first dimension represents the
                #velocity along x axis ranging between [-2,+2] m/s, the second dimension represents
                #the velocity along y axis ranging between [-2,+2] m/s and the third dimension
                #represents the timestamps with 10Hz sampling rate.
                
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


                #label,output
                # if(receiverid!="r4.dat"):
                #     print(height,"no")
                #     continue
                # if(count1>1000):
                #     break
                print(receiverid,height,username)
                
                # Select Orientation
                # if (orientation not in [1,2,4,5]):
                #     continue
                col=csi_amp.shape[0]
                data_1=csi_amp.reshape(col,3,30)
                data_1=data_1.transpose(2,1,0)
                # print("data1shape:",data_1.shape)
                # Normalization
                data_normed_1 = normalize_data(data_1)
                
                # Update T_MAX
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]
                    # print(T_MAX)
            except Exception:
                continue

            data.append(data_normed_1.tolist())#转换成列表
            # print(np.array(data).shape)
            # print(height,'1')
            label.append(height) #追加label
            # print(label)
    print(np.array(data).shape)
    data = zero_padding(data, T_MAX)
    
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
    print(data.shape)
    data = np.expand_dims(data, axis=-1)
    
    label = np.array(label)
    print(label)
    # print(data)
    print(data.shape)
    return data, label


testdata,testlabel=load_data(data_dir,ALL_MOTION)
oldlabel=testlabel
big=186-155
small=155
newlabel=[]
leng=len(testlabel)
for i in range(0,leng):
    temp=(testlabel[i]-min(testlabel))/(max(testlabel)-min(testlabel))
    newlabel.append(temp)
testlabel = np.array(newlabel)

nfile = h5py.File('model_widar3_trained.h5')
model=keras.models.load_model(nfile)
model.summary()

label_test_pred = model.predict(testdata)
actsum=0
predsum=0
testlen=len(testdata)

print("testlen=",testlen)
countuser=0
totalheight=0
for n in range(0,testlen):
    
    pred=label_test_pred[n]*big+small
    act=testlabel[n]*big+small
    # print("------")
    countuser=countuser+1
    totalheight+=pred
    print(countuser,totalheight)
    # prediff=abs(pred-act)
    # baseline=abs(171-act) #?
    # predsum=predsum+prediff
    # actsum=actsum+baseline
    # print(label_test_pred[n]*big+small,label_test[n]*big+small)
# predaverage=predsum/testlen
# actaverage=actsum/testlen
# print(predaverage)
# print(actaverage)
predaverage=totalheight/countuser
print(predaverage)
