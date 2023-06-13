# no 10
# use BVP data
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
from keras.models import Model, load_model
import keras.backend as K
from sklearn.metrics import confusion_matrix
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from wifilib import *

# Parameters
use_existing_model = False
fraction_for_test = 0.05
data_dir = '/achive/220301/shiyd/Wiar-validuser'
ALL_MOTION = [1, 2]
# gender
# ALL_MOTION = [1, 2, 3, 4, 5, 6]
# ALL_MOTION = [1,2,3,4]
N_MOTION = len(ALL_MOTION)  # 2
T_MAX = 531
n_epochs = 30
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_batch_size = 32  # 32
f_learning_rate = 0.001
exclude_user = [1,2,3,6,8,10]
# 标准化
# data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]


def normalize_data(data_1):
    data_1_max = np.concatenate(
        (data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate(
        (data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
    data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
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

# One-Hot 编码
# label(list)=>_label(ndarray): [N,]=>[N,num_class]


def onehot_encoding(label, num_class):
    label = np.array(label).astype('int32')
    label = np.squeeze(label)
    _label = np.eye(num_class)[label - 1]  # from label to onehot
    return _label


def load_data(path_to_data, motion_sel):
    global T_MAX
    global count1
    data = []
    label = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):  # 遍历所有文件
        for data_file_name in data_files:
            count1 = count1+1
            file_path = os.path.join(data_root, data_file_name)
            try:
                # 拆分文件名
                currentuser = int(data_root.split('/')[5])
                print(currentuser)
                if(currentuser in exclude_user):
                    continue
                 # {'__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Sun Nov 11 21:52:08 2018',
                 # '__version__': '1.0', '__globals__': [], 'velocity_spectrum_ro'[data....]
                 # user1-1-1-1-1-1-1e-07-100-20-100000-L0.mat
                # 0 gender type  male:1,female:2
                gender_1 = int(data_file_name.split('_')[0])

                if(gender_1 not in motion_sel):
                    continue
                if(count1 > 3000):
                    break
                bf = read_bf_file(file_path)
                csi_list = list(map(get_scale_csi, bf))
                csi_np = (np.array(csi_list))
                csi_amp = np.abs(csi_np)
                # Select Motion
                # if (label_1 not in motion_sel):
                #     continue
                isnan = np.isnan(csi_amp)
                if(True in isnan):
                    print(data_file_name)
                    # count2+=1
                    continue

                # Select Location
                # if (location not in [1,2,3,5]):
                #     continue
                col = csi_amp.shape[0]
                data_1 = csi_amp.reshape(col, 3, 30)
                data_1 = data_1.transpose(2, 1, 0)
                # print("data1shape:",data_1.shape)
                # Normalization
                data_normed_1 = normalize_data(data_1)

                # label,output

                # # Normalization
                # data_normed_1 = normalize_data(data_1)

                # Update T_MAX
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]
            except Exception:
                continue

            data.append(data_normed_1.tolist())  # 转换成列表
            label.append(gender_1)  # 追加label
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

# 建立模型?


def assemble_model(input_shape, n_class):
    model_input = Input(shape=input_shape, dtype='float32',
                        name='name_model_input')

    x = TimeDistributed(Conv2D(16, kernel_size=(3, 3), activation='relu', data_format='channels_last',
                               input_shape=input_shape))(model_input)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = TimeDistributed(Dropout(f_dropout_ratio))(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = GRU(n_gru_hidden_units, return_sequences=False)(x)
    x = Dropout(f_dropout_ratio)(x)
    model_output = Dense(n_class, activation='softmax',
                         name='name_model_output')(x)

    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )
    return model


print(sys.argv)
if len(sys.argv) < 2:
    print('Please specify GPU ...')
    exit(0)
if (sys.argv[1] == '1' or sys.argv[1] == '0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    config = tf.compat.v1.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    tf.random.set_seed(1)
else:
    print('Wrong GPU number, 0 or 1 supported!')
    exit(0)
count1 = 0
# 加载数据,
data, label = load_data(data_dir, ALL_MOTION)
print('\nLoaded dataset of ' +
      str(label.shape[0]) + ' samples, each sized ' + str(data[0, :, :].shape) + '\n')


# train_test_split
# 数据,标签,样本占比0.1
[data_train, data_test, label_train, label_test] = train_test_split(
    data, label, test_size=fraction_for_test)
print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' +
      'Test on ' + str(label_test.shape[0]) + ' samples\n')

# print(label_train)
# 加标签
label_train = onehot_encoding(label_train, N_MOTION)


if use_existing_model:
    model = load_model('model_widar3_trained.h5')
    model.summary()
else:
    # 训练模型
    model = assemble_model(input_shape=(T_MAX, 30, 3, 1), n_class=N_MOTION)
    model.summary()
    model.fit({'name_model_input': data_train}, {'name_model_output': label_train},
              batch_size=n_batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_split=0.1, shuffle=True)
    print('Saving trained model...')
    model.save('model_widar3_trained.h5')

print('Testing...')
label_test_pred = model.predict(data_test)
label_test_pred = np.argmax(label_test_pred, axis=-1) + 1

cm = confusion_matrix(label_test, label_test_pred)
print(cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.around(cm, decimals=2)
print(cm)

test_accuracy = np.sum(label_test == label_test_pred) / (label_test.shape[0])
print(test_accuracy)
print("T_MAX=", T_MAX)
