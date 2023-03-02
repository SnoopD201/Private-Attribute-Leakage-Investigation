# 添加了对每个人的预测准确率判断


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
from keras.models import Model, Sequential, load_model
import keras.backend as K
from sklearn.metrics import confusion_matrix, mean_squared_error
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from wifilib import *
import random
from read_data import load_data

# Parameters
use_existing_model = False
fraction_for_test = 0.2
data_dir = '/export/shiyd/CSI-gesture/pushpull'
ALL_MOTION = [178, 161, 170, 160, 180, 172, 168,
              175, 165, 170, 175, 176, 155, 158, 175, 186]
# ALL_MOTION = [1,2,3,4]
# N_MOTION = len(ALL_MOTION)
N_MOTION = 1
n_epochs = 30
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_batch_size = 32  # 32
f_learning_rate = 0.001


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
    model_output = Dense(n_class, name='name_model_output')(x)

    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=keras.optimizers.Adam(lr=f_learning_rate),
                  loss='mse',
                  metrics=['mae', 'mse']
                  )
    return model


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
tf.random.set_seed(1)

# 数据预处理
# [data_train, data_test, label_train, label_test] = train_test_split(
#     data_dir, test_size=fraction_for_test)

# 加载数据
f = open('train.txt')
train_file_list = []
for line in f:
    train_file_list.append(line.strip())
f = open('test.txt')
test_file_list = []
for line in f:
    test_file_list.append(line.strip())

train_data, train_label, min_value, max_value, T_MAX = load_data(
    train_file_list, ALL_MOTION, cal_distance=True)

test_data, test_label, _, _, _ = load_data(
    test_file_list, ALL_MOTION, cal_distance=False, min_value=min_value, max_value=max_value, T_MAX=T_MAX)

print('\nTrain on ' + str(train_label.shape[0]) + ' samples\n' +
      'Test on ' + str(test_label.shape[0]) + ' samples\n')

# 加标签
# label_train = onehot_encoding(label_train, N_MOTION)

if use_existing_model:
    model = load_model('model_widar3_trained.h5')
    model.summary()
else:
    # 训练模型
    model = assemble_model(input_shape=(T_MAX, 30, 3, 1), n_class=N_MOTION)
    model.summary()
    model.fit({'name_model_input': train_data}, {'name_model_output': train_label},
              batch_size=n_batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_split=0.1, shuffle=True)
    print('Saving trained model...')
    model.save('model_widar3_trained.h5')

print('Testing...')
label_test_pred = model.predict(test_data)

test_label_actual = (test_label * (max_value-min_value)) + min_value
pred_label_actual = (label_test_pred * (max_value-min_value)) + min_value

predaverage = sum(abs(pred_label_actual-test_label_actual))/len(test_data)
actaverage =  sum(abs(171-test_label_actual))/len(test_data)
print("average prediction=", predaverage)
print("baseline=", actaverage)


# label_test_pred = model.predict(data_test)
# label_test_pred = np.argmax(label_test_pred, axis=-1) + 1

# print(mean_squared_error(label_test,label_test_pred))
# original=[]
# for i in range(0,len(label_test_pred)):
#     temp=label_test_pred[i]*big+small
#     original.append(temp)

# print(label_test_pred[0],label_test[0])
# """查看训练效果，从中随机提取20个数据下标"""
# indices = np.random.choice(len(data_test), size=50)
# house_data.loc[indices,:]
# count = 0
