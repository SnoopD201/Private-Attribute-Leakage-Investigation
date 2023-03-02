from __future__ import print_function

import os
import threading

import time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from wifilib import *
from read_data import DataGenerator
import keras
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, \
    TimeDistributed
from keras.models import Model

# Parameters
use_existing_model = False
fraction_for_test = 0.2
train_data_dir = '/export/shiyd/CSI-/r4-train'
test_data_dir = '/export/shiyd/CSI-/r4-test'

f_dropout_ratio = 0.5
n_gru_hidden_units = 128
f_learning_rate = 0.001


def regression_model(input_shape, n_class):
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


# ALL_MOTION = [1,2,3,4]
# N_MOTION = len(ALL_MOTION)
N_MOTION = 1
T_MAX = 0
n_epochs = 30
batch_size = 32  # 32
tmax = 3879
# 活动识别
# 隐私预测(性别,身高,体重-baseline)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
tf.random.set_seed(1)
if use_existing_model:
    model = load_model('model_widar3_trained.h5')
    model.summary()
else:
    # 训练模型
    model = regression_model(input_shape=(tmax, 30, 3, 1), n_class=N_MOTION)
    model.summary()

    training_generator = DataGenerator(train_data_dir, batch_size)
    validation_generator = DataGenerator(test_data_dir, batch_size)
    print('  wf>>>>>>>>>>>>>>>>>>>>> PID:%d  ident:%d' %
          (os.getpid(), threading.currentThread().ident))
    start_time = time.time()
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4,
                        epochs=n_epochs,
                        steps_per_epoch=(25837-5160)//batch_size,
                        validation_steps=5160//batch_size)
    print('Total used time: %d ' % (time.time() - start_time))
    # model.fit({'name_model_input': data_train}, {'name_model_output': label_train},
    #           batch_size=n_batch_size,
    #           epochs=n_epochs,
    #           verbose=1,
    #           validation_split=0.1, shuffle=True)
    print('Saving trained model...')
    model.save('model_widar3_trained.h5')

# print(sys.argv)
# if len(sys.argv) < 2:
#     print('Please specify GPU ...')
#     exit(0)
# if (sys.argv[1] == '1' or sys.argv[1] == '0'):
#     os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
#     config = tf.compat.v1.ConfigProto()
#     #config.gpu_options.per_process_gpu_memory_fraction = 0.4
#     config.gpu_options.allow_growth = True
#     tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#     tf.random.set_seed(1)
# else:
#     print('Wrong GPU number, 0 or 1 supported!')
#     exit(0)

# config = tf.compat.v1.ConfigProto()
# #config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# count1=0
# #加载数据,
# data, label = load_data(data_dir, ALL_MOTION)
# oldlabel=label
# big=max(label)-min(label)
# small=min(label)
# newlabel=[]
# leng=len(label)
# for i in range(0,leng):
#     temp=(label[i]-min(label))/(max(label)-min(label))
#     newlabel.append(temp)
# label = np.array(newlabel)

# print('\nLoaded dataset of ' + str(label.shape[0]) + ' samples, each sized ' + str(data[0, :, :].shape) + '\n')

# [data_train, data_test, label_train, label_test] = train_test_split(data, label, test_size=fraction_for_test)
# print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' + \
#       'Test on ' + str(label_test.shape[0]) + ' samples\n')


# print('Testing...')
# label_test_pred = model.predict(data_test)
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


# actsum=0
# predsum=0
# testlen=len(data_test)
# for n in range(0,testlen):
#     pred=label_test_pred[n]*big+small
#     act=label_test[n]*big+small
#     prediff=abs(pred-act)
#     baseline=abs(171-act) #?
#     predsum=predsum+prediff
#     actsum=actsum+baseline
#     # print(label_test_pred[n]*big+small,label_test[n]*big+small)
# predaverage=predsum/testlen
# actaverage=actsum/testlen
# print(predaverage)
# print(actaverage)


# cm = confusion_matrix(label_test, label_test_pred)
# print(cm)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# cm = np.around(cm, decimals=2)
# print(cm)

# test_accuracy = np.sum(label_test == label_test_pred) / (label_test.shape[0])
# print(test_accuracy)


# def load_data(path_to_data, motion_sel):
#     global T_MAX
#     global count1
#     data = []
#     label = []
#     for data_root, data_dirs, data_files in os.walk(path_to_data):#遍历所有文件
#         for data_file_name in data_files:
#             count1=count1+1
#             file_path = os.path.join(data_root, data_file_name)
#             try:
#                 height=int(data_file_name.split('-')[1])
#                 #拆分文件名
#                 bf = read_bf_file(file_path)
#                 csi_list = list(map(get_scale_csi, bf))
#                 csi_np = (np.array(csi_list))
#                 csi_amp = np.abs(csi_np)

#                 receiverid=data_file_name.split('-')[8]

#                 if(height not in motion_sel):
#                     print(height)
#                     continue
#                 #label,output
#                 # if(receiverid!="r4.dat"):
#                 #     print(height,"no")
#                 #     continue
#                 if(count1>5000):
#                     break
#                 print(receiverid,height,count1)

#                 # Select Orientation
#                 # if (orientation not in [1,2,4,5]):
#                 #     continue
#                 col=csi_amp.shape[0]
#                 data_1=csi_amp.reshape(col,3,30)
#                 data_1=data_1.transpose(2,1,0)
#                 # print("data1shape:",data_1.shape)
#                 # Normalization
#                 data_normed_1 = normalize_data(data_1)

#                 # Update T_MAX
#                 if T_MAX < np.array(data_1).shape[2]:
#                     T_MAX = np.array(data_1).shape[2]
#                     # print(T_MAX)
#             except Exception:
#                 continue

#             data.append(data_normed_1.tolist())#转换成列表
#             # print(np.array(data).shape)
#             # print(height,'1')
#             label.append(height) #追加label
#             # print(label)
#     print("1------(",np.array(data).shape)
#     data = zero_padding(data, T_MAX)
#     # data = data.transpose(3,0,1,2)

#     data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
#     print("2------(",data.shape)
#     data = np.expand_dims(data, axis=-1)

#     label = np.array(label)
#     print("3------(",label)
#     # print(data)
#     print("4------(",data.shape)
#     return data, label
