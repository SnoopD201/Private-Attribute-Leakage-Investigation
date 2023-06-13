import shutil
import os
import os.path
from wifilib import *
import random
L1 = random.sample(range(1, 6087), 160)
L2 = random.sample(range(1, 4000), 160)
L3 = random.sample(range(1, 3000), 160)
L4 = random.sample(range(1, 750), 160)
L5 = random.sample(range(1, 1125), 160)
L6 = random.sample(range(1, 750), 160)
L7 = random.sample(range(1, 750), 160)
L8 = random.sample(range(1, 750), 160)
L9 = random.sample(range(1, 750), 160)
L10 = random.sample(range(1, 1125), 160)
L11 = random.sample(range(1, 1125), 160)
L12 = random.sample(range(1, 1125), 160)
L13 = random.sample(range(1, 1125), 160)
L14 = random.sample(range(1, 1125), 160)
L15 = random.sample(range(1, 1125), 160)
L16 = random.sample(range(1, 1125), 160)

# print(L1)
# def remove_file(old_path, new_path):
#     # print(old_path)
#     # print(new_path)
#     filelist = os.listdir(old_path) #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
#     print(filelist)
#     for file in filelist:
#         file_path = os.path.join(old_path, file)
#         dst = os.path.join(new_path, file)
#         # print('file_path:', file_path)
#         # print('dst:', dst)
#         shutil.move(file_path, dst)
# count=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

path_to_data1 = "/achive/220301/shiyd/Widar-splituser/1"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L1:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/1"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1

# 16个人,每个人分别生成160个随机数,在user i文件夹内遍历,复制出来就行

path_to_data1 = "/achive/220301/shiyd/Widar-splituser/2"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L2:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/2"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/3"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L3:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/3"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/4"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L4:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/4"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1

path_to_data1 = "/achive/220301/shiyd/Widar-splituser/5"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L5:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/5"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/6"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L6:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/6"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/7"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L7:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/7"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/8"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L8:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/8"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/9"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L9:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/9"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/10"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L10:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/10"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/11"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L11:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/11"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/12"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L12:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/12"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
path_to_data1 = "/achive/220301/shiyd/Widar-splituser/13"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L13:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/13"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1

path_to_data1 = "/achive/220301/shiyd/Widar-splituser/14"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L14:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/14"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1

path_to_data1 = "/achive/220301/shiyd/Widar-splituser/15"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L15:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/15"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1

path_to_data1 = "/achive/220301/shiyd/Widar-splituser/16"
count = 0
for data_root, data_dirs, data_files in os.walk(path_to_data1):  # 遍历所有文件
    for data_file_name in data_files:
        # height = data_file_name.split('_')[1]
        # weight=data_file_name.split('_')[2]
        file_path = os.path.join(data_root, data_file_name)
        # print(file_path)
        # bf = read_bf_file(file_path)
        # csi_list = list(map(get_scale_csi, bf))
        # csi_np = (np.array(csi_list))
        # csi_amp = np.abs(csi_np)
        # isnan=np.isnan(csi_amp)
        # if(True in isnan):
        #     print(data_file_name)
        #     continue
        # else:
        #     new_path="/achive/220301/shiyd/Wiar-valid"
        #     dst=os.path.join(new_path,data_file_name)
        #     shutil.copy(file_path,dst)
        if count in L16:
            new_path = "/achive/220301/shiyd/Widar-validuser-new-new/16"
            dst = os.path.join(new_path, data_file_name)
            print(dst)
            shutil.copy(file_path, dst)
        count = count+1
