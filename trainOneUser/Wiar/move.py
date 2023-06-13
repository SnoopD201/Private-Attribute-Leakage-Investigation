import shutil
import os
import os.path
from wifilib import *

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
path_to_data = "/achive/220301/shiyd/Wiar-valid"
for data_root, data_dirs, data_files in os.walk(path_to_data):  # 遍历所有文件
    for data_file_name in data_files:
        height = data_file_name.split('_')[1]
        weight=data_file_name.split('_')[2]
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
        if height=='173' and weight=='85':
            new_path="/achive/220301/shiyd/Wiar-validuser/1"
            dst=os.path.join(new_path,data_file_name)
            print(dst)
            shutil.copy(file_path,dst)
        elif height=='180' and weight=='75':
            new_path="/achive/220301/shiyd/Wiar-validuser/2"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(file_path,dst)
        elif height=='165' and weight=='65':
            new_path="/achive/220301/shiyd/Wiar-validuser/3"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(file_path,dst)
        elif height=='160' and weight=='60':
            new_path="/achive/220301/shiyd/Wiar-validuser/4"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(file_path,dst)
        elif height=='162' and weight=='53':
            new_path="/achive/220301/shiyd/Wiar-validuser/5"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(file_path,dst)
        elif height=='170' and weight=='60':
            new_path="/achive/220301/shiyd/Wiar-validuser/6"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(file_path,dst)
        elif height=='165' and weight=='50':
            new_path="/achive/220301/shiyd/Wiar-validuser/7"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(file_path,dst)
        elif height=='155' and weight=='65':
            new_path="/achive/220301/shiyd/Wiar-validuser/8"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(file_path,dst)
        elif height=='180' and weight=='85':
            new_path="/achive/220301/shiyd/Wiar-validuser/9"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(file_path,dst)
        elif height=='175' and weight=='70':
            new_path="/achive/220301/shiyd/Wiar-validuser/10"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(file_path,dst)

















    
        