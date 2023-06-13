import os
from wifilib import *

T_MAX=0
path_to_data = "/achive/220301/shiyd/Widar-splituser"
for data_root, data_dirs, data_files in os.walk(path_to_data):  # 遍历所有文件
    for data_file_name in data_files:
        file_path = os.path.join(data_root, data_file_name)
        bf = read_bf_file(file_path)
        csi_list = list(map(get_scale_csi, bf))
        csi_np = (np.array(csi_list))
        csi_amp = np.abs(csi_np)

        col = csi_amp.shape[0]
        data_1 = csi_amp.reshape(col, 3, 30)
        data_1 = data_1.transpose(2, 1, 0)

        # Update T_MAX
        if T_MAX < np.array(data_1).shape[2]:
            T_MAX = np.array(data_1).shape[2]
            print(T_MAX)

print("final result =",T_MAX)