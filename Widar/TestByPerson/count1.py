import os
import os.path

count=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
path_to_data = "/export/shiyd/CSI-/r4-test"
big=0
small=200
for data_root, data_dirs, data_files in os.walk(path_to_data):  # 遍历所有文件
    for data_file_name in data_files:
        currentname = data_file_name
        filename = currentname.split('.')[0]
        # 去除后缀后的文件名
        # print(filename)
        # 查看user:
        username = filename.split('-')[3]
        # receiverid=data_file_name.split('-')[8]
        
        # print(userrole)
        # 总路径
        height=int(data_file_name.split('-')[1])
        if(big<height):
            big=height
        if(small>height):
            small=height
        



print(big)
print(small)    