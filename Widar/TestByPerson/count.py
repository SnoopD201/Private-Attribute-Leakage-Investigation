import os
import os.path

count=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
path_to_data = "/export/shiyd/CSI-resize/r4"
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
        if username=="user1":
            count[0]+=1
        elif username=="user2":
            count[1]+=1
        elif username=="user3":
            count[2]+=1
        elif username=="user4":
            count[3]+=1
        elif username=="user5":
            count[4]+=1
        elif username=="user6":
            count[5]+=1
        elif username=="user7":
            count[6]+=1
        elif username=="user8":
            count[7]+=1
        elif username=="user9":
            count[8]+=1
        elif username=="user10":
            count[9]+=1
        elif username=="user11":
            count[10]+=1
        elif username=="user12":
            count[11]+=1
        elif username=="user13":
            count[12]+=1
        elif username=="user14":
            count[13]+=1
        elif username=="user15":
            count[14]+=1
        elif username=="user16":
            count[15]+=1
            


print(count)    