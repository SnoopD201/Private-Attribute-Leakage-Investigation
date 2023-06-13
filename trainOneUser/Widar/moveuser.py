

import shutil
import os
import os.path			#导入模块
list1 = [0 for n in range(1,17)]
# path = '/achive/220301/shiyd/Widar-splituser/'	#设置创建后文件夹存放的位置
path_to_data = '/achive/220301/shiyd/CSI-resize/r4'	#设置创建后文件夹存放的位置
for data_root, data_dirs, data_files in os.walk(path_to_data):#遍历所有文件
    for data_file_name in data_files:
        # print(data_file_name)
        
        src = os.path.join(data_root, data_file_name)
        if "178-70-user1" in data_file_name:
                list1[0]=list1[0]+1
                new_path="/achive/220301/shiyd/Widar-splituser/1"
                dst=os.path.join(new_path,data_file_name)
                shutil.copy(src,dst)
        elif "user2" in data_file_name:
            list1[1]=list1[1]+1
            new_path="/achive/220301/shiyd/Widar-splituser/2"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user3" in data_file_name:
            list1[2]=list1[2]+1
            new_path="/achive/220301/shiyd/Widar-splituser/3"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst) 
        elif "user4" in data_file_name:
            list1[3]=list1[3]+1
            new_path="/achive/220301/shiyd/Widar-splituser/4"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user5" in data_file_name:
            list1[4]=list1[4]+1
            new_path="/achive/220301/shiyd/Widar-splituser/5"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user6" in data_file_name:
            list1[5]=list1[5]+1
            new_path="/achive/220301/shiyd/Widar-splituser/6"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user7" in data_file_name:
            list1[6]=list1[6]+1
            new_path="/achive/220301/shiyd/Widar-splituser/7"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user8" in data_file_name:
            list1[7]=list1[7]+1
            new_path="/achive/220301/shiyd/Widar-splituser/8"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user9" in data_file_name:
            list1[8]=list1[8]+1
            new_path="/achive/220301/shiyd/Widar-splituser/9"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user10" in data_file_name:
            list1[9]=list1[9]+1
            new_path="/achive/220301/shiyd/Widar-splituser/10"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user11" in data_file_name:
            list1[10]=list1[10]+1
            new_path="/achive/220301/shiyd/Widar-splituser/11"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user12" in data_file_name:
            list1[11]=list1[11]+1
            new_path="/achive/220301/shiyd/Widar-splituser/12"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user13" in data_file_name:
            list1[12]=list1[12]+1
            new_path="/achive/220301/shiyd/Widar-splituser/13"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user14" in data_file_name:
            list1[13]=list1[13]+1
            new_path="/achive/220301/shiyd/Widar-splituser/14"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user15" in data_file_name:
            list1[14]=list1[14]+1
            new_path="/achive/220301/shiyd/Widar-splituser/15"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
        elif "user16" in data_file_name:
            list1[15]=list1[15]+1
            new_path="/achive/220301/shiyd/Widar-splituser/16"
            dst=os.path.join(new_path,data_file_name)
            shutil.copy(src,dst)
print(list1)











