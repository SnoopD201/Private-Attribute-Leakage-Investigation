import os
path_to_data='/achive/220301/shiyd/Wiar-validuser'  
for data_root, data_dirs, data_files in os.walk(path_to_data):#遍历所有文件
        for data_file_name in data_files:

            # print(file_path)
        
            currentuser=data_root.split('/')[5]
            # print(currentuser)
            if(currentuser=='10'):
                print("--------------")
                