# import os
# for i in range(1,11):
#     path=r'/home/ydshi/Widar/CSI-reader/github/rearrangeHAR/trainOneUser/Wiar/Gender/test-'+str(i)+'.py'
#     file=open(path,'w')
#     file.write("#")
dataroot="1/hfskljfa/23/fds/sd/10"
user=int(dataroot.split('/')[5])
print(user)
nouser=[1,2,3,4,5,10]
if(user in nouser):
    print("-============")
    