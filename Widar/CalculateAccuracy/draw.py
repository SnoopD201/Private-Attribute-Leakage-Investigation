# coding:utf-8
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data_dir = "logheight.txt"
Train_Loss_list = []
Train_Mae_list = []
Valid_Loss_list = []
Valid_Mae_list = []
f1 = open(data_dir,'r')
data = []
#把训练结果输出到result.txt里，比较笨的办法，按字节位数去取数字结果
for line in f1:
    if(line.find('loss')>=0):
    # if (string.find(line, 'train') != -1):
        Train_Loss_list.append(float(line[67:73]))
        Train_Mae_list.append(float(line[81:87]))
    # if (string.find(line, 'valid') != -1):
        Valid_Loss_list.append(float(line[114:120]))
        Valid_Mae_list.append(float(line[132:138]))
f1.close()
#迭代了30次，所以x的取值范围为(0，30)，然后再将每次相对应的准确率以及损失率附在x上
# print(Train_Accuracy_list)
# x1=range(60)
# y1=Train_Accuracy_list
# y2=Valid_Accuracy_list
# plt.plot(x1,y1,label='Accuracy')
# plt.savefig('test1.png')
x1 = range(0, 30)
x2 = range(0, 30)
y1 = Train_Mae_list
y2 = Train_Loss_list
y3 = Valid_Mae_list
y4 = Valid_Loss_list
plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-',color='r')
plt.plot(x1, y1, 'o-',label="Train_Mae")
plt.plot(x1, y3, 'o-',label="Valid_Mae")
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-',label="Train_Loss")
plt.plot(x2, y4, '.-',label="Valid_Loss")
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.legend(loc='best')
plt.show()
plt.savefig("test2.png")