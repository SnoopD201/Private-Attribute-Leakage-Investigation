# https://github.com/SnoopD201/Private-Attribute-Leakage-Investigation
wifilib.py:read csi data

--version:
python 3.6.13
tensorflow 2.0.0
keras 2.3.1

dataset:http://tns.thss.tsinghua.edu.cn/widar3.0/#

存在nan值的CSI:r2,r3,r5,r6
r1和r4正常
即最靠近发射天线的两个接收器





#### 性别,身高,体重都跑一遍
#### 训练集里有没有包括测试集里的人



预测单个人的16个人,15个人输入数据,一个预测


record 3.2:


wiar:2207 samples

widar:over 19000 samples

实验补充:

分动作测量,Widar4个主要动作
wiar: 16个动作.



下面考虑训练集中每次去掉一个人后预测的准确度.


如何衡量准确度
考虑可能出现的误差


两个数据库规模不一致如何处理
