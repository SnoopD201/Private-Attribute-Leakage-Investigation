# rearrangeHAR
wifilib.py:read csi data

--version:
python 3.6.13
tensorflow 2.0.0
keras 2.3.1

dataset:http://tns.thss.tsinghua.edu.cn/widar3.0/#

存在nan值的CSI:r2,r3,r5,r6
r1和r4正常
即最靠近发射天线的两个接收器



- 1.性别分类,拿CSI原始数据
- 2.扩展到所有数据,分批输入测试,一根天线
- 3.扩展到所有天线?
- 4.按照动作分类测试,4个主要动作和其他次要动作------数据集处理


#### 性别,身高,体重都跑一遍
#### 训练集里有没有包括测试集里的人


- 写论文---模板      √             找类似的文章  






实验:
每个sample(每个人)  跑r4天线的所有的身高体重性别
分训练集和数据集

所有6根天线的,删掉不好的数据

预测单个人的16个人,15个人输入数据,一个预测