
f = open('read-height.txt','r')
lines = f.readlines()
for lines in lines:
   if "2304/2304" in lines:
     print(lines)