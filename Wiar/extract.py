
f = open('weightlog.txt','r')
lines = f.readlines()
for lines in lines:
   if "1588/1588" in lines:
     print(lines)