import numpy as np
import pandas as pd
import sys
import random
from sklearn import preprocessing

file=sys.argv[1]
n=sys.argv[2]

data = pd.read_table(file, header=None)

#data.drop(labels=0, axis=1, inplace=True)

list=data.sample(2).values
list=list.tolist()
print(type(list))
print(list)
with open('input/initCoord.txt','w') as fp:
	for i in range(len(list)):
		fp.writelines(["%s " % item  for item in list[i]])
		fp.writelines("\n")
