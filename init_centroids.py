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
with open('input/initCoord','w') as fp:
	file.writelines('\t'.join(str(j) for j in i) + '\n' for i in top_list)
