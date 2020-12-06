import pandas as pd
import os
import serial_support as ss

file='Dataset/dataset.csv'
op='input/X.txt'
text_indices=[1,8]

def writeTxt(dataset,filepath):
    n=len(dataset.columns)
    with open(filepath,"w") as fp:
    	dataset.iloc[1:-1,:].to_string(fp,index=False)

if __name__=="__main__":
    dataset=pd.read_csv(file)
    for i in text_indices:
        temp=ss.encoder(dataset,i)
        title=dataset.columns[i]
        dataset[title]=temp
    writeTxt(dataset)
