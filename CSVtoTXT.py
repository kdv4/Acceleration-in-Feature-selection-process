import pandas as pd
import os
import serial_support as ss

file='cars.csv'
op='./TxtFiles'
text_indices=[1,8]

def writeTxt(dataset):
    n=len(dataset.columns)
    for i in range(0,n-1):
        with open(op+"/"+f"X{i+1}.txt","w") as fp:
            dataset.iloc[:,i].to_string(fp,index=False)

if __name__=="__main__":
    dataset=pd.read_csv(file)
    for i in text_indices:
        temp=ss.encoder(dataset,i)
        title=dataset.columns[i]
        dataset[title]=temp
    writeTxt(dataset)