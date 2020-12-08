import kmeans
import correlation as cr
import pandas as pd
import serial_support as ss
from sklearn.model_selection import train_test_split
import math
from time import time
import warnings
import sys
import os
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import parallel_support as ps
import matplotlib.pyplot as plt

file='Dataset/covtype.csv'
text_indices=[]
start_C=1
out='0'

def plot(X,YS,YP):
	speed=[]
	for i in range(len(YP)):
		speed.append(YS[i]/YP[i])
	plt.plot(X,speed)
	plt.xlabel("No fo Data Points")
	plt.ylabel("Speed Up")
	
	#plt.xticks()
	plt.show()

if __name__ == "__main__":
	dataset=pd.read_csv(file)
	cnt=0
	listS=[]
	listP=[]
	listD=[]
	end_R=50
	for i in text_indices:
        	temp=ss.encoder(dataset,i)
        	title=dataset.columns[i]
        	dataset[title]=temp
    
	os.system("nvcc Parallel_cuda.cu -o parallel_cuda")
	while(end_R<len(dataset)):
		
		listD.append(end_R)
		#This is use to convert Text dat to Numeric Data

		#Part of dataset we want
		X_raw= dataset.iloc[0:end_R,start_C:-1]
		tmp=list(dataset.columns)
		category=tmp[start_C:-1]

		if(out.isnumeric()):
			Y_raw=dataset.iloc[:end_R,int(out)]
		else:
			Y_raw= dataset.loc[:end_R,out]   

		#Kmenas+Feature selection
		k=30

		start=time()
		centroids=kmeans.kmeansCluster(X_raw,k)
		end=time()
		calc=(end-start)*1000
		listS.append(calc)
		#print("Time taken by it: "+str((end-start)*1000)+" ms")

		#Kmeans Parallel+Feature Selection

		#4.2 Initial Centroids
		#ps.init_centroid(X_raw,k)
		ps.init_centroids_k(X_raw,k,category)
		ps.Write_TXT("input/X.txt",X_raw)
		
		start=time()
		#4.3 Call cuda Kmeans
		os.system(f"./parallel_cuda {len(X_raw.columns)} input/X.txt {len(X_raw)} {k}")
		end=time()
		calc=(end-start)*1000
		listP.append(calc)
		#print("Time taken by it: "+str((end-start)*1000)+" ms")
		cnt+=1
		end_R=50*pow(2,cnt)
		
	print(listD)
	print(listS)
	print(listP)
	plot(listD,listS,listP)
