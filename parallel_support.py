import pandas as pd
import numpy as np
from numpy import asarray
import re
import sys
import correlation as cr
import os
from sklearn.cluster import KMeans
import serial_support as ss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def Write_XL(OPfile,category):
	IPfile=pd.read_csv("output/centroids.txt",sep=" ",header=None)
	IPfile=IPfile.iloc[:,:-1]
	IPfile.columns=category
	IPfile.to_excel(OPfile,index=False)
	
def Write_TXT(filepath,dataset):
	with open(filepath,"w") as fp:	
		dataset.iloc[:,:].to_string(fp,index=False,header=False)
    
def init_centroid(dataset,n):
	samples=dataset.sample(n)
	Write_TXT("input/initCoord.txt",samples)

def init_centroids_k(X_train,n,category):
	kmeans = KMeans(n_clusters = n, init = 'k-means++', random_state = 42,max_iter=1)
	kmeans.fit_predict(X_train)
	centroids=kmeans.cluster_centers_
	ss.Write_XL(centroids,len(centroids),len(centroids[0]),category,"output/init_centroid_P.xls")
	data_kmeans=pd.read_excel('output/init_centroid_P.xls')
	Write_TXT("input/initCoord.txt",data_kmeans)
	
def plot(X,Y):
	plt.plot(X,Y)
	plt.xlabel("No of Clusters")
	plt.ylabel("Accuracy")
	plt.show()
	
def check_kmenas(X_raw,Y_raw,start,end,category):
	X={}
	for k in range(start,end):
		init_centroids_k(X_raw,k,category)
		os.system(f"./parallel_cuda {len(X_raw.columns)} input/X.txt {len(X_raw)} {k}")	    
		Write_XL("output/centroids_P.xls",category)
		data_kmeans=pd.read_excel('output/centroids_P.xls')
		X_kmeans=data_kmeans.iloc[:,:] 
		selected_features = cr.cal_vif(X_kmeans)
		X_modify=X_raw.loc[:,selected_features.columns]
		X_train, X_test, Y_train, Y_test = train_test_split(X_modify, Y_raw, test_size = 0.2, random_state = 0)
		X[k]=ss.navie_byes(X_train,Y_train,X_test,Y_test)
	plot(list(X.keys()),list(X.values()))
