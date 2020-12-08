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

file='Dataset/dataset.csv'
text_indices=[1]
out='1'
start=2

if __name__ == "__main__":
    dataset=pd.read_csv(file)
    end=len(dataset.columns)

    #This is use to convert Text dat to Numeric Data
    for i in text_indices:
        temp=ss.encoder(dataset,i)
        title=dataset.columns[i]
        dataset[title]=temp

    #Part of dataset we want
    X_raw= dataset.iloc[:,start:end]
    tmp=list(dataset.columns)
    category=tmp[start:end]

    if(out.isnumeric()):
        Y_raw=dataset.iloc[:,int(out)]
        # category.insert(0,tmp[int(out)])
    else:
        Y_raw= dataset.loc[:,out]
        # category.insert(0,out)   
 
    #Direct Classifier
    start=time()
    X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y_raw, test_size = 0.2, random_state = 0)
    print("Accuracy of Raw data is: "+str(ss.navie_byes(X_train,Y_train,X_test,Y_test)))
    end=time()
    print("Time taken by it: "+str((end-start)*1000)+" ms")
    
   #Direct Feature selection
    start=time()
    selected_features = cr.cal_vif(X_raw) 
    X_train, X_test, Y_train, Y_test = train_test_split(selected_features, Y_raw, test_size = 0.2, random_state = 0)
    print("Accuracy using direct feature selection is: "+str(ss.navie_byes(X_train,Y_train,X_test,Y_test)))
    end=time()
    print("Time taken by it: "+str((end-start)*1000)+" ms")
    
    #Kmeans Serial+Feature Selection
    start_c=int(math.sqrt(len(dataset)/2))
    #ss.check_kmenas(X_raw,Y_raw,start_c,start_c*2+5,category)
    start=time()
    centroids=kmeans.kmeansCluster(X_raw,int(input("Enter number of Cluster: ")))
    ss.Write_XL(centroids,len(centroids),len(centroids[0]),category,"output/centroids.xls")
    
    data_kmeans=pd.read_excel('output/centroids.xls')
    X_kmeans=data_kmeans.iloc[:,:] 
    
    selected_features = cr.cal_vif(X_kmeans)
    X_modify=X_raw.loc[:,selected_features.columns]

    X_train, X_test, Y_train, Y_test = train_test_split(X_modify, Y_raw, test_size = 0.2, random_state = 0)
    print("Accuracy using Kmenas+feature selection is: "+str(ss.navie_byes(X_train,Y_train,X_test,Y_test)))

    end=time()
    print("Time taken by it: "+str((end-start)*1000)+" ms")
    
   
    #Kmeans Parallel+Feature Selection
   	
    #4.1 Write in TXT file
    ps.Write_TXT("input/X.txt",X_raw)
    os.system(f"nvcc Parallel_cuda1.cu -o parallel_cuda1")

    start_c=int(math.sqrt(len(X_raw)/2))
    ps.check_kmenas(X_raw,Y_raw,start_c,start_c*3,category)
    
    
    #4.2 Initial Centroids
    k=int(input("Enter No of cluster you want for parallel kmeans: "))
    #ps.init_centroid(X_raw,k)
    ps.init_centroids_k(X_raw,k,category)
    
    start=time()
    #4.3 Call cuda Kmeans
    os.system(f"./parallel_cuda1 {len(X_raw.columns)} input/X.txt {len(X_raw)} {k}")	    
    
    #4.4 Write back into Excel
    ps.Write_XL("output/centroids_P.xls",category)
    
    #4.5 Feasture selection
    data_kmeans=pd.read_excel('output/centroids_P.xls')
    X_kmeans=data_kmeans.iloc[:,:] 
    
    selected_features = cr.cal_vif(X_kmeans)
    X_modify=X_raw.loc[:,selected_features.columns]

    X_train, X_test, Y_train, Y_test = train_test_split(X_modify, Y_raw, test_size = 0.2, random_state = 0)
    print("Accuracy using Kmenas Parallel+feature selection is: "+str(ss.navie_byes(X_train,Y_train,X_test,Y_test)))
    end=time()
    print("Time taken by it: "+str((end-start)*1000)+" ms")
    
