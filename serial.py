import kmeans
import correlation as cr
import pandas as pd
import serial_support as ss
from sklearn.model_selection import train_test_split
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

#TODO: Find high dimensional dataset 

#mandatory parameter needs to change
file='dataset.csv'
text_indices=[1]
start=2
out='1'
delimiter=';'

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

    X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y_raw, test_size = 0.2, random_state = 0)
    print("Accuracy of Raw data is: "+str(ss.navie_byes(X_train,Y_train,X_test,Y_test)))
    

    #Direct Feature selection
    selected_features = cr.cal_vif(X_raw) 
    X_train, X_test, Y_train, Y_test = train_test_split(selected_features, Y_raw, test_size = 0.2, random_state = 0)
    print("Accuracy using direct feature selection is: "+str(ss.navie_byes(X_train,Y_train,X_test,Y_test)))
    
    
    #Kmeans+Feature Selection
    
    start=int(math.sqrt(len(dataset)/2))
    ss.check_kmenas(X_raw,Y_raw,start,start*2+5,category)

    centroids=kmeans.kmeansCluster(X_raw,18)
    ss.Write_XL(centroids,len(centroids),len(centroids[0]),category)
    
    data_kmeans=pd.read_excel('centroids.xls')
    X_kmeans=data_kmeans.iloc[:,:] 
    
    selected_features = cr.cal_vif(X_kmeans)

    X_modify=X_raw.loc[:,selected_features.columns]

    X_train, X_test, Y_train, Y_test = train_test_split(X_modify, Y_raw, test_size = 0.2, random_state = 0)
    print("Accuracy using Kmenas+feature selection is: "+str(ss.navie_byes(X_train,Y_train,X_test,Y_test)))
    
