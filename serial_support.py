from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import kmeans
import correlation as cr

#this function is use to convert text data into numeric form
def encoder(dataset,indices):
    encoder = LabelEncoder()
    X_trans= dataset.iloc[:, indices].values
    X_trans = encoder.fit_transform(X_trans)
    return X_trans

#This function is used to save file as xls
def Write_XL(List,row,col,category,fileName): 
    from xlwt import Workbook

    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 1')
    
    #This loop is for writing header
    for i in range(col):
        sheet1.write(0,i,category[i])

    for i in range(row):
        for j in range(col):
            sheet1.write(i+1,j,List[i][j])

    wb.save(fileName)

def navie_byes(X_train,Y_train,X_test,Y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    y_pred = gnb.predict(X_test)
    acc=metrics.accuracy_score(Y_test, y_pred)
    return acc

def plot(X,Y):
    plt.plot(X,Y)
    plt.xlabel("No of Clusters")
    plt.ylabel("Accuracy")
    plt.show()

def check_kmenas(X_raw,Y_raw,start,end,category):
    X={}
    for i in range(start,end):
        centroids=kmeans.kmeansCluster(X_raw,i)
        Write_XL(centroids,len(centroids),len(centroids[0]),category,"output/centroids.xls")
    
        data_kmeans=pd.read_excel('output/centroids.xls')
        X_kmeans=data_kmeans.iloc[:,:] 
        
        selected_features = cr.cal_vif(X_kmeans)

        X_modify=X_raw.loc[:,selected_features.columns]

        X_train, X_test, Y_train, Y_test = train_test_split(X_modify, Y_raw, test_size = 0.2, random_state = 0)

        X[i]=navie_byes(X_train,Y_train,X_test,Y_test)

    plot(list(X.keys()),list(X.values()))
