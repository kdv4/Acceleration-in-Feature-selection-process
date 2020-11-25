import kmeans
import correlation as cr
import pandas as pd
import serial_support as ss

#mandatory parameter needs to change
file='cars.csv'
text_indices=[1,8]
start=0
end=9
y=1

if __name__ == "__main__":
    dataset=pd.read_csv(file)

    #This is use to convert Text dat to Numeric Data
    for i in text_indices:
        temp=ss.encoder(dataset,i)
        title=dataset.columns[i]
        dataset[title]=temp

    #Part of dataset we want
    X= dataset.iloc[:,start:end]
    # print(X.head())
    category=list(X.columns)

    #TODO: Need to make it with for loop;
    X_car=ss.encoder(dataset,1)
    X_origin=ss.encoder(dataset,8)
    # #Change Value in dataset according to it's return value
    X['Car'] = X_car
    X['Origin']=X_origin

    #Direct Feature selection
    selected_features = cr.cal_vif(X) 
    print(selected_features.head())
    #[category,importances,indices]=cr.find_correlation(X,y)
    #cr.visualize_corrlation(category,importances,indices)

    #Kmeans+Feature Selection
    centroids=kmeans.kmeansCluster(X)
    ss.Write_XL(centroids,len(centroids),len(centroids[0]),category)

    dataset=pd.read_excel('centroids.xls')
    # [category,importances,indices]=cr.find_correlation(X)
    # cr.visualize_corrlation(category,importances,indices)
    selected_features = cr.cal_vif(dataset) 
    print(selected_features.head())


