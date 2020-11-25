# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as r
import math

file='cars.csv'

def show_kmeans(X,no_cluster,y_kmeans,kmeans):
    # Visualising the clusters
    from colormap import rgb2hex
    for i in range(0,no_cluster):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 100, c = rgb2hex(r.randint(1,255),r.randint(1,255),r.randint(1,255)), label = f'Cluster {i+1}')


    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 120, c = 'yellow', label = 'Centroids')
    plt.title('Clusters of cars')
    plt.xlabel('Car Feature')
    plt.ylabel('Car MPG')
    plt.legend()
    plt.show()

def show_elbow(no_cluster,wcss):
    #Plot Number of cluster value graph
    plt.plot(range(1, no_cluster), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()




def kmeansCluster(dataset):
    
    #y = dataset.iloc[:, 0].values

    # Splitting the dataset into the Training set and Test set
    """from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    """
    # Feature Scaling
    """from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)"""

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    '''
    no_cluster=int(input("Enter Number of cluster you want to check: "))
    wcss = []
    for i in range(1, no_cluster):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    #To display graph of k v/s Wcss
    #show_elbow(no_cluster,wcss)

    #no_cluster=int(input("select no of cluter: "))
    '''

    no_cluster=int(math.sqrt(len(dataset)/2))
    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = no_cluster, init = 'k-means++', random_state = 42,max_iter=300)
    y_kmeans = kmeans.fit_predict(dataset)
    
    return kmeans.cluster_centers_
    #Visualize kmeans in 2D which is kind of not possible
    #kmeans_show(X,no_cluster,y_kmeans,kmeans)
    
if __name__ == "__main__":
    dataset=pd.read_csv(file)
    X= dataset.iloc[:,1:9]
    kmeansCluster(file)