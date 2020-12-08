# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as r
import math
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer


def show_elbow(no_cluster,wcss):
    #Plot Number of cluster value graph
    plt.plot(range(1, no_cluster), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def Elbow(sample,feature):
	# Generate synthetic dataset with 8 random clusters
	X, y = make_blobs(n_samples=sample, n_features=feature, centers=8, random_state=42)

	# Instantiate the clustering model and visualizer
	model = KMeans()
	visualizer = KElbowVisualizer(model, k=(4,40), metric='calinski_harabasz', timings=False,locate_elbow=True)

	visualizer.fit(X)        # Fit the data to the visualizer
	visualizer.show() 

def kmeansCluster(X_train,no_cluster):
    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = no_cluster, init = 'k-means++', random_state = 42,max_iter=300)
    kmeans.fit_predict(X_train)
    return kmeans.cluster_centers_
