import numpy as np
import os
import kmeans_support as kms
data_file_name="data.csv"

if __name__ == "__main__":
    filename = os.path.dirname(__file__) + data_file_name
    data_points = np.genfromtxt(filename, delimiter=",")
    centroids = kms.create_centroids()
    total_iteration = 100
    
    [cluster_label, new_centroids] = kms.iterate_k_means(data_points, centroids, total_iteration)
    #print(cluster_label)
    #print(new_centroids)
    kms.print_label_data([cluster_label, new_centroids])
    print()
