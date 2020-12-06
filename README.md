#Ac Project
***In this project we are implementing serial and parallel code of feature selectiona.
We used kmenas prior to feature selection***

##How to run serial code?
python3 serial.py

##How to run Parallel code?
python3 CSVtoTXT.py path_of_csv_file
python3 init_centroids.py path_of_txt_file no_of_centroids
nvcc Paralle_cuda.cu -o parallel
./parallel no_feature path_of_txt_file no_data no_cluster
