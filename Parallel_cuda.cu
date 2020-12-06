#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <time.h>

float gpu_time_used;
#define I(row, col, ncols) (row * ncols + col)

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 

//f = number of features (from argv[1])
__global__ void get_dst(float *dst, float *x, float *mu_x, int f){
  int i = blockIdx.x;
  int j = threadIdx.x;

  //updated (in loop):
  dst[I(i, j, blockDim.x)] = (x[i*f] - mu_x[j*f]) * (x[i*f] - mu_x[j*f]);
  
  for(int l = 1 ; l<f ; l++)
  {
        dst[I(i, j, blockDim.x)] += (x[i*f +l] - mu_x[j*f + l]) *  (x[i*f +l] - mu_x[j*f + l]); 
  }
  //printf("%d %d %f \n",i,j,dst[I(i, j, blockDim.x)]);
}

__global__ void regroup(int *group, float *dst, int k){
  int i = blockIdx.x;
  int j;
  float min_dst;
  
  min_dst = dst[I(i, 0, k)];
  group[i] = 1;

  for(j = 1; j < k; ++j){
    if(dst[I(i, j, k)] < min_dst){
      min_dst = dst[I(i, j, k)];
      group[i] = j + 1;
    }
  }
}

//updated
__global__ void clear(float *sum_x, int *nx, int f){
  int j = threadIdx.x;
  for(int l = 0 ; l<f ; l++)
  {
        sum_x[j*f + l] = 0;
        nx[j*f + l] = 0;
  }
}

//changes : need 2d array for sum and nx also, f: number of features
__global__ void recenter_step1(float *sum_x, int *nx,
             float *x, int *group, int n, int f){
  int i;
  int j = threadIdx.x;

  for(i = 0; i < n; ++i){
    if(group[i] == (j + 1)){

      //loop through entire sum and n
      for(int l = 0 ; l<f ; l++)
      {
        sum_x[j*f + l] += x[i*f +l];
        nx[j*f + l]++; 
      }
    }
  }               
}

__global__ void recenter_step2(float *mu_x, float *sum_x,
         int *nx, int f){
  int j = threadIdx.x;

  // added loop
  for(int l = 0 ; l<f ; l++)
      {
        mu_x[j*f + l] = sum_x[j*f +l]/nx[j*f + l]; 
      }
}

void kmeans(int nreps, int n, int k,
            float *x_d, float *mu_x_d,
            int *group_d, int *nx_d,
            float *sum_x_d, float *dst_d, int f){
  int i;
  for(i = 0; i < nreps; ++i){
    get_dst<<<n,k>>>(dst_d, x_d, mu_x_d, f);
    regroup<<<n,1>>>(group_d, dst_d, k);
    clear<<<1,k>>>(sum_x_d, nx_d, f);
    recenter_step1<<<1,k>>>(sum_x_d, nx_d, x_d, group_d, n,f);
    recenter_step2<<<1,k>>>(mu_x_d, sum_x_d, nx_d,f);
  }
}

void read_data(float *x, float *mu_x, int *n, int *k,char* arg, int no_feature, int no_data);
void print_results(int *group, float *mu_x, int n, int k,char* arg,int no_feature);

int main(int argc,char* argv[]){
  
  //Argv 1: No of Features
  //Argv 2: Input path
  //Argv 3: No of datapoints
  //Argv 4: No of cluster
  
  /* cpu variables */
  int n=atoi(argv[3]); /* number of points */
  int k=atoi(argv[4]); /* number of clusters */
  int f=atoi(argv[1]); /*number of Features */
  int *group;
  float *x = NULL, *mu_x = NULL;
  x = (float*) malloc(atoi(argv[3])*atoi(argv[1])*sizeof(float));
  mu_x = (float*) malloc(atoi(argv[3])*atoi(argv[1])*sizeof(float));
  	
  /* gpu variables */
  int *group_d, *nx_d;
  float *x_d, *mu_x_d, *sum_x_d, *dst_d;

  /* read data from files on cpu */
  read_data(x, mu_x, &n, &k,argv[2], atoi(argv[1]), atoi(argv[3]));
	
  /* allocate cpu memory */
  group = (int*) malloc(n*sizeof(int));

  /* allocate gpu memory */
  CUDA_CALL(cudaMalloc((void**) &group_d,n*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**) &nx_d, f*k*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**) &x_d, f*n*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &mu_x_d, f*k*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &sum_x_d, f*k*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &dst_d, n*k*sizeof(float)));

  /* write data to gpu */
  CUDA_CALL(cudaMemcpy(x_d, x, f*n*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(mu_x_d, mu_x, f*k*sizeof(float), cudaMemcpyHostToDevice));
  
  
  /* perform kmeans */
  const auto start = std::chrono::high_resolution_clock::now();
  kmeans(1, n, k, x_d, mu_x_d, group_d, nx_d, sum_x_d, dst_d, atoi(argv[1]));

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
  std::cerr << "CUDA Took: " << duration.count() << "s" << " for "<<argv[3]<<" points." << std::endl;


gpu_time_used = duration.count();

  /* read back data from gpu */
  CUDA_CALL(cudaMemcpy(group, group_d, n*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(mu_x, mu_x_d, f*k*sizeof(float), cudaMemcpyDeviceToHost));
  //CUDA_CALL(cudaMemcpy(mu_y, mu_y_d, k*sizeof(float), cudaMemcpyDeviceToHost));

  /* print results and clean up */  
  print_results(group, mu_x, n, k,argv[3],atoi(argv[1]));


  free(x);
  free(mu_x);
  free(group);

  CUDA_CALL(cudaFree(x_d));
  CUDA_CALL(cudaFree(mu_x_d));
  CUDA_CALL(cudaFree(group_d));
  CUDA_CALL(cudaFree(nx_d));
  CUDA_CALL(cudaFree(sum_x_d));
  CUDA_CALL(cudaFree(dst_d));

  return 0;
}

void read_data(float *x, float *mu_x, int *n, int *k,char* arg, int no_feature, int no_data){
  FILE *fp;
  int i,j;
  fp = fopen(arg, "r");
  
  for(i = 0; i < no_data; i++)
    {
        for(j = 0; j < no_feature; j++)
        {
            fscanf(fp," %f", &x[i*no_feature+j]);   
        }    
    }
    
   fp = fopen("input/initCoord.txt", "r");
   for(i = 0; i < *k; i++)
    {
        for(j = 0; j < no_feature; j++)
        {
            fscanf(fp," %f", &mu_x[i*no_feature+j]);   
        }    
    }
  fclose(fp);
  
 
}


void print_results(int *group, float *mu_x, int n, int k,char* arg,int no_feature){
  FILE *fp;
  int i,j;
  std::string str(arg),str1,str2;
  str = "output/" + str;

   str1 = str + "_group_members.txt";
  fp = fopen(str1.c_str(), "w");
  for(i = 0; i < n; ++i){
    fprintf(fp, "%d\n", group[i]);
  }
  fclose(fp);
  
  str2 = str + "_centroids.txt";
  fp = fopen(str2.c_str(), "w");
  
  for(i=0;i < k; ++i){
   	for(j = 0; j < no_feature; ++j){
    		fprintf(fp, "%0.6f ", mu_x[i*no_feature+j]);
  	}
  	fprintf(fp, "\n");
  }
  
  fclose(fp);

  fp = fopen("CUDAtimes.txt", "a");
    fprintf(fp, "%0.6f\n", gpu_time_used);
fclose(fp);
}
