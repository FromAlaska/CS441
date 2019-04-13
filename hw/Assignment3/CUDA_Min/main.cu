// Jim Samson
// 04 April 2019
// Cuda Minimum Finding
// Homework Part 2
// 

#include <stdio.h>
#include <limits.h>

#define HIGHEST_VALUE 8000000
#define THREADS 8

__global__ void findLowest(int numMin, int *array_val, int *cudaResult ) {
    int low = threadIdx.x * numMin;
    int high = low + numMin -1;
    int min = array_val[low];
    for (unsigned int i = low; i < high; i++){
        if(array_val[i] < min){
            min = array_val[i];
        }
    }
    cudaResult[threadIdx.x] = min;
    printf("Thread %d returned: %d \n", threadIdx.x, min);
}

int main() {
    int *array_val;
    int *cudaResult;
    int min = INT_MAX;
    int testMin = INT_MAX;
    int *cuda_return;
    int *dev_a;

    array_val = (int *) malloc(sizeof(int)*HIGHEST_VALUE);
    cudaResult = (int *) malloc(sizeof(int)*THREADS);

    for(unsigned int i = 0; i < HIGHEST_VALUE; i++) {
        array_val[i] = rand() % 100000;
        if (testMin > array_val[i]){
            testMin = array_val[i];
        } 
    }

    printf("Minimum value is: %d \n", testMin);

    int numMin = HIGHEST_VALUE / THREADS;
   
    cudaMalloc((void**)&cuda_return, HIGHEST_VALUE*sizeof(int));
    cudaMalloc((void**)&dev_a, HIGHEST_VALUE*sizeof(int));
    cudaMemcpy(dev_a, array_val, HIGHEST_VALUE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_return, cudaResult, THREADS*sizeof(int), cudaMemcpyHostToDevice);
    findLowest<<<1,8>>>(numMin, dev_a, cuda_return);
    cudaMemcpy(cudaResult, cuda_return, THREADS*sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned int i = 0; i < THREADS; i++) {
        if(min > cudaResult[i]) {
            min = cudaResult[i];
        }
    }

    cudaFree(cuda_return);
    cudaFree(dev_a);
    printf("The Cuda Value is %d \n", min); 
}