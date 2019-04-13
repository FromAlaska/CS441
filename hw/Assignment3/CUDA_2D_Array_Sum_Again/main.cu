// Jim Samson
// 04 April 2019
// CUDA 2D Array Sum Again
// Homework Part 4

#include "stdio.h"
#define COLUMNS 8 
#define ROWS 8

__global__ void addValue(int *array_val, int *b_array_val) {
	int cacheIndex = threadIdx.x;
	int i = blockDim.x/2;
	while (i > 0) {
		if (cacheIndex < i) {
			array_val[blockIdx.x * COLUMNS +cacheIndex] += array_val[blockIdx.x * COLUMNS + cacheIndex +i];
		}
		__syncthreads();
		i /=2; 
	}
	if (cacheIndex == 0)
		b_array_val[blockIdx.x] = array_val[blockIdx.x * COLUMNS];
		
}

int main() {
	int array_val[ROWS][COLUMNS], b_array_val[COLUMNS];
	int *dev_a;
	int *dev_b;
	int sum =0;
	int cudSum =0;
	cudaMalloc((void **)&dev_a, ROWS*COLUMNS*sizeof(int));
	cudaMalloc((void **)&dev_b, COLUMNS*sizeof(int));
	
	for(int y=0; y<ROWS; y++) {
		for(int x=0; x<COLUMNS; x++){
			array_val[y][x] = x+y;
			sum+= x+y;
		}
	}
	printf("Sum is: %d \n", sum);
	cudaMemcpy(dev_a, array_val, ROWS*COLUMNS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b_array_val, COLUMNS*sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 thread(COLUMNS,ROWS);
	addValue<<<8,8>>>(dev_a,dev_b);
	cudaMemcpy(b_array_val,dev_b, COLUMNS*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i=0; i<ROWS; i++) {
		cudSum+= b_array_val[i];
	} 
	printf("cuda sum is: %d \n", cudSum);

	cudaFree(dev_a);
	cudaFree(dev_b);
	return 0;

}