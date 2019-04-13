// Jim Samson
// 04 April 2019
// CUDA 2D Array Sum
// Homework part 3

#include <stdio.h>
#define COLUMNS 4
#define ROWS 3

__global__ void addValue(int * array_val, int*b_array_val) {
    int x = threadIdx.x;
    int sum = 0;

    for(unsigned int i = 0; i < ROWS; i++) {
        sum += array_val[i*COLUMNS+x];
    }
    b_array_val[x] = sum;
}

int main() {
    int array_val[ROWS][COLUMNS], b_array_val[COLUMNS];
    int *dev_a;
    int *dev_b;
    int sum = 0;
    int sum_total = 0; 

    cudaMalloc((void **)&dev_a, ROWS*COLUMNS*sizeof(int));
    cudaMalloc((void **)&dev_b, COLUMNS*sizeof(int));

    for (int y = 0; y< ROWS; y++) {
        for(int x = 0; x < COLUMNS; x++){
            array_val[y][x] = x;
            sum += x;
        }

    printf("Exact sum is: %d \n", sum);
    
    cudaMemcpy(dev_a, array_val, ROWS*COLUMNS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_array_val, COLUMNS*sizeof(int), cudaMemcpyHostToDevice);
    addValue<<<1,COLUMNS>>>(dev_a, dev_b);
    cudaMemcpy(b_array_val, dev_b, COLUMNS*sizeof(int), cudaMemcpyDeviceToHost);
    
    for(unsigned int i = 0; i < COLUMNS; i++){
        sum_total += b_array_val[i];
    }

    printf("The cuda sum is: %d \n", sum_total);

    cudaFree(array_val);
    cudaFree(b_array_val);
}