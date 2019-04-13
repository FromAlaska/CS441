// Jim Samson
// MPI MinimumFinding

/*
Write array_val C/C++program that uses MPI(not CUDA) with 8 processes.
Process 0 shouldgeneratean array of 8,000,000 random integerswith values ranging from 0 to 99999inclusive. 
Broadcast the array to all other processes.  
Process 0 should find the minimum from the first 1/8 of the array, 
process 1 should find the minimum from the second1/8 of the array, etc.  
After each process has found their minimum, send it back to process 0 which should receive and output the overall minimum.

Have Process 0 also find the minimum by sequentially searching through the entire 8,000,000 values and 
test to make sure the right value was returned from the other processors.
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>


int lowestNumber(int array_val[], int low ,int high);

#define N 8000000


int lowestNumber(int array_val[], int low, int high){
    int lowestNumber = array_val[low];
    for (int index = low; index < high; index++){
        if(array_val[index] < lowestNumber){
            lowestNumber = array_val[index];
        }
    }
    return lowestNumber;
}

int main(int argc, char * argv[]) {
    int *array_val;
    int index;
    int rank, p;
    int tag = 0;
    int testMin = INT_MAX;
    int min = INT_MAX;
    int source;
    MPI_Status status;
 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);


    array_val = (int *) malloc(sizeof(int)*N);

    if (rank == 0){
        for(index = 0; index < N; index++) {
            array_val[index] = rand() % 100000;
            if (testMin > array_val[index]){
                testMin = array_val[index];
            } 
        }
        printf("The minimum value is: %d \n", testMin);
    }

    MPI_Bcast(array_val, N, MPI_INT, 0, MPI_COMM_WORLD);
    int numToMinimize = N / p;
    int low = rank * numToMinimize;
    int high = low + numToMinimize -1;
    int processValue = lowestNumber(array_val, low, high);

    MPI_Barrier(MPI_COMM_WORLD);



    if(rank != 0) {
        MPI_Send(&processValue, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    else if (rank == 0) {
        if (min > processValue){
            min = processValue;
        }

        for (source = 1; source <p; source ++){
            MPI_Recv(&processValue, 1, MPI_INT, source, tag, MPI_COMM_WORLD,&status);
            if(min > processValue){
                min = processValue;
            }
        }

        printf("The MPI value is: %d \n", min);

    }

    free(array_val);
    MPI_Finalize();
    return 0;   
}
