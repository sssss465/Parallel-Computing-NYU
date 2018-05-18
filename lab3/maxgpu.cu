#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include <time.h>

// Parallel Computing Lab 3
// Author: Andrew Huang

// forward declare
void deviceProperties(void);
long getMax(long * a, long);

#define THREADS_PER_BLOCK 1024 // 3.x

void deviceProperties(void){ // displays device properties
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}
__global__ void getMaxCUDA(long arr[], long size, long result[]){ // Cuda CALL kernal func
    __shared__ long arr_all[THREADS_PER_BLOCK];
    long gid = blockIdx.x * blockDim.x + threadIdx.x;
    arr_all[threadIdx.x] = -INT_MAX;
    if (gid < size){
        arr_all[threadIdx.x] = arr[gid]; // bounds
    }
    __syncthreads();
    for (long s = blockIdx.x/2; s >0; s = s/2){
        __syncthreads();
        if (threadIdx.x < s && gid < size) {
            arr_all[threadIdx.x] = max(arr_all[threadIdx.x], arr_all[threadIdx.x + s]);
        }
    }
    if (threadIdx.x == 0)result[blockIdx.x] = arr_all[0];
}

long getMax(long arr[], long size){ // array and n;
   // safetogo
    long * new_arr; // this is due to overflow
    long * answer;
    long * result;
    long * arr_copy; // we have to make a copy to the device
    long new_size;
    if (size % THREADS_PER_BLOCK != 0) {
        new_size = (size / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK;
    } else {
        new_size = size;
    }
    new_arr = (long *) malloc(sizeof(long) * new_size);
    for (long i = 0; i < new_size;i++){
        if (i < size){
            new_arr[i] = arr[i];
        } else {
            new_arr[i]=0;
        }
    }
    long block_count = new_size / THREADS_PER_BLOCK;
    cudaMalloc((void **) &arr_copy, sizeof(long) *new_size);
    cudaMemcpy((void *) arr_copy, (void *) new_arr, sizeof(long)*new_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &result, sizeof(long) *block_count); // block results
    do {
        block_count = ceil((float)new_size / (float)THREADS_PER_BLOCK);
        getMaxCUDA<<<block_count, THREADS_PER_BLOCK>>>(arr_copy, new_size, result);
        new_size = block_count;
        arr_copy = result;
     } while (block_count > 1);
    answer = (long*) malloc(sizeof(long) * block_count);
    cudaMemcpy((void *)answer, (void *)result, block_count * sizeof(long),cudaMemcpyDeviceToHost);
    long res =answer[0];
    cudaFree(result);
    cudaFree(arr_copy);
    free(new_arr);
    free(answer);
    return res;
}

int main(int argc, char * argv[]){
    if (argc!= 2){
        printf("Usage: maxgpu N\n");
        printf("where N is the size of the array");
        exit(1);
    }
    int n; // number of integers and size
    long *arr; // array
    n = atoi(argv[1]);

    arr = (long *)malloc(sizeof(long) * n);
    if (!arr){
        printf("failed to allocate array\n");
        exit(1);
    }
    srand(time(NULL));
    for (long i = 0; i < n; i ++){
        arr[i] = rand() % n; 
    }
    // cuda time
    //deviceProperties();
    long res = getMax(arr, n);
    printf("The maximum number in the array is: %ld\n", res);
    free(arr);
    return 0;
} 
