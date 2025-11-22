#include "reduce.h"
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdio.h>

#define N_BLOCK_SIZE 1024
#define N_SM_NUM 216
#define N_SM_SHARED_MEM_SIZE 24576
#define N_SM_SHARED_SIZE N_SM_SHARED_MEM_SIZE / 8 // Use sizeof(double) = 8
#define N_SM_BUF N_SM_SHARED_SIZE / N_BLOCK_SIZE
#define MAXIMUM_THREADS N_SM_NUM * N_BLOCK_SIZE
#define N_MEM_SIZE sizeof(double) * N_SM_NUM

__device__ void warpReduce(volatile double* sdata, int tid) 
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void cudaReduceGeneral(const float* d_data, const size_t size, const size_t executions,
                                 double* devResult)
{
    __shared__ double buffer[N_SM_SHARED_SIZE];
    for (size_t n = 0; n < N_SM_BUF; n++) {
        buffer[threadIdx.x + blockDim.x * n] = 0.0;
        __syncthreads();
    }

    for (size_t m = 0; m < executions; m++) {
        for (size_t n = 0; n < N_SM_BUF; n++) {
            if ( blockIdx.x * N_SM_SHARED_SIZE + N_SM_SHARED_SIZE * gridDim.x * m + blockDim.x * n + threadIdx.x < size) {
                buffer[blockDim.x * n + threadIdx.x] += (double)(d_data[ blockIdx.x * N_SM_SHARED_SIZE + N_SM_SHARED_SIZE * gridDim.x * m + blockDim.x * n + threadIdx.x]); // Cast input to double
            }
            __syncthreads();
        }
    }

    for (size_t n = 1; n < N_SM_BUF; n++) {
        buffer[threadIdx.x] += buffer[blockDim.x * n + threadIdx.x];
        __syncthreads(); 
    }

    for (size_t n = blockDim.x / 2; n > 32; n >>= 1) {
        if (threadIdx.x < n) {
            buffer[threadIdx.x] += buffer[threadIdx.x + n];
        }
        __syncthreads();
    }
    if(threadIdx.x < 32) warpReduce(buffer, threadIdx.x);

    if (threadIdx.x == 0) {
        devResult[blockIdx.x] += buffer[threadIdx.x];
    }
    __syncthreads();
}

float gpuReduce(const float* d_data, const int size)
{
    int executions;
    double *devResult;
    double *hostResult;
    double result = 0.0;

    hostResult = (double*)malloc(N_MEM_SIZE);
    if (hostResult == NULL) {
        printf("NULLPTR\n");
        return 0.0;
    }
    cudaMalloc(&devResult, N_MEM_SIZE);

    memset(hostResult, 0, N_MEM_SIZE);
    cudaMemset(devResult, 0, N_MEM_SIZE);

    executions = (size + ((size_t)MAXIMUM_THREADS * (size_t)N_SM_BUF) - 1) / ((size_t)MAXIMUM_THREADS * (size_t)N_SM_BUF);

    cudaReduceGeneral<<<N_SM_NUM, N_BLOCK_SIZE>>>(d_data, size, executions, devResult);
    cudaMemcpy(hostResult, devResult, N_MEM_SIZE, cudaMemcpyDeviceToHost);

    for (size_t n = 0; n < N_SM_NUM; n++) {
        result += hostResult[n];
    }

    free(hostResult);
    cudaFree(devResult);
    printf("%f\n",result);
    return (float)result;
}