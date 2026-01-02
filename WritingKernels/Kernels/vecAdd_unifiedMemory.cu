#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>


//The code for a kernel is specified using the __global__ declaration specifier. 
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength) {
    // block dim = num of threads
    // threadIdx = 0 to num of threads
    //blockIdx = 0 to num of blocks
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex >= vectorLength) {
        return;
    }

    C[workIndex] = A[workIndex] + B[workIndex];
}

void UnifiedMemory(int vectorLength) {
    //  the arrays A, B, and C must be in memory accessible to the GPU
    float* A = nullptr;
    float*B = nullptr;
    float*C = nullptr;

    //using unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // 1. CPU INITIALIZATION
    // We fill the arrays with data before sending them to the GPU
    for (int i = 0; i < vectorLength; i++) {
        A[i] = static_cast<float>(i);       // A = 0, 1, 2...
        B[i] = static_cast<float>(i * 2);   // B = 0, 2, 4...
    }

    // Print Inputs (Just the first 5 to keep terminal clean)
    printf("--- Inputs (First 5) ---\n");
    for (int i = 0; i < 5; i++) {
        printf("Index %d: A=%.1f, B=%.1f\n", i, A[i], B[i]);
    }

    // The number of threads that will execute the kernel in parallel is specified as part of the kernel launch.
    // first is the num of blocks, and second is the number of trheads in the blocks
    int threads = 256;
    dim3 blockSize(threads, 1); // 256 threads per block
    int blocks = (vectorLength + threads-1)/threads;// With this code, more threads than needed can be launched without causing out-of-bounds accesses to the arrays. Launching extra threads in a block that do no work does not incur a large overhead cost, however launching thread blocks in which no threads do work should be avoided
    dim3 gridSize(blocks, 1); // 4 blocks
    vecAdd<<<gridSize, blockSize>>>(A,B,C, vectorLength); // the triple chevron <<< >>> encapsulate the execution configuration for the kernel launch

    // wait for kernel to complete execution before the host can proceed
    cudaDeviceSynchronize();

    // 4. PRINT RESULTS
    printf("\n--- Results (First 5) ---\n");
    for (int i = 0; i < 5; i++) {
        printf("Index %d: %.1f + %.1f = %.1f\n", i, A[i], B[i], C[i]);
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main() {
    UnifiedMemory(1024);
}

//nvcc -arch=sm_75 -o vecAdd_unifiedMemory vecAdd_unifiedMemory.cu