#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello from GPU thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch 2 blocks, 4 threads each
    helloCUDA<<<2, 4>>>();
    
    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();
    return 0;
}