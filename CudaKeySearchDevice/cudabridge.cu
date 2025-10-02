#include "cudabridge.h"

void keyFinderKernel(unsigned int blocks, unsigned int threads, unsigned int points, bool useDouble, int compression);

void callKeyFinderKernel(unsigned int blocks, unsigned int threads, unsigned int points, bool useDouble, int compression)
{
    keyFinderKernel(blocks, threads, points, useDouble, compression);
    waitForKernel();
}


void waitForKernel()
{
    // Check for kernel launch error
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        throw cuda::CudaException(err);
    }

    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
	fflush(stdout);
	if(err != cudaSuccess) {
		throw cuda::CudaException(err);
	}
}