#ifndef _HEAP_CUH
#define _HEAP_CUH

#include <cuda_runtime.h>

#include "uint256.h"

struct heap
{
    unsigned int _globalThreads;
    unsigned int _globalThreadId;

    uint256 *_reader;

    __device__ heap(uint256 *heap) : _reader(heap)
    {
        _globalThreads = gridDim.x * blockDim.x;
        _globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ unsigned int getShift(const unsigned int idx) const
    {
        return idx * _globalThreads + _globalThreadId;
    }

    __device__ uint256 &operator[](const unsigned int idx)
    {
        return _reader[getShift(idx)];
    }
};

#endif