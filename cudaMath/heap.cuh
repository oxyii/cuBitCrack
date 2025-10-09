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

struct heap_buf
{
    unsigned int _globalThreads;
    unsigned int _globalThreadId;

    uint4 *_readerLow;
    uint4 *_readerHigh;

    uint4 *_buf;

    __device__ heap_buf(uint4 *heapLow, uint4 *heapHigh, uint4 *buf)
    : _readerLow(heapLow), _readerHigh(heapHigh), _buf(buf)
    {
        _globalThreads = gridDim.x * blockDim.x;
        _globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ unsigned int getShift(const unsigned int idx) const
    {
        return idx * _globalThreads + _globalThreadId;
    }

    __device__ uint256_buf operator[](const unsigned int idx)
    {
        return uint256_buf(&_readerLow[getShift(idx)], &_readerHigh[getShift(idx)]);
    }

    __device__ void set(const unsigned int idx, const uint256_buf &val)
    {
        _readerLow[getShift(idx)] = *val.low;
        _readerHigh[getShift(idx)] = *val.high;
    }

    __device__ void set(const unsigned int idx, const uint256 &val)
    {
        _readerLow[getShift(idx)] = val.low;
        _readerHigh[getShift(idx)] = val.high;
    }
};

#endif