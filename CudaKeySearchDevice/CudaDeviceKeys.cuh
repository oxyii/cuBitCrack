#ifndef _EC_CUH
#define _EC_CUH

#include <cuda_runtime.h>

namespace ec {
	__device__ uint4 *getXLowPtr();
	__device__ uint4 *getXHighPtr();

	__device__ uint4 *getYLowPtr();
	__device__ uint4 *getYHighPtr();
}

#endif