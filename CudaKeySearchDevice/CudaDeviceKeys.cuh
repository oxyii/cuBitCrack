#ifndef _EC_CUH
#define _EC_CUH

#include <cuda_runtime.h>
#include "uint256.h"

namespace ec {
	__device__ uint256 *getXPtr();

	__device__ uint256 *getYPtr();
}

#endif