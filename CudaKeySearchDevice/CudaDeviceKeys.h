#ifndef _EC_H
#define _EC_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include "secp256k1.h"
#include "uint256.h"

class CudaDeviceKeys
{

private:
	unsigned int _blocks;
	unsigned int _threads;
	unsigned int _pointsPerThread;

	unsigned int _numKeys;

	uint256 *_devBasePrivate;
	uint256 *_devStride;

	uint256 *_devChain;

	uint256 *_devBasePointX;
	uint256 *_devBasePointY;

	unsigned int _step;

	cudaError_t allocateChainBuf(size_t count);

	cudaError_t initializeBaseKey(const secp256k1::uint256 &basePrivateKey, const secp256k1::uint256 &stride);

	cudaError_t initializeBasePoints();

public:
	CudaDeviceKeys()
	{
		_numKeys = 0;
		_devBasePrivate = NULL;
		_devStride = NULL;
		_devChain = NULL;
		_devBasePointX = NULL;
		_devBasePointY = NULL;
		_step = 0;
	}

	~CudaDeviceKeys()
	{
		clearPrivateKeys();
	}

	cudaError_t init(unsigned int blocks, unsigned int threads, unsigned int pointsPerThread, const secp256k1::uint256 &basePrivateKey, const secp256k1::uint256 &stride);

	cudaError_t doStep();

	void clearPrivateKeys();
};

#endif