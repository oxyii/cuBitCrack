#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaDeviceKeys.h"
#include "CudaDeviceKeys.cuh"
#include "Logger.h"
#include "util.h"
#include "secp256k1.cuh"
#include "uint256.h"
#include "heap.cuh"


__global__ void multiplyStepKernel(const uint256 *basePrivateKey, const uint256 *stride, unsigned int pointsPerThread, unsigned int step, uint256 *chain, const uint256 *gxPtr, const uint256 *gyPtr);

//

cudaError_t CudaDeviceKeys::initializeBaseKey(const secp256k1::uint256 &basePrivateKey, const secp256k1::uint256 &stride)
{
	cudaError_t err = cudaMalloc(&_devBasePrivate, sizeof(uint256));
	if (err)
	{
		return err;
	}
	err = cudaMalloc(&_devStride, sizeof(uint256));
	if (err)
	{
		return err;
	}

	// Clear base private key and stride
	err = cudaMemset(_devBasePrivate, 0, sizeof(uint256));
	if (err)
	{
		return err;
	}
	err = cudaMemset(_devStride, 0, sizeof(uint256));
	if (err)
	{
		return err;
	}

	// Copy base private key and stride to device memory (64 bytes total instead of megabytes)
	unsigned int baseKey[8];
	unsigned int strideKey[8];

	basePrivateKey.exportWords(baseKey, 8, secp256k1::uint256::BigEndian);
	stride.exportWords(strideKey, 8, secp256k1::uint256::BigEndian);

	// Copy base private key to device memory
	err = cudaMemcpy(_devBasePrivate, new uint256(baseKey), sizeof(baseKey), cudaMemcpyHostToDevice);
	if (err)
	{
		return err;
	}

	// Copy stride to device memory
	return cudaMemcpy(_devStride, new uint256(strideKey), sizeof(strideKey), cudaMemcpyHostToDevice);
}

cudaError_t CudaDeviceKeys::initializeBasePoints()
{
	// generate a table of points G, 2G, 4G, 8G...(2^255)G
	std::vector<secp256k1::ecpoint> table;

	table.push_back(secp256k1::G());
	for (int i = 1; i < 256; i++)
	{

		secp256k1::ecpoint p = doublePoint(table[i - 1]);
		if (!pointExists(p))
		{
			throw std::string("Point does not exist!");
		}
		table.push_back(p);
	}

	unsigned int count = 256;

	cudaError_t err = cudaMalloc(&_devBasePointX, sizeof(unsigned int) * count * 8);

	if (err)
	{
		return err;
	}

	err = cudaMalloc(&_devBasePointY, sizeof(unsigned int) * count * 8);
	if (err)
	{
		return err;
	}

	unsigned int *tmpX = new unsigned int[count * 8];
	unsigned int *tmpY = new unsigned int[count * 8];

	for (int i = 0; i < 256; i++)
	{
		unsigned int bufX[8];
		unsigned int bufY[8];
		table[i].x.exportWords(bufX, 8, secp256k1::uint256::BigEndian);
		table[i].y.exportWords(bufY, 8, secp256k1::uint256::BigEndian);

		for (int j = 0; j < 8; j++)
		{
			tmpX[i * 8 + j] = bufX[j];
			tmpY[i * 8 + j] = bufY[j];
		}
	}

	err = cudaMemcpy(_devBasePointX, tmpX, count * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);

	delete[] tmpX;

	if (err)
	{
		delete[] tmpY;
		return err;
	}

	err = cudaMemcpy(_devBasePointY, tmpY, count * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);

	delete[] tmpY;

	return err;
}

/**
* Allocates device memory for storing the multiplication chain used in
the batch inversion operation
*/
cudaError_t CudaDeviceKeys::allocateChainBuf(size_t count)
{
	cudaError_t err = cudaMalloc(&_devChain, count * sizeof(uint256));
	if (err)
	{
		return err;
	}

	return err;
}

cudaError_t CudaDeviceKeys::init(unsigned int blocks, unsigned int threads, unsigned int pointsPerThread, const secp256k1::uint256 &basePrivateKey, const secp256k1::uint256 &stride)
{
	_blocks = blocks;
	_threads = threads;
	_pointsPerThread = pointsPerThread;

	cudaError_t err = initializeBaseKey(basePrivateKey, stride);
	if (err)
	{
		return err;
	}

	err = initializeBasePoints();
	if (err)
	{
		return err;
	}

	return allocateChainBuf(_blocks * _threads * _pointsPerThread);
}

void CudaDeviceKeys::clearPrivateKeys()
{
	cudaFree(_devBasePointX);
	cudaFree(_devBasePointY);
	cudaFree(_devBasePrivate);
	cudaFree(_devStride);
	cudaFree(_devChain);

	_devChain = NULL;
	_devBasePointX = NULL;
	_devBasePointY = NULL;
	_devBasePrivate = NULL;
	_devStride = NULL;
}

cudaError_t CudaDeviceKeys::doStep()
{
	multiplyStepKernel<<<_blocks, _threads>>>(_devBasePrivate, _devStride, _pointsPerThread, _step, _devChain, _devBasePointX, _devBasePointY);

	// Wait for kernel to complete
	cudaError_t err = cudaDeviceSynchronize();
	fflush(stdout);
	_step++;
	return err;
}

__global__ void multiplyStepKernel(const uint256 *basePrivateKey, const uint256 *stride, unsigned int pointsPerThread, unsigned int step, uint256 *chain, const uint256 *gxPtr, const uint256 *gyPtr)
{
	heap_buf heapX(ec::getXLowPtr(), ec::getXHighPtr(), nullptr);
	heap_buf heapY(ec::getYLowPtr(), ec::getYHighPtr(), nullptr);
	heap heapC(chain);

	uint256 gx = gxPtr[step];
	uint256 gy = gyPtr[step];

	const uint256 systemBaseKey = *basePrivateKey;
	const uint256 strideValue = *stride;

	// OPTIMIZATION: Allocate computation buffers once outside loops
	uint256 totalOffset, p;

	// Multiply together all (_Gx - x) and then invert
	uint256 inverse = uint256(1);

	int batchIdx = 0;
	for (int i = 0; i < pointsPerThread; i++)
	{
		computeOffset(strideValue, heapX.getShift(i), totalOffset);
		add(systemBaseKey, totalOffset, p);

		unsigned int bit = p[7 - step / 32] & 1 << ((step % 32));

		uint256_buf x = heapX[i];

		if (bit != 0)
		{
			if (!x.isInfinity())
			{
				beginBatchAddWithDouble(gx, gy, x, heapC, i, batchIdx, inverse);
				batchIdx++;
			}
		}
	}

	doBatchInverse(inverse);

	for (int i = pointsPerThread - 1; i >= 0; i--)
	{
		computeOffset(strideValue, heapX.getShift(i), totalOffset);
		add(systemBaseKey, totalOffset, p);

		unsigned int bit = p[7 - step / 32] & 1 << ((step % 32));

		if (bit != 0)
		{
			if (!heapX[i].isInfinity())
			{
				batchIdx--;
				completeBatchAddWithDouble(gx, gy, heapX, heapY, heapC, i, batchIdx, inverse);
			}
			else
			{
				heapX.set(i, gx);
				heapY.set(i, gy);
			}
		}
	}
}
