#ifndef _SECP256K1_CUH
#define _SECP256K1_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "uint256.h"
#include "heap.cuh"
#include "ptx.cuh"

/**
 Prime modulus 2^256 - 2^32 - 977
 */
__constant__ static unsigned int _P[8] = {
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F};

__device__ __forceinline__ static unsigned int add(const uint256 &a, const uint256 &b, uint256 &c)
{
	add_cc(c[7], a[7], b[7]);
	addc_cc(c[6], a[6], b[6]);
	addc_cc(c[5], a[5], b[5]);
	addc_cc(c[4], a[4], b[4]);
	addc_cc(c[3], a[3], b[3]);
	addc_cc(c[2], a[2], b[2]);
	addc_cc(c[1], a[1], b[1]);
	addc_cc(c[0], a[0], b[0]);

	unsigned int carry = 0;
	addc(carry, 0, 0);

	return carry;
}

__device__ __forceinline__ static unsigned int sub(const uint256 &a, const uint256 &b, uint256 &c)
{
	sub_cc(c[7], a[7], b[7]);
	subc_cc(c[6], a[6], b[6]);
	subc_cc(c[5], a[5], b[5]);
	subc_cc(c[4], a[4], b[4]);
	subc_cc(c[3], a[3], b[3]);
	subc_cc(c[2], a[2], b[2]);
	subc_cc(c[1], a[1], b[1]);
	subc_cc(c[0], a[0], b[0]);

	unsigned int borrow = 0;
	subc(borrow, 0, 0);

	return (borrow & 0x01);
}

__device__ __forceinline__ static void addModP(const uint256 &a, const uint256 &b, uint256 &c)
{
	add_cc(c[7], a[7], b[7]);
	addc_cc(c[6], a[6], b[6]);
	addc_cc(c[5], a[5], b[5]);
	addc_cc(c[4], a[4], b[4]);
	addc_cc(c[3], a[3], b[3]);
	addc_cc(c[2], a[2], b[2]);
	addc_cc(c[1], a[1], b[1]);
	addc_cc(c[0], a[0], b[0]);

	unsigned int carry = 0;
	addc(carry, 0, 0);

	bool gt = false;
	for (int i = 0; i < 8; i++)
	{
		if (c[i] > _P[i])
		{
			gt = true;
			break;
		}
		else if (c[i] < _P[i])
		{
			break;
		}
	}

	if (carry || gt)
	{
		sub_cc(c[7], c[7], _P[7]);
		subc_cc(c[6], c[6], _P[6]);
		subc_cc(c[5], c[5], _P[5]);
		subc_cc(c[4], c[4], _P[4]);
		subc_cc(c[3], c[3], _P[3]);
		subc_cc(c[2], c[2], _P[2]);
		subc_cc(c[1], c[1], _P[1]);
		subc(c[0], c[0], _P[0]);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ static void toInt64(const uint256 &a, uint64_t *out)
{
	out[0] = ((uint64_t)a[6] << 32) | (uint64_t)a[7];
	out[1] = ((uint64_t)a[4] << 32) | (uint64_t)a[5];
	out[2] = ((uint64_t)a[2] << 32) | (uint64_t)a[3];
	out[3] = ((uint64_t)a[0] << 32) | (uint64_t)a[1];
}

__device__ __forceinline__ static void toInt32(const uint64_t *a, uint256 &out)
{
	out[7] = (unsigned int)(a[0] & 0xFFFFFFFFU);
	out[6] = (unsigned int)(a[0] >> 32);
	out[5] = (unsigned int)(a[1] & 0xFFFFFFFFU);
	out[4] = (unsigned int)(a[1] >> 32);
	out[3] = (unsigned int)(a[2] & 0xFFFFFFFFU);
	out[2] = (unsigned int)(a[2] >> 32);
	out[1] = (unsigned int)(a[3] & 0xFFFFFFFFU);
	out[0] = (unsigned int)(a[3] >> 32);
}

__device__ __forceinline__ static void ModSub256(uint64_t *r, uint64_t *a, uint64_t *b)
{

	uint64_t t;
	uint64_t T[4];
	USUBO(r[0], a[0], b[0]);
	USUBC(r[1], a[1], b[1]);
	USUBC(r[2], a[2], b[2]);
	USUBC(r[3], a[3], b[3]);
	USUB(t, 0ULL, 0ULL);
	T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
	T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
	T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
	T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
	UADDO1(r[0], T[0]);
	UADDC1(r[1], T[1]);
	UADDC1(r[2], T[2]);
	UADD1(r[3], T[3]);
}

__device__ __forceinline__ static void subModP(const uint256 &a, const uint256 &b, uint256 &c)
{
	uint64_t A[4], B[4], R[4];

	toInt64(a, A);
	toInt64(b, B);

	ModSub256(R, A, B);

	toInt32(R, c);
}

__device__ __forceinline__ static void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b)
{

	uint64_t r512[8];
	uint64_t t[5];
	uint64_t ah, al;

	r512[5] = 0;
	r512[6] = 0;
	r512[7] = 0;

	// 256*256 multiplier
	UMult(r512, a, b[0]);
	UMult(t, a, b[1]);
	UADDO1(r512[1], t[0]);
	UADDC1(r512[2], t[1]);
	UADDC1(r512[3], t[2]);
	UADDC1(r512[4], t[3]);
	UADD1(r512[5], t[4]);
	UMult(t, a, b[2]);
	UADDO1(r512[2], t[0]);
	UADDC1(r512[3], t[1]);
	UADDC1(r512[4], t[2]);
	UADDC1(r512[5], t[3]);
	UADD1(r512[6], t[4]);
	UMult(t, a, b[3]);
	UADDO1(r512[3], t[0]);
	UADDC1(r512[4], t[1]);
	UADDC1(r512[5], t[2]);
	UADDC1(r512[6], t[3]);
	UADD1(r512[7], t[4]);

	// Reduce from 512 to 320
	UMult(t, (r512 + 4), 0x1000003D1ULL);
	UADDO1(r512[0], t[0]);
	UADDC1(r512[1], t[1]);
	UADDC1(r512[2], t[2]);
	UADDC1(r512[3], t[3]);

	// Reduce from 320 to 256
	UADD1(t[4], 0ULL);
	UMULLO(al, t[4], 0x1000003D1ULL);
	UMULHI(ah, t[4], 0x1000003D1ULL);
	UADDO(r[0], r512[0], al);
	UADDC(r[1], r512[1], ah);
	UADDC(r[2], r512[2], 0ULL);
	UADD(r[3], r512[3], 0ULL);
}

__device__ __forceinline__ static void mulModP(const uint256 &a, const uint256 &b, uint256 &c)
{
	uint64_t A[4], B[4], R[4];

	toInt64(a, A);
	toInt64(b, B);

	_ModMult(R, A, B);

	toInt32(R, c);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ static void subModP(const uint256 &a, const uint256_buf &b, uint256 &c)
{
	uint256 copyB = b.toUint256();
	subModP(a, copyB, c);
}

__device__ __forceinline__ static void mulModP(const uint256_buf &a, const uint256_buf &b, uint256 &c)
{
	uint256 copyA = a.toUint256(), copyB = b.toUint256();
	mulModP(copyA, copyB, c);
}

__device__ __forceinline__ static void mulModP(const uint256 &a, const uint256_buf &b, uint256 &c)
{
	uint256 copyB = b.toUint256();
	mulModP(a, copyB, c);
}

__device__ __forceinline__ static void mulModP(const uint256 &a, uint256 &c)
{
	uint256 tmp;
	mulModP(a, c, tmp);

	c = tmp;
}

__device__ __forceinline__ static void squareModP(const uint256 &a, uint256 &b)
{
	mulModP(a, a, b);
}

__device__ __forceinline__ static void squareModP(uint256 &x)
{
	uint256 tmp;
	squareModP(x, tmp);
	x = tmp;
}

__device__ __forceinline__ static void invModP(uint256 &value)
{
	uint256 x = value;
	uint256 y = uint256(1);

	// 0xd - 1101
	mulModP(x, y);
	squareModP(x);
	// mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);

	// 0x2 - 0010
	// mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	// mulModP(x, y);
	squareModP(x);
	// mulModP(x, y);
	squareModP(x);

	// 0xc = 0x1100
	// mulModP(x, y);
	squareModP(x);
	// mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);

	// 0xfffff
	for (int i = 0; i < 20; i++)
	{
		mulModP(x, y);
		squareModP(x);
	}

	// 0xe - 1110
	// mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);

	// 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffff
	for (int i = 0; i < 219; i++)
	{
		mulModP(x, y);
		squareModP(x);
	}
	mulModP(x, y);

	value = y;
}

__device__ static void beginBatchAdd(const uint256 &px, const uint256 &x, heap_buf &heapC, int i, uint256 &inverse)
{
	uint256 t;

	subModP(px, x, t);

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(t, inverse);

	heapC.set(i, inverse);
}

__device__ static void beginBatchAddWithDouble(const uint256 &px, const uint256 &py, const uint256 &x, heap_buf &heapC, int i, int batchIdx, uint256 &inverse)
{
	uint256 t;

	if (x.equals(px))
		addModP(py, py, t);
	else
		// x = Gx - x
		subModP(px, x, t);

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(t, inverse);

	heapC.set(batchIdx, inverse);
}

__device__ static void beginBatchAddWithDouble(const uint256 &px, const uint256 &py, const uint256 &x, heap &heapC, int i, int batchIdx, uint256 &inverse)
{
	uint256 t;

	if (x.equals(px))
		addModP(py, py, t);
	else
		// x = Gx - x
		subModP(px, x, t);

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(t, inverse);

	heapC[batchIdx] = inverse;
}

__device__ __forceinline__ static void doBatchInverse(uint256 &inverse)
{
	invModP(inverse);
}

__device__ static void completeBatchAdd(const uint256 &px, const uint256 &py, heap_buf &heapX, heap_buf &heapY, int i, heap_buf &heapC, uint256 &inverse)
{
	uint256 rise, s, s2, k, newX, newY;
	uint256_buf x = heapX[i];

	if (i >= 1)
	{
		const uint256_buf c = heapC[i - 1];
		mulModP(inverse, c, s);

		uint256 diff;
		subModP(px, x, diff);
		mulModP(diff, inverse);
	}
	else
	{
		s = inverse;
	}

	uint256_buf y = heapY[i];

	subModP(py, y, rise);

	mulModP(rise, s);

	// Rx = s^2 - Gx - Qx
	mulModP(s, s, s2);
	subModP(s2, px, newX);
	subModP(newX, x, newX);

	heapX.set(i, newX);

	// Ry = s(px - rx) - py
	subModP(px, newX, k);
	mulModP(s, k, newY);
	subModP(newY, py, newY);

	heapY.set(i, newY);
}

__device__ static void completeBatchAddWithDouble(const uint256 &px, const uint256 &py, heap_buf &heapX, heap_buf &heapY, heap_buf &heapC, int i, int batchIdx, uint256 &inverse)
{
	uint256 s, newX, newY;
	uint256_buf x = heapX[i];

	if (batchIdx >= 1)
	{
		uint256 diff;
		const uint256_buf c = heapC[batchIdx - 1];

		mulModP(inverse, c, s);

		if (x.equals(px))
		{
			addModP(py, py, diff);
		}
		else
		{
			subModP(px, x, diff);
		}

		mulModP(diff, inverse);
	}
	else
	{
		s = inverse;
	}

	uint256 s2, k;

	if (x.equals(px))
	{
		// currently s = 1 / 2y

		uint256 x2, tx2;

		// 3x^2
		mulModP(x, x, x2);
		addModP(x2, x2, tx2);
		addModP(x2, tx2, tx2);

		// s = 3x^2 * 1/2y
		mulModP(tx2, s);

		// s^2
		mulModP(s, s, s2);

		// Rx = s^2 - 2px
		subModP(s2, x, newX);
		subModP(newX, x, newX);

		heapX.set(i, newX);

		// Ry = s(px - rx) - py
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);

		heapY.set(i, newY);
	}
	else
	{
		uint256 rise;
		uint256_buf y = heapY[i];

		subModP(py, y, rise);

		mulModP(rise, s);

		// Rx = s^2 - Gx - Qx
		mulModP(s, s, s2);

		subModP(s2, px, newX);
		subModP(newX, x, newX);

		heapX.set(i, newX);

		// Ry = s(px - rx) - py
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);

		heapY.set(i, newY);
	}
}

__device__ static void completeBatchAddWithDouble(const uint256 &px, const uint256 &py, heap_buf &heapX, heap_buf &heapY, heap &heapC, int i, int batchIdx, uint256 &inverse)
{
	uint256 s, newX, newY;
	uint256_buf x = heapX[i];

	if (batchIdx >= 1)
	{
		uint256 diff;
		const uint256 c = heapC[batchIdx - 1];

		mulModP(inverse, c, s);

		if (x.equals(px))
		{
			addModP(py, py, diff);
		}
		else
		{
			subModP(px, x, diff);
		}

		mulModP(diff, inverse);
	}
	else
	{
		s = inverse;
	}

	uint256 s2, k;

	if (x.equals(px))
	{
		// currently s = 1 / 2y

		uint256 x2, tx2;

		// 3x^2
		mulModP(x, x, x2);
		addModP(x2, x2, tx2);
		addModP(x2, tx2, tx2);

		// s = 3x^2 * 1/2y
		mulModP(tx2, s);

		// s^2
		mulModP(s, s, s2);

		// Rx = s^2 - 2px
		subModP(s2, x, newX);
		subModP(newX, x, newX);

		heapX.set(i, newX);

		// Ry = s(px - rx) - py
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);

		heapY.set(i, newY);
	}
	else
	{
		uint256 rise;
		uint256_buf y = heapY[i];

		subModP(py, y, rise);

		mulModP(rise, s);

		// Rx = s^2 - Gx - Qx
		mulModP(s, s, s2);

		subModP(s2, px, newX);
		subModP(newX, x, newX);

		heapX.set(i, newX);

		// Ry = s(px - rx) - py
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);

		heapY.set(i, newY);
	}
}

/**
 * MEMORY OPTIMIZATION: Helper functions for computing private keys on-the-fly
 * Instead of storing millions of private keys in GPU memory, we compute them dynamically
 */

/**
 * Shift a 256-bit number left by specified number of bits
 */
__device__ __forceinline__ static void shiftLeft(const uint256 &src, int bits, uint256 &dst)
{
	dst.clear();

	if (bits == 0)
	{
		dst = src;
		return;
	}

	// Handle large shifts efficiently
	if (bits >= 256)
	{
		return;
	}

	int wordShift = bits / 32; // Number of 32-bit words to shift
	int bitShift = bits % 32;  // Remaining bits to shift within word

	if (bitShift == 0)
	{
		// Word-aligned shift - simple copy
		for (int i = wordShift; i < 8; i++)
		{
			dst[i] = src[i - wordShift];
		}
	}
	else
	{
		// Bit-level shift - need to handle carry between words
		for (int i = wordShift; i < 8; i++)
		{
			if (i - wordShift < 8)
			{
				dst[i] = src[i - wordShift] << bitShift;
			}
			// Handle carry from previous word
			if (i - wordShift - 1 >= 0 && i - wordShift - 1 < 8)
			{
				dst[i] |= src[i - wordShift - 1] >> (32 - bitShift);
			}
		}
	}
}

/**
 * Compute offset for private key calculation: offset = stride * linearIndex
 * This replaces storing millions of precomputed private keys with on-the-fly calculation
 */
__device__ __forceinline__ static void computeOffset(const uint256 &strideValue, unsigned int linearIndex, uint256 &offset)
{
	offset.clear();

	// Special case: index 0 -> offset 0
	if (linearIndex == 0)
	{
		return;
	}

	// Special case: index 1 -> offset = stride
	if (linearIndex == 1)
	{
		offset = strideValue;
		return;
	}

	// OPTIMIZATION: Allocate temp buffers once outside the loop
	uint256 temp, newOffset;

	// Process each bit of linearIndex
	for (int bit = 0; bit < 32 && (linearIndex >> bit); bit++)
	{
		if (linearIndex & (1 << bit))
		{
			// temp = strideValue << bit
			shiftLeft(strideValue, bit, temp);

			// offset += temp (reuse allocated buffers)
			add(offset, temp, newOffset);
			offset = newOffset;
		}
	}
}

#endif