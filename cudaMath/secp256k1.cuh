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

__device__ __forceinline__ static void subModP(const uint256 &a, const uint256 &b, uint256 &c)
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

	if (borrow)
	{
		add_cc(c[7], c[7], _P[7]);
		addc_cc(c[6], c[6], _P[6]);
		addc_cc(c[5], c[5], _P[5]);
		addc_cc(c[4], c[4], _P[4]);
		addc_cc(c[3], c[3], _P[3]);
		addc_cc(c[2], c[2], _P[2]);
		addc_cc(c[1], c[1], _P[1]);
		addc(c[0], c[0], _P[0]);
	}
}

__device__ __forceinline__ static void mulModP(const uint256 &a, const uint256 &b, uint256 &c)
{
	uint256 high = uint256();

	unsigned int t = a[7];

	// a[7] * b (low)
	for (int i = 7; i >= 0; i--)
	{
		c[i] = t * b[i];
	}

	// a[7] * b (high)
	mad_hi_cc(c[6], t, b[7], c[6]);
	madc_hi_cc(c[5], t, b[6], c[5]);
	madc_hi_cc(c[4], t, b[5], c[4]);
	madc_hi_cc(c[3], t, b[4], c[3]);
	madc_hi_cc(c[2], t, b[3], c[2]);
	madc_hi_cc(c[1], t, b[2], c[1]);
	madc_hi_cc(c[0], t, b[1], c[0]);
	madc_hi(high[7], t, b[0], high[7]);

	// a[6] * b (low)
	t = a[6];
	mad_lo_cc(c[6], t, b[7], c[6]);
	madc_lo_cc(c[5], t, b[6], c[5]);
	madc_lo_cc(c[4], t, b[5], c[4]);
	madc_lo_cc(c[3], t, b[4], c[3]);
	madc_lo_cc(c[2], t, b[3], c[2]);
	madc_lo_cc(c[1], t, b[2], c[1]);
	madc_lo_cc(c[0], t, b[1], c[0]);
	madc_lo_cc(high[7], t, b[0], high[7]);
	addc(high[6], high[6], 0);

	// a[6] * b (high)
	mad_hi_cc(c[5], t, b[7], c[5]);
	madc_hi_cc(c[4], t, b[6], c[4]);
	madc_hi_cc(c[3], t, b[5], c[3]);
	madc_hi_cc(c[2], t, b[4], c[2]);
	madc_hi_cc(c[1], t, b[3], c[1]);
	madc_hi_cc(c[0], t, b[2], c[0]);
	madc_hi_cc(high[7], t, b[1], high[7]);
	madc_hi(high[6], t, b[0], high[6]);

	// a[5] * b (low)
	t = a[5];
	mad_lo_cc(c[5], t, b[7], c[5]);
	madc_lo_cc(c[4], t, b[6], c[4]);
	madc_lo_cc(c[3], t, b[5], c[3]);
	madc_lo_cc(c[2], t, b[4], c[2]);
	madc_lo_cc(c[1], t, b[3], c[1]);
	madc_lo_cc(c[0], t, b[2], c[0]);
	madc_lo_cc(high[7], t, b[1], high[7]);
	madc_lo_cc(high[6], t, b[0], high[6]);
	addc(high[5], high[5], 0);

	// a[5] * b (high)
	mad_hi_cc(c[4], t, b[7], c[4]);
	madc_hi_cc(c[3], t, b[6], c[3]);
	madc_hi_cc(c[2], t, b[5], c[2]);
	madc_hi_cc(c[1], t, b[4], c[1]);
	madc_hi_cc(c[0], t, b[3], c[0]);
	madc_hi_cc(high[7], t, b[2], high[7]);
	madc_hi_cc(high[6], t, b[1], high[6]);
	madc_hi(high[5], t, b[0], high[5]);

	// a[4] * b (low)
	t = a[4];
	mad_lo_cc(c[4], t, b[7], c[4]);
	madc_lo_cc(c[3], t, b[6], c[3]);
	madc_lo_cc(c[2], t, b[5], c[2]);
	madc_lo_cc(c[1], t, b[4], c[1]);
	madc_lo_cc(c[0], t, b[3], c[0]);
	madc_lo_cc(high[7], t, b[2], high[7]);
	madc_lo_cc(high[6], t, b[1], high[6]);
	madc_lo_cc(high[5], t, b[0], high[5]);
	addc(high[4], high[4], 0);

	// a[4] * b (high)
	mad_hi_cc(c[3], t, b[7], c[3]);
	madc_hi_cc(c[2], t, b[6], c[2]);
	madc_hi_cc(c[1], t, b[5], c[1]);
	madc_hi_cc(c[0], t, b[4], c[0]);
	madc_hi_cc(high[7], t, b[3], high[7]);
	madc_hi_cc(high[6], t, b[2], high[6]);
	madc_hi_cc(high[5], t, b[1], high[5]);
	madc_hi(high[4], t, b[0], high[4]);

	// a[3] * b (low)
	t = a[3];
	mad_lo_cc(c[3], t, b[7], c[3]);
	madc_lo_cc(c[2], t, b[6], c[2]);
	madc_lo_cc(c[1], t, b[5], c[1]);
	madc_lo_cc(c[0], t, b[4], c[0]);
	madc_lo_cc(high[7], t, b[3], high[7]);
	madc_lo_cc(high[6], t, b[2], high[6]);
	madc_lo_cc(high[5], t, b[1], high[5]);
	madc_lo_cc(high[4], t, b[0], high[4]);
	addc(high[3], high[3], 0);

	// a[3] * b (high)
	mad_hi_cc(c[2], t, b[7], c[2]);
	madc_hi_cc(c[1], t, b[6], c[1]);
	madc_hi_cc(c[0], t, b[5], c[0]);
	madc_hi_cc(high[7], t, b[4], high[7]);
	madc_hi_cc(high[6], t, b[3], high[6]);
	madc_hi_cc(high[5], t, b[2], high[5]);
	madc_hi_cc(high[4], t, b[1], high[4]);
	madc_hi(high[3], t, b[0], high[3]);

	// a[2] * b (low)
	t = a[2];
	mad_lo_cc(c[2], t, b[7], c[2]);
	madc_lo_cc(c[1], t, b[6], c[1]);
	madc_lo_cc(c[0], t, b[5], c[0]);
	madc_lo_cc(high[7], t, b[4], high[7]);
	madc_lo_cc(high[6], t, b[3], high[6]);
	madc_lo_cc(high[5], t, b[2], high[5]);
	madc_lo_cc(high[4], t, b[1], high[4]);
	madc_lo_cc(high[3], t, b[0], high[3]);
	addc(high[2], high[2], 0);

	// a[2] * b (high)
	mad_hi_cc(c[1], t, b[7], c[1]);
	madc_hi_cc(c[0], t, b[6], c[0]);
	madc_hi_cc(high[7], t, b[5], high[7]);
	madc_hi_cc(high[6], t, b[4], high[6]);
	madc_hi_cc(high[5], t, b[3], high[5]);
	madc_hi_cc(high[4], t, b[2], high[4]);
	madc_hi_cc(high[3], t, b[1], high[3]);
	madc_hi(high[2], t, b[0], high[2]);

	// a[1] * b (low)
	t = a[1];
	mad_lo_cc(c[1], t, b[7], c[1]);
	madc_lo_cc(c[0], t, b[6], c[0]);
	madc_lo_cc(high[7], t, b[5], high[7]);
	madc_lo_cc(high[6], t, b[4], high[6]);
	madc_lo_cc(high[5], t, b[3], high[5]);
	madc_lo_cc(high[4], t, b[2], high[4]);
	madc_lo_cc(high[3], t, b[1], high[3]);
	madc_lo_cc(high[2], t, b[0], high[2]);
	addc(high[1], high[1], 0);

	// a[1] * b (high)
	mad_hi_cc(c[0], t, b[7], c[0]);
	madc_hi_cc(high[7], t, b[6], high[7]);
	madc_hi_cc(high[6], t, b[5], high[6]);
	madc_hi_cc(high[5], t, b[4], high[5]);
	madc_hi_cc(high[4], t, b[3], high[4]);
	madc_hi_cc(high[3], t, b[2], high[3]);
	madc_hi_cc(high[2], t, b[1], high[2]);
	madc_hi(high[1], t, b[0], high[1]);

	// a[0] * b (low)
	t = a[0];
	mad_lo_cc(c[0], t, b[7], c[0]);
	madc_lo_cc(high[7], t, b[6], high[7]);
	madc_lo_cc(high[6], t, b[5], high[6]);
	madc_lo_cc(high[5], t, b[4], high[5]);
	madc_lo_cc(high[4], t, b[3], high[4]);
	madc_lo_cc(high[3], t, b[2], high[3]);
	madc_lo_cc(high[2], t, b[1], high[2]);
	madc_lo_cc(high[1], t, b[0], high[1]);
	addc(high[0], high[0], 0);

	// a[0] * b (high)
	mad_hi_cc(high[7], t, b[7], high[7]);
	madc_hi_cc(high[6], t, b[6], high[6]);
	madc_hi_cc(high[5], t, b[5], high[5]);
	madc_hi_cc(high[4], t, b[4], high[4]);
	madc_hi_cc(high[3], t, b[3], high[3]);
	madc_hi_cc(high[2], t, b[2], high[2]);
	madc_hi_cc(high[1], t, b[1], high[1]);
	madc_hi(high[0], t, b[0], high[0]);

	// At this point we have 16 32-bit words representing a 512-bit value
	// high[0 ... 7] and c[0 ... 7]
	const unsigned int s = 977;

	// Store high[6] and high[7] since they will be overwritten
	unsigned int high7 = high[7];
	unsigned int high6 = high[6];

	// Take high 256 bits, multiply by 2^32, add to low 256 bits
	// That is, take high[0 ... 7], shift it left 1 word and add it to c[0 ... 7]
	add_cc(c[6], high[7], c[6]);
	addc_cc(c[5], high[6], c[5]);
	addc_cc(c[4], high[5], c[4]);
	addc_cc(c[3], high[4], c[3]);
	addc_cc(c[2], high[3], c[2]);
	addc_cc(c[1], high[2], c[1]);
	addc_cc(c[0], high[1], c[0]);
	addc_cc(high[7], high[0], 0);
	addc(high[6], 0, 0);

	// Take high 256 bits, multiply by 977, add to low 256 bits
	// That is, take high[0 ... 5], high6, high7, multiply by 977 and add to c[0 ... 7]
	mad_lo_cc(c[7], high7, s, c[7]);
	madc_lo_cc(c[6], high6, s, c[6]);
	madc_lo_cc(c[5], high[5], s, c[5]);
	madc_lo_cc(c[4], high[4], s, c[4]);
	madc_lo_cc(c[3], high[3], s, c[3]);
	madc_lo_cc(c[2], high[2], s, c[2]);
	madc_lo_cc(c[1], high[1], s, c[1]);
	madc_lo_cc(c[0], high[0], s, c[0]);
	addc_cc(high[7], high[7], 0);
	addc(high[6], high[6], 0);

	mad_hi_cc(c[6], high7, s, c[6]);
	madc_hi_cc(c[5], high6, s, c[5]);
	madc_hi_cc(c[4], high[5], s, c[4]);
	madc_hi_cc(c[3], high[4], s, c[3]);
	madc_hi_cc(c[2], high[3], s, c[2]);
	madc_hi_cc(c[1], high[2], s, c[1]);
	madc_hi_cc(c[0], high[1], s, c[0]);
	madc_hi_cc(high[7], high[0], s, high[7]);
	addc(high[6], high[6], 0);

	// Repeat the same steps, but this time we only need to handle high[6] and high[7]
	high7 = high[7];
	high6 = high[6];

	// Take the high 64 bits, multiply by 2^32 and add to the low 256 bits
	add_cc(c[6], high[7], c[6]);
	addc_cc(c[5], high[6], c[5]);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], 0, 0);

	// Take the high 64 bits, multiply by 977 and add to the low 256 bits
	mad_lo_cc(c[7], high7, s, c[7]);
	madc_lo_cc(c[6], high6, s, c[6]);
	addc_cc(c[5], c[5], 0);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], high[7], 0);

	mad_hi_cc(c[6], high7, s, c[6]);
	madc_hi_cc(c[5], high6, s, c[5]);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], high[7], 0);

	bool overflow = high[7] != 0;

	unsigned int borrow = sub(c, _P, c);

	if (overflow)
	{
		if (!borrow)
		{
			sub(c, _P, c);
		}
	}
	else
	{
		if (borrow)
		{
			add(c, _P, c);
		}
	}
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

__device__ static void beginBatchAdd(const uint256 &px, const uint256 &x, heap &heapC, int i, int batchIdx, uint256 &inverse)
{
	uint256 t;

	subModP(px, x, t);

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(t, inverse);

	heapC[batchIdx] = inverse;
}

__device__ static void beginBatchAddWithDouble(const uint256 &px, const uint256 &py, const uint256 &x, heap &heapC, int i, int batchIdx, uint256 &inverse)
{
	uint256 t;

	if (px.equals(x))
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

__device__ static void completeBatchAdd(const uint256 &px, const uint256 &py, heap &heapX, heap &heapY, int i, int batchIdx, heap &heapC, uint256 &inverse)
{
	uint256 rise, s, s2, k, newX, newY, x = heapX[i];

	if (batchIdx >= 1)
	{
		const uint256 c = heapC[batchIdx - 1];
		mulModP(inverse, c, s);

		uint256 diff;
		subModP(px, x, diff);
		mulModP(diff, inverse);
	}
	else
	{
		s = inverse;
	}

	uint256 y = heapY[i];

	subModP(py, y, rise);

	mulModP(rise, s);

	// Rx = s^2 - Gx - Qx
	mulModP(s, s, s2);
	subModP(s2, px, newX);
	subModP(newX, x, newX);

	heapX[i] = newX;

	// Ry = s(px - rx) - py
	subModP(px, newX, k);
	mulModP(s, k, newY);
	subModP(newY, py, newY);

	heapY[i] = newY;
}

__device__ static void completeBatchAddWithDouble(const uint256 &px, const uint256 &py, heap &heapX, heap &heapY, heap &heapC, int i, int batchIdx, uint256 &inverse)
{
	uint256 s, newX, newY, x = heapX[i];

	if (batchIdx >= 1)
	{
		uint256 diff;
		const uint256 c = heapC[batchIdx - 1];

		mulModP(inverse, c, s);

		if (px.equals(x))
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

	if (px.equals(x))
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

		heapX[i] = newX;

		// Ry = s(px - rx) - py
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);

		heapY[i] = newY;
	}
	else
	{
		uint256 rise, y = heapY[i];

		subModP(py, y, rise);

		mulModP(rise, s);

		// Rx = s^2 - Gx - Qx
		mulModP(s, s, s2);

		subModP(s2, px, newX);
		subModP(newX, x, newX);

		heapX[i] = newX;

		// Ry = s(px - rx) - py
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);

		heapY[i] = newY;
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