#ifndef _UINT256_H
#define _UINT256_H

#include <cuda_runtime.h>
#include <cstdint>

/**
 * CUDA-optimized 256-bit unsigned integer class
 * Designed specifically for GPU computations
 */
class uint256
{
public:
    // Primary data representation - compatible with CUDA's uint4
    union
    {
        struct
        {
            uint4 low;  // Lower 128 bits (LSB)
            uint4 high; // Upper 128 bits (MSB)
        };
        uint32_t words[8]; // Access as 32-bit words
    };

    // ===== CONSTRUCTORS =====
    __host__ __device__ __forceinline__ uint256() : low{0, 0, 0, 0}, high{0, 0, 0, 0} {}

    __host__ __device__ __forceinline__ uint256(uint32_t val) : low{0, 0, 0, 0}, high{0, 0, 0, val} {}

    __host__ __device__ __forceinline__ uint256(const uint4 &low_val, const uint4 &high_val)
        : low(low_val), high(high_val) {}

    __host__ __device__ __forceinline__ uint256(const uint32_t data[8])
    {
        low.x = data[0];
        low.y = data[1];
        low.z = data[2];
        low.w = data[3];
        high.x = data[4];
        high.y = data[5];
        high.z = data[6];
        high.w = data[7];
    }

    // ===== MEMORY ACCESS OPERATORS =====
    __device__ __forceinline__ uint32_t &operator[](int index)
    {
        return words[index];
    }

    __device__ __forceinline__ uint32_t const &operator[](int index) const
    {
        return words[index];
    }

    // ===== ENDIAN SWAP OPERATIONS =====
    __device__ __forceinline__ static uint32_t swapEndian32(uint32_t x)
    {
        return (x << 24) | ((x << 8) & 0x00ff0000) |
               ((x >> 8) & 0x0000ff00) | (x >> 24);
    }

    __device__ __forceinline__ void swapEndian()
    {
        low.x = swapEndian32(low.x);
        low.y = swapEndian32(low.y);
        low.z = swapEndian32(low.z);
        low.w = swapEndian32(low.w);
        high.x = swapEndian32(high.x);
        high.y = swapEndian32(high.y);
        high.z = swapEndian32(high.z);
        high.w = swapEndian32(high.w);
    }

    // ===== COMPARISON OPERATORS =====
    __device__ __forceinline__ bool operator==(const uint256 &other) const
    {
        return equals(other);
    }

    __device__ __forceinline__ bool operator!=(const uint256 &other) const
    {
        return !(*this == other);
    }

    __device__ __forceinline__ bool equals(const uint256 &other) const
    {
        if (high.x != other.high.x)
            return false;
        if (high.y != other.high.y)
            return false;
        if (high.z != other.high.z)
            return false;
        if (high.w != other.high.w)
            return false;

        // Then compare least significant words
        if (low.x != other.low.x)
            return false;
        if (low.y != other.low.y)
            return false;
        if (low.z != other.low.z)
            return false;
        if (low.w != other.low.w)
            return false;

        return true;
    }

    __device__ __forceinline__ bool isZero() const
    {
        if (high.x != 0)
            return false;
        if (high.y != 0)
            return false;
        if (high.z != 0)
            return false;
        if (high.w != 0)
            return false;

        // Then compare least significant words
        if (low.x != 0)
            return false;
        if (low.y != 0)
            return false;
        if (low.z != 0)
            return false;
        if (low.w != 0)
            return false;

        return true;
    }

    __device__ __forceinline__ bool isInfinity() const
    {
        if (high.x != 0xFFFFFFFFu)
            return false;
        if (high.y != 0xFFFFFFFFu)
            return false;
        if (high.z != 0xFFFFFFFFu)
            return false;
        if (high.w != 0xFFFFFFFFu)
            return false;

        // Then compare least significant words
        if (low.x != 0xFFFFFFFFu)
            return false;
        if (low.y != 0xFFFFFFFFu)
            return false;
        if (low.z != 0xFFFFFFFFu)
            return false;
        if (low.w != 0xFFFFFFFFu)
            return false;

        return true;
    }

    // ===== UTILITY METHODS =====
    __device__ __forceinline__ void clear()
    {
        low = make_uint4(0, 0, 0, 0);
        high = make_uint4(0, 0, 0, 0);
    }

    // ===== STATIC METHODS =====
    __device__ __forceinline__ static uint256 max()
    {
        uint256 result;
        result.low = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
        result.high = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
        return result;
    }
};

#endif