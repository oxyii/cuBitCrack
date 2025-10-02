#ifndef _CUDA_KEY_SEARCH_DEVICE
#define _CUDA_KEY_SEARCH_DEVICE

#include "KeySearchDevice.h"
#include <vector>
#include <cuda_runtime.h>
#include "secp256k1.h"
#include "CudaDeviceKeys.h"
#include "CudaAtomicList.h"
#include "cudaUtil.h"

class CudaHashLookup
{

private:
    unsigned int *_bloomFilterPtr;

    cudaError_t setTargetBloomFilter(const std::vector<struct hash160> &targets);

    cudaError_t setTargetConstantMemory(const std::vector<struct hash160> &targets);

    unsigned int getOptimalBloomFilterBits(double p, size_t n);

    void cleanup();

    void initializeBloomFilter(const std::vector<struct hash160> &targets, unsigned int *filter, unsigned int mask);

    void initializeBloomFilter64(const std::vector<struct hash160> &targets, unsigned int *filter, unsigned long long mask);

public:
    CudaHashLookup()
    {
        _bloomFilterPtr = NULL;
    }

    ~CudaHashLookup()
    {
        cleanup();
    }

    cudaError_t setTargets(const std::vector<struct hash160> &targets);
};

// Structures that exist on both host and device side
struct CudaDeviceResult {
    unsigned int thread;
    unsigned int block;
    unsigned int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
};

class CudaKeySearchDevice : public KeySearchDevice {

private:

    int _device;

    unsigned int _blocks;

    unsigned int _threads;

    unsigned int _pointsPerThread;

    int _compression;

    std::vector<KeySearchResult> _results;

    std::string _deviceName;

    secp256k1::uint256 _startExponent;

    uint64_t _iterations;

    void cudaCall(cudaError_t err);

    void generateStartingPoints();

    CudaDeviceKeys _deviceKeys;

    CudaAtomicList _resultList;

    CudaHashLookup _targetLookup;

    void getResultsInternal();

    std::vector<hash160> _targets;

    bool isTargetInList(const unsigned int hash[5]);

    void removeTargetFromList(const unsigned int hash[5]);

    uint32_t getPrivateKeyOffset(unsigned int thread, unsigned int block, unsigned int point);

    secp256k1::uint256 _stride;

    bool verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5], bool compressed);

public:
    CudaKeySearchDevice(int device, unsigned int threads, unsigned int pointsPerThread, unsigned int blocks = 0);

    ~CudaKeySearchDevice()
    {
        clearPublicKeys();
    }

    virtual void init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride);

    virtual void doStep();

    virtual cudaError_t initializePublicKeys(size_t count);

    virtual void clearPublicKeys();

    virtual void setTargets(const std::set<KeySearchTarget> &targets);

    virtual size_t getResults(std::vector<KeySearchResult> &results);

    virtual uint64_t keysPerStep();

    virtual std::string getDeviceName();

    virtual void getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem);

    virtual secp256k1::uint256 getNextKey();
};

#endif