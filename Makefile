
CUR_DIR=$(shell pwd)
DIRS=util AddressUtil CmdParse CryptoUtil KeyFinderLib CLKeySearchDevice CudaKeySearchDevice cudaMath clUtil cudaUtil secp256k1lib Logger

INCLUDE = $(foreach d, $(DIRS), -I$(CUR_DIR)/$d)

LIBDIR=$(CUR_DIR)/lib
BINDIR=$(CUR_DIR)/bin
LIBS+=-L$(LIBDIR)

# C++ options
CXX=g++
CXXFLAGS=-O3 -std=c++17 -funroll-loops -ftree-vectorize -finline-functions

# CUDA variables
#COMPUTE_CAP=89
# CUDA variables - OPTIMIZED for RTX 40xx(sm_89) can add/change to 50xx(sm_120) and H100/H200(sm_90)
COMPUTE_CAPS=-gencode=arch=compute_89,code=sm_89 #-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_120,code=sm_120

NVCC=nvcc
#NVCCFLAGS=-std=c++11 -gencode=arch=compute_${COMPUTE_CAP},code=\"sm_${COMPUTE_CAP}\" -Xptxas="-v" -Xcompiler "${CXXFLAGS}"
NVCCFLAGS=-std=c++17 \
          $(COMPUTE_CAPS) \
          --fmad=true \
          -Xptxas -v \
		  -Xptxas -O3 \
		  -Xptxas -warn-double-usage \
          -Xcompiler "${CXXFLAGS}" \
          -rdc=true

CUDA_HOME=/usr/local/cuda
CUDA_LIB=${CUDA_HOME}/lib64
CUDA_INCLUDE=${CUDA_HOME}/include
CUDA_MATH=$(CUR_DIR)/cudaMath

export INCLUDE
export LIBDIR
export BINDIR
export NVCC
export NVCCFLAGS
export LIBS
export CXX
export CXXFLAGS
export CUDA_LIB
export CUDA_INCLUDE
export CUDA_MATH
export COMPUTE_CAPS

TARGETS=dir_addressutil dir_cmdparse dir_cryptoutil dir_keyfinderlib dir_keyfinder dir_secp256k1lib dir_util dir_logger dir_addrgen dir_cudaKeySearchDevice dir_cudautil

all:	${TARGETS}

dir_cudaKeySearchDevice: dir_keyfinderlib dir_cudautil dir_logger
	make --directory CudaKeySearchDevice

dir_addressutil:	dir_util dir_secp256k1lib dir_cryptoutil
	make --directory AddressUtil

dir_cmdparse:
	make --directory CmdParse

dir_cryptoutil:
	make --directory CryptoUtil

dir_keyfinderlib:	dir_util dir_secp256k1lib dir_cryptoutil dir_addressutil dir_logger
	make --directory KeyFinderLib

KEYFINDER_DEPS=dir_keyfinderlib dir_cudaKeySearchDevice

dir_keyfinder:	$(KEYFINDER_DEPS)
	make --directory KeyFinder

dir_cudautil:
	make --directory cudaUtil

dir_secp256k1lib:	dir_cryptoutil
	make --directory secp256k1lib

dir_util:
	make --directory util

dir_cudainfo:
	make --directory cudaInfo

dir_logger:
	make --directory Logger

dir_addrgen:	dir_cmdparse dir_addressutil dir_secp256k1lib
	make --directory AddrGen

clean:
	make --directory AddressUtil clean
	make --directory CmdParse clean
	make --directory CryptoUtil clean
	make --directory KeyFinderLib clean
	make --directory KeyFinder clean
	make --directory cudaUtil clean
	make --directory secp256k1lib clean
	make --directory util clean
	make --directory cudaInfo clean
	make --directory Logger clean
	make --directory CudaKeySearchDevice clean
	rm -rf ${LIBDIR}
	rm -rf ${BINDIR}
