/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


// This radix sort implementation was developed by Nadathur Satish, 
// Mark Harris, and Michael Garland at NVIDIA.
//
// Refer to the following paper for further details:
//
//   "Designing efficient sorting algorithms for manycore GPUs"
//   Nadathur Satish, Mark Harris, and Michael Garland,
//   NVIDIA Technical Report NVR-2008-001, September 2008
//
//   http://www.nvidia.com/object/nvidia_research_pub_002.html


#include <thrust/detail/config.h>

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <cassert>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/detail/device/cuda/synchronize.h>

#include <thrust/detail/raw_buffer.h>

#include <thrust/detail/util/align.h>

#include "stable_radix_sort_util.h"


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN


namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
{

inline
void checkCudaError(const char *msg)
{
#if defined(_DEBUG) || defined(DEBUG)
    cudaError_t e = cudaThreadSynchronize();
    if( e != cudaSuccess )
    {
        fprintf(stderr, "CUDA Error %s : %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
    e = cudaGetLastError();
    if( e != cudaSuccess )
    {
        fprintf(stderr, "CUDA Error %s : %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
#endif
}

namespace RadixSort
{
    const unsigned int cta_size  = 256;
    const unsigned int warp_size = 32;
}


#ifdef __DEVICE_EMULATION__
#define __SYNC __syncthreads();
#else
#define __SYNC
#endif

typedef unsigned int uint;

//----------------------------------------------------------------------------
// scan4 scans 4*RadixSort::cta_size numElements in a block (4 per thread)
//----------------------------------------------------------------------------
template <typename T>
__device__ uint4 scan4(T idata)  //T = uint4
{    
    extern  __shared__  uint ptr[];

    // sum of thread's 4 values 
    uint val = idata.x + idata.y + idata.z + idata.w;

    uint idx = threadIdx.x;

    // padding to avoid conditionals in the scan
    ptr[threadIdx.x] = 0; 
    
    idx += RadixSort::cta_size;
    
    ptr[idx] = val;

    __syncthreads();
    
    val += ptr[idx -   1]; __syncthreads(); ptr[idx] = val; __syncthreads();  
    val += ptr[idx -   2]; __syncthreads(); ptr[idx] = val; __syncthreads(); 
    val += ptr[idx -   4]; __syncthreads(); ptr[idx] = val; __syncthreads();
    val += ptr[idx -   8]; __syncthreads(); ptr[idx] = val; __syncthreads();
    val += ptr[idx -  16]; __syncthreads(); ptr[idx] = val; __syncthreads();
    val += ptr[idx -  32]; __syncthreads(); ptr[idx] = val; __syncthreads();
    val += ptr[idx -  64]; __syncthreads(); ptr[idx] = val; __syncthreads();
    val += ptr[idx - 128]; __syncthreads(); ptr[idx] = val; __syncthreads();

    // safe because array is padded with 0s to the left
    val = ptr[idx - 1]; 

    return make_uint4(val, 
                      val + idata.x,
                      val + idata.x + idata.y,
                      val + idata.x + idata.y + idata.z);
}

//----------------------------------------------------------------------------
//
// Rank is the core of the radix sort loop.  Given a predicate, it
// computes the output position for each thread in an ordering where all
// True threads come first, followed by all False threads.
// 
// This version handles 4 predicates per thread; hence, "rank4".
//
//----------------------------------------------------------------------------
template <int ctasize>
__device__ uint4 rank4(uint4 preds)
{
    uint4 address = scan4(preds);  

    __shared__ uint numtrue;
    if (threadIdx.x == ctasize-1)
    {
        numtrue = address.w + preds.w;
    }
    __syncthreads();

    uint4 rank;
    uint idx = threadIdx.x << 2;
    rank.x = (preds.x) ? address.x : numtrue + idx   - address.x;
    rank.y = (preds.y) ? address.y : numtrue + idx + 1 - address.y;
    rank.z = (preds.z) ? address.z : numtrue + idx + 2 - address.z;
    rank.w = (preds.w) ? address.w : numtrue + idx + 3 - address.w;	

    return rank;
}

//----------------------------------------------------------------------------
// Uses rank to sort one bit at a time: Sorts a block according
// to bits startbit -> nbits + startbit
//----------------------------------------------------------------------------
template<uint nbits, uint startbit>
__device__ void radixSortBlock(uint4 &key, uint4 &value)
{
    extern __shared__ uint sMem1[];

    for(uint shift = startbit; shift < (startbit + nbits); ++shift)
    {        
        uint4 lsb;
        lsb.x = !((key.x >> shift) & 0x1);
        lsb.y = !((key.y >> shift) & 0x1);
        lsb.z = !((key.z >> shift) & 0x1);
        lsb.w = !((key.w >> shift) & 0x1);

        uint4 r = rank4<RadixSort::cta_size>(lsb);

#if 1
        // This arithmetic strides the ranks across 4 cta_size regions
        sMem1[(r.x & 3) * RadixSort::cta_size + (r.x >> 2)] = key.x;
        sMem1[(r.y & 3) * RadixSort::cta_size + (r.y >> 2)] = key.y;
        sMem1[(r.z & 3) * RadixSort::cta_size + (r.z >> 2)] = key.z;
        sMem1[(r.w & 3) * RadixSort::cta_size + (r.w >> 2)] = key.w;
        __syncthreads();

        // The above allows us to read without 4-way bank conflicts:
        key.x = sMem1[threadIdx.x];
        key.y = sMem1[threadIdx.x +     RadixSort::cta_size];
        key.z = sMem1[threadIdx.x + 2 * RadixSort::cta_size];
        key.w = sMem1[threadIdx.x + 3 * RadixSort::cta_size];

        __syncthreads();

        sMem1[(r.x & 3) * RadixSort::cta_size + (r.x >> 2)] = value.x;
        sMem1[(r.y & 3) * RadixSort::cta_size + (r.y >> 2)] = value.y;
        sMem1[(r.z & 3) * RadixSort::cta_size + (r.z >> 2)] = value.z;
        sMem1[(r.w & 3) * RadixSort::cta_size + (r.w >> 2)] = value.w;
        __syncthreads();

        value.x = sMem1[threadIdx.x];
        value.y = sMem1[threadIdx.x +     RadixSort::cta_size];
        value.z = sMem1[threadIdx.x + 2 * RadixSort::cta_size];
        value.w = sMem1[threadIdx.x + 3 * RadixSort::cta_size];
#else
        sMem1[r.x] = key.x;
        sMem1[r.y] = key.y;
        sMem1[r.z] = key.z;
        sMem1[r.w] = key.w;
        __syncthreads();

        // This access has 4-way bank conflicts
        key = sMem[threadIdx.x];

        __syncthreads();

        sMem1[r.x] = value.x;
        sMem1[r.y] = value.y;
        sMem1[r.z] = value.z;
        sMem1[r.w] = value.w;
        __syncthreads();

        value = sMem[threadIdx.x];
#endif

        __syncthreads();
    }
}

//----------------------------------------------------------------------------
//
// radixSortBlocks sorts each block of data independently in shared
// memory.  
//
// Done in two separate stages.  This stage calls radixSortBlock on each block 
// independently, sorting on the basis of bits (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------
template<uint nbits, uint startbit, bool fullBlocks, class PreProcess>
__global__ void radixSortBlocks(uint4* keysOut, uint4* valuesOut, 
                                uint4* keysIn, uint4* valuesIn, 
                                uint numElements, uint startBlock,
                                PreProcess preprocess)
{
    extern __shared__ uint4 sMem[];

    uint4 key, value;

    const uint blockId = blockIdx.x + startBlock;
    const uint i = blockId * blockDim.x + threadIdx.x;
    const uint idx = i << 2;

    // handle non-full last block if array is not multiple of 1024 numElements
    if (!fullBlocks && idx+3 >= numElements)
    {
        if (idx >= numElements)
        {
            key   = make_uint4(UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX);
            value = make_uint4(UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX);
        }
        else
        {
            // for non-full block, we handle uint1 values instead of uint4
            uint *keys1    = (uint*)keysIn;
            uint *values1  = (uint*)valuesIn;

            key.x = (idx   < numElements) ? preprocess(keys1[idx])   : UINT_MAX;
            key.y = (idx+1 < numElements) ? preprocess(keys1[idx+1]) : UINT_MAX;
            key.z = (idx+2 < numElements) ? preprocess(keys1[idx+2]) : UINT_MAX;
            key.w = UINT_MAX;

            value.x = (idx   < numElements) ? values1[idx]   : UINT_MAX;
            value.y = (idx+1 < numElements) ? values1[idx+1] : UINT_MAX;
            value.z = (idx+2 < numElements) ? values1[idx+2] : UINT_MAX;
            value.w = UINT_MAX;
        }
    }
    else
    {
        key = keysIn[i];
        value = valuesIn[i];

        key.x = preprocess(key.x);
        key.y = preprocess(key.y);
        key.z = preprocess(key.z);
        key.w = preprocess(key.w);
    }

    __syncthreads();
    radixSortBlock<nbits, startbit>(key, value);
    //__syncthreads();  // IS THIS NECESSARY?

    // handle non-full last block if array is not multiple of 1024 numElements
    if(!fullBlocks && idx+3 >= numElements)
    {
        if (idx < numElements) 
        {
            // for non-full block, we handle uint1 values instead of uint4
            uint *keys1   = (uint*)keysOut;
            uint *values1 = (uint*)valuesOut;

            keys1[idx]   = key.x;
            values1[idx] = value.x;

            if (idx + 1 < numElements)
            {
                keys1[idx + 1]   = key.y;
                values1[idx + 1] = value.y;

                if (idx + 2 < numElements)
                {
                    keys1[idx + 2]   = key.z;
                    values1[idx + 2] = value.z;
                }
            }
        }
    }
    else
    {
        keysOut[i]   = key;
        valuesOut[i] = value;
    }
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each 
// block counts the number of keys that fall into each radix in the group, and 
// finds the starting offset of each radix in the block.  Writes the radix 
// counts to counters, and the starting offsets to blockOffsets.
//----------------------------------------------------------------------------
template<uint startbit, bool fullBlocks>
__global__ void findRadixOffsets(uint2 *keys, 
                                 uint  *counters, 
                                 uint  *blockOffsets, 
                                 uint   numElements,
                                 uint   totalBlocks,
                                 uint   startBlock)
{
    extern __shared__ uint2 sMem2[];

    uint2 *sRadix2         = (uint2*)sMem2;
    uint  *sRadix1         = (uint*) sRadix2; 
    uint  *sStartPointers  = (uint*)(sMem2 + RadixSort::cta_size);

    uint blockId = blockIdx.x + startBlock;
    const uint i = blockId * blockDim.x + threadIdx.x;

    uint2 radix2;

    // handle non-full last block if array is not multiple of 1024 numElements
    if(!fullBlocks && ((i + 1) << 1 ) > numElements )
    {
        // handle uint1 rather than uint2 for non-full blocks
        uint *keys1 = (uint*)keys;
        uint j = i << 1; 

        radix2.x = (j < numElements) ? keys1[j] : UINT_MAX; 
        j++;
        radix2.y = (j < numElements) ? keys1[j] : UINT_MAX;
    }
    else
    {
        radix2 = keys[i];
    }

    sRadix1[2 * threadIdx.x]     = (radix2.x >> startbit) & 0xF;
    sRadix1[2 * threadIdx.x + 1] = (radix2.y >> startbit) & 0xF;

    // Finds the position where the sRadix1 entries differ and stores start 
    // index for each radix.
    if(threadIdx.x < 16) 
    { 
        sStartPointers[threadIdx.x] = 0; 
    }
    __syncthreads();

    if((threadIdx.x > 0) && (sRadix1[threadIdx.x] != sRadix1[threadIdx.x - 1]) ) 
    {
        sStartPointers[sRadix1[threadIdx.x]] = threadIdx.x;
    }
    if(sRadix1[threadIdx.x + RadixSort::cta_size] != sRadix1[threadIdx.x + RadixSort::cta_size - 1]) 
    {
        sStartPointers[sRadix1[threadIdx.x + RadixSort::cta_size]] = threadIdx.x + RadixSort::cta_size;
    }
    __syncthreads();

    if(threadIdx.x < 16) 
    {
        blockOffsets[blockId*16 + threadIdx.x] = sStartPointers[threadIdx.x];
    }
    __syncthreads();

    // Compute the sizes of each block.
    if((threadIdx.x > 0) && (sRadix1[threadIdx.x] != sRadix1[threadIdx.x - 1]) ) 
    {
        sStartPointers[sRadix1[threadIdx.x - 1]] = 
            threadIdx.x - sStartPointers[sRadix1[threadIdx.x - 1]];
    }
    if(sRadix1[threadIdx.x + RadixSort::cta_size] != sRadix1[threadIdx.x + RadixSort::cta_size - 1] ) 
    {
        sStartPointers[sRadix1[threadIdx.x + RadixSort::cta_size - 1]] = 
            threadIdx.x + RadixSort::cta_size - sStartPointers[sRadix1[threadIdx.x + RadixSort::cta_size - 1]];
    }

    if(threadIdx.x == RadixSort::cta_size - 1) 
    {
        sStartPointers[sRadix1[2 * RadixSort::cta_size - 1]] = 
            2 * RadixSort::cta_size - sStartPointers[sRadix1[2 * RadixSort::cta_size - 1]];
    }
    __syncthreads();
    
    if(threadIdx.x < 16) 
    {
        counters[threadIdx.x * totalBlocks + blockId] = 
            sStartPointers[threadIdx.x];
    }
}


//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets 
// have been found. Depends on RadixSort::cta_size being 16 * number of radices 
// (i.e. 16 * 2^nbits).
// 
// This is quite fast and fully coalesces memory writes, albeit by doing extra 
// (potentially wasted) work allocating threads to portions of memory that are 
// not written out. Significantly faster than the generic approach on G80.
//----------------------------------------------------------------------------
template<uint startbit, bool fullBlocks, bool manualCoalesce, class PostProcess>
__global__ void reorderData(uint  *outKeys, 
                            uint  *outValues, 
                            uint2 *keys, 
                            uint2 *values, 
                            uint  *blockOffsets, 
                            uint  *offsets, 
                            uint  *sizes, 
                            uint   numElements,
                            uint   totalBlocks,
                            uint   startBlock,
                            PostProcess postprocess)
{
    __shared__ uint2 sKeys2[RadixSort::cta_size];
    __shared__ uint2 sValues2[RadixSort::cta_size];
    __shared__ uint sOffsets[16];
    __shared__ uint sBlockOffsets[16];

    uint *sKeys1   = (uint*)sKeys2; 
    uint *sValues1 = (uint*)sValues2; 

    const uint blockId = blockIdx.x + startBlock;
    const uint i = blockId * blockDim.x + threadIdx.x;

    // handle non-full last block if array is not multiple of 1024 numElements
    if(!fullBlocks && (((i + 1) << 1) > numElements))
    {
        uint *keys1   = (uint*)keys;
        uint *values1 = (uint*)values;
        uint j = i << 1; 

        sKeys1[threadIdx.x << 1]   = (j < numElements) ? keys1[j]   : UINT_MAX; 
        sValues1[threadIdx.x << 1] = (j < numElements) ? values1[j] : UINT_MAX; 
        j++; 
        sKeys1[(threadIdx.x << 1) + 1]   = (j < numElements) ? keys1[j]   : UINT_MAX; 
        sValues1[(threadIdx.x << 1) + 1] = (j < numElements) ? values1[j] : UINT_MAX; 
    }
    else
    {
        sKeys2[threadIdx.x]   = keys[i];
        sValues2[threadIdx.x] = values[i];
    }

    if (!manualCoalesce)
    {
        if(threadIdx.x < 16)  
        {
            sOffsets[threadIdx.x]      = offsets[threadIdx.x * totalBlocks + blockId];
            sBlockOffsets[threadIdx.x] = blockOffsets[blockId * 16 + threadIdx.x];
        }
        __syncthreads();

        uint radix = (sKeys1[threadIdx.x] >> startbit) & 0xF;
	    uint globalOffset = sOffsets[radix] + threadIdx.x - sBlockOffsets[radix];
	    
        if (fullBlocks || globalOffset < numElements)
        {
	        outKeys[globalOffset]   = postprocess(sKeys1[threadIdx.x]);
	        outValues[globalOffset] = sValues1[threadIdx.x];
        }

        radix = (sKeys1[threadIdx.x + RadixSort::cta_size] >> startbit) & 0xF;
	    globalOffset = sOffsets[radix] + threadIdx.x + RadixSort::cta_size - sBlockOffsets[radix];
	    
        if (fullBlocks || globalOffset < numElements)
        {
	        outKeys[globalOffset]   = postprocess(sKeys1[threadIdx.x + RadixSort::cta_size]);
	        outValues[globalOffset] = sValues1[threadIdx.x + RadixSort::cta_size];
        }
    }
    else
    {
        __shared__ uint sSizes[16];

        if(threadIdx.x < 16)  
        {
            sOffsets[threadIdx.x]      = offsets[threadIdx.x * totalBlocks + blockId];
            sBlockOffsets[threadIdx.x] = blockOffsets[blockId * 16 + threadIdx.x];
            sSizes[threadIdx.x]        = sizes[threadIdx.x * totalBlocks + blockId];
        }
        __syncthreads();

        // 1 half-warp is responsible for writing out all values for 1 radix. 
        // Loops if there are more than 16 values to be written out. 
        // All start indices are rounded down to the nearest multiple of 16, and
        // all end indices are rounded up to the nearest multiple of 16.
        // Thus it can do extra work if the start and end indices are not multiples of 16
        // This is bounded by a factor of 2 (it can do 2X more work at most).

        const uint halfWarpID     = threadIdx.x >> 4;

        const uint halfWarpOffset = threadIdx.x & 0xF;
        const uint leadingInvalid = sOffsets[halfWarpID] & 0xF;

        uint startPos = sOffsets[halfWarpID] & 0xFFFFFFF0;
        uint endPos   = (sOffsets[halfWarpID] + sSizes[halfWarpID]) + 15 - 
                        ((sOffsets[halfWarpID] + sSizes[halfWarpID] - 1) & 0xF);
        uint numIterations = endPos - startPos;

        uint outOffset = startPos + halfWarpOffset;
        uint inOffset  = sBlockOffsets[halfWarpID] - leadingInvalid + halfWarpOffset;

        for(uint j = 0; j < numIterations; j += 16, outOffset += 16, inOffset += 16)
        {       
            if( (outOffset >= sOffsets[halfWarpID]) && 
                (inOffset - sBlockOffsets[halfWarpID] < sSizes[halfWarpID])) 
            {
                if(blockId < totalBlocks - 1 || outOffset < numElements) 
                {
                    outKeys[outOffset]   = postprocess(sKeys1[inOffset]);
                    outValues[outOffset] = sValues1[inOffset];
                }
            }       
        }
    }
}

//----------------------------------------------------------------------------
// Perform one step of the radix sort.  Sorts by nbits key bits per step, 
// starting at startbit.
//----------------------------------------------------------------------------
template<uint nbits, uint startbit, class PreProcess, class PostProcess>
void radixSortStep(uint *keys, 
                   uint *values, 
                   uint *tempKeys, 
                   uint *tempValues, 
                   uint *counters, 
                   uint *countersSum, 
                   uint *blockOffsets, 
                   uint numElements,
                   bool manualCoalesce,
                   PreProcess  preprocess,
                   PostProcess postprocess)
{
    const uint eltsPerBlock  = RadixSort::cta_size * 4;
    const uint eltsPerBlock2 = RadixSort::cta_size * 2;

    bool fullBlocks = ((numElements % eltsPerBlock) == 0);
    uint numBlocks = (fullBlocks) ? 
        (numElements / eltsPerBlock) : 
        (numElements / eltsPerBlock + 1);
    uint numBlocks2 = ((numElements % eltsPerBlock2) == 0) ?
        (numElements / eltsPerBlock2) : 
        (numElements / eltsPerBlock2 + 1);

    const uint max1DBlocks = 65535;

    
    for (uint block = 0; block < numBlocks; block += max1DBlocks)
    {
        uint blocks   = min(max1DBlocks, numBlocks - block);
        
        if (blocks < max1DBlocks && !fullBlocks)
        {
            radixSortBlocks<nbits, startbit, false>
                <<<blocks, RadixSort::cta_size, 4 * RadixSort::cta_size * sizeof(uint)>>>
                    ((uint4*)tempKeys, (uint4*)tempValues, (uint4*)keys, (uint4*)values, numElements, block, preprocess);
            synchronize_if_enabled("radixSortBlocks");
        }
        else
        {
            radixSortBlocks<nbits, startbit, true>
                <<<blocks, RadixSort::cta_size, 4 * RadixSort::cta_size * sizeof(uint)>>>
                    ((uint4*)tempKeys, (uint4*)tempValues, (uint4*)keys, (uint4*)values, numElements, block, preprocess);
            synchronize_if_enabled("radixSortBlocks");
        }
    }

    for (uint block = 0; block < numBlocks2; block += max1DBlocks)
    {
        uint blocks   = min(max1DBlocks, numBlocks2 - block);

        if (blocks < max1DBlocks && !fullBlocks)
        {
            findRadixOffsets<startbit, false>
                <<<blocks, RadixSort::cta_size, 3 * RadixSort::cta_size * sizeof(uint)>>>
                    ((uint2*)tempKeys, counters, blockOffsets, numElements, numBlocks2, block);
            synchronize_if_enabled("findRadixOffsets");
        }
        else
        {
            findRadixOffsets<startbit, true>
                <<<blocks, RadixSort::cta_size, 3 * RadixSort::cta_size * sizeof(uint)>>>
                    ((uint2*)tempKeys, counters, blockOffsets, numElements, numBlocks2, block);
            synchronize_if_enabled("findRadixOffsets");
        }
    }

    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(counters), 
                            thrust::device_ptr<unsigned int>(counters + 16 * numBlocks2),
                            thrust::device_ptr<unsigned int>(countersSum));
                            
    for (uint block = 0; block < numBlocks2; block += max1DBlocks)
    {
        uint blocks   = min(max1DBlocks, numBlocks2 - block);
        
        if (blocks < max1DBlocks && !fullBlocks)
        {
            if (manualCoalesce)
            {
                reorderData<startbit, false, true><<<blocks, RadixSort::cta_size>>>
                    (keys, values, (uint2*)tempKeys, (uint2*)tempValues, 
                     blockOffsets, countersSum, counters, numElements, numBlocks2, block, postprocess);
                synchronize_if_enabled("reorderData");
            }
            else
            {
                reorderData<startbit, false, false><<<blocks, RadixSort::cta_size>>>
                    (keys, values, (uint2*)tempKeys, (uint2*)tempValues, 
                     blockOffsets, countersSum, counters, numElements, numBlocks2, block, postprocess);
                synchronize_if_enabled("reorderData");
            }
        }
        else
        {
            if (manualCoalesce)
            {
                reorderData<startbit, true, true><<<blocks, RadixSort::cta_size>>>
                    (keys, values, (uint2*)tempKeys, (uint2*)tempValues, 
                     blockOffsets, countersSum, counters, numElements, numBlocks2, block, postprocess);
                synchronize_if_enabled("reorderData");
            }
            else
            {
                reorderData<startbit, true, false><<<blocks, RadixSort::cta_size>>>
                    (keys, values, (uint2*)tempKeys, (uint2*)tempValues, 
                     blockOffsets, countersSum, counters, numElements, numBlocks2, block, postprocess);
                synchronize_if_enabled("reorderData");
            }
        }
    }    

    checkCudaError("radixSortStep");
}


//----------------------------------------------------------------------------
// Main radix sort function.  Sorts in place in the keys and values arrays,
// but uses the other device arrays as temporary storage.  All pointer 
// parameters are device pointers.  Uses exclusive_scan() for the prefix 
// sum of radix counters.
//----------------------------------------------------------------------------
template <class PreProcess, class PostProcess>
void radixSort(uint *keys, 
               uint *values, 
               uint *tempKeys, 
               uint *tempValues,
               uint *counters,
               uint *countersSum,
               uint *blockOffsets,
               uint numElements, 
               uint keyBits,
               bool manualCoalesce,
               PreProcess  preprocess,
               PostProcess postprocess)
{
#define RS_KeyValue(bit,pre)                                                                  \
    if (bit + 4 < keyBits)                                                                    \
        radixSortStep<4,(bit)>(keys, values, tempKeys, tempValues,                            \
                               counters, countersSum, blockOffsets,                           \
                               numElements, manualCoalesce,                                   \
                               pre,                                                           \
                               thrust::identity<uint>());                                     \
    else if (bit < keyBits)                                                                   \
        radixSortStep<4,(bit)>(keys, values, tempKeys, tempValues,                            \
                               counters, countersSum, blockOffsets,                           \
                               numElements, manualCoalesce,                                   \
                               pre,                                                           \
                               postprocess);

    RS_KeyValue( 0, preprocess);
    RS_KeyValue( 4, thrust::identity<uint>());
    RS_KeyValue( 8, thrust::identity<uint>());
    RS_KeyValue(12, thrust::identity<uint>());
    RS_KeyValue(16, thrust::identity<uint>());
    RS_KeyValue(20, thrust::identity<uint>());
    RS_KeyValue(24, thrust::identity<uint>());
    RS_KeyValue(28, thrust::identity<uint>());

#undef RS_KeyValue

    checkCudaError("radixSort");
}


//----------------------------------------------------------------------------
// Key-only Sorts
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Uses rank to sort one bit at a time: Sorts a block according
// to bits startbit -> nbits + startbit
//----------------------------------------------------------------------------
template<uint nbits, uint startbit>
__device__ void radixSortBlockKeysOnly(uint4 &key)
{
    extern __shared__ uint sMem1[];

    for(uint shift = startbit; shift < (startbit + nbits); ++shift)
    {        
        uint4 lsb;
        lsb.x = !((key.x >> shift) & 0x1);
        lsb.y = !((key.y >> shift) & 0x1);
        lsb.z = !((key.z >> shift) & 0x1);
        lsb.w = !((key.w >> shift) & 0x1);

        uint4 r = rank4<256>(lsb);

#if 1
        // This arithmetic strides the ranks across 4 cta_size regions
        sMem1[(r.x & 3) * RadixSort::cta_size + (r.x >> 2)] = key.x;
        sMem1[(r.y & 3) * RadixSort::cta_size + (r.y >> 2)] = key.y;
        sMem1[(r.z & 3) * RadixSort::cta_size + (r.z >> 2)] = key.z;
        sMem1[(r.w & 3) * RadixSort::cta_size + (r.w >> 2)] = key.w;
        __syncthreads();

        // The above allows us to read without 4-way bank conflicts:
        key.x = sMem1[threadIdx.x];
        key.y = sMem1[threadIdx.x +     RadixSort::cta_size];
        key.z = sMem1[threadIdx.x + 2 * RadixSort::cta_size];
        key.w = sMem1[threadIdx.x + 3 * RadixSort::cta_size];
#else
        sMem1[r.x] = key.x;
        sMem1[r.y] = key.y;
        sMem1[r.z] = key.z;
        sMem1[r.w] = key.w;
        __syncthreads();

        // This access has 4-way bank conflicts
        key = sMem[threadIdx.x];
#endif

        __syncthreads();
    }
}

//----------------------------------------------------------------------------
//
// radixSortBlocks sorts each block of data independently in shared
// memory.  
//
// Done in two separate stages.  This stage calls radixSortBlock on each block 
// independently, sorting on the basis of bits (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------
template<uint nbits, uint startbit, bool fullBlocks, class PreProcess>
__global__ void radixSortBlocksKeysOnly(uint4* keysOut, uint4* keysIn, uint numElements, uint startBlock, PreProcess preprocess)
{
    extern __shared__ uint4 sMem[];

    uint4 key;

    const uint blockId = blockIdx.x + startBlock;
    const uint i = blockId * blockDim.x + threadIdx.x;
    const uint idx = i << 2;

    // handle non-full last block if array is not multiple of 1024 numElements
    if (!fullBlocks && idx+3 >= numElements)
    {
        if (idx >= numElements)
        {
            key   = make_uint4(UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX);
        }
        else
        {
            // for non-full block, we handle uint1 values instead of uint4
            uint *keys1    = (uint*)keysIn;

            key.x = (idx   < numElements) ? preprocess(keys1[idx])   : UINT_MAX;
            key.y = (idx+1 < numElements) ? preprocess(keys1[idx+1]) : UINT_MAX;
            key.z = (idx+2 < numElements) ? preprocess(keys1[idx+2]) : UINT_MAX;
            key.w = UINT_MAX;
        }
    }
    else
    {
        key = keysIn[i];
        key.x = preprocess(key.x);
        key.y = preprocess(key.y);
        key.z = preprocess(key.z);
        key.w = preprocess(key.w);
    }
    __syncthreads();
    radixSortBlockKeysOnly<nbits, startbit>(key);
    //__syncthreads();  // IS THIS NECESSARY?

    // handle non-full last block if array is not multiple of 1024 numElements
    if(!fullBlocks && idx+3 >= numElements)
    {
        if (idx < numElements) 
        {
            // for non-full block, we handle uint1 values instead of uint4
            uint *keys1   = (uint*)keysOut;

            keys1[idx]   = key.x;

            if (idx + 1 < numElements)
            {
                keys1[idx + 1]   = key.y;

                if (idx + 2 < numElements)
                {
                    keys1[idx + 2]   = key.z;
                }
            }
        }
    }
    else
    {
        keysOut[i]   = key;
    }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets 
// have been found. Depends on RadixSort::cta_size being 16 * number of radices 
// (i.e. 16 * 2^nbits).
// 
// This is quite fast and fully coalesces memory writes, albeit by doing extra 
// (potentially wasted) work allocating threads to portions of memory that are 
// not written out. Significantly faster than the generic approach on G80.
//----------------------------------------------------------------------------
template<uint startbit, bool fullBlocks, bool manualCoalesce, class PostProcess>
__global__ void reorderDataKeysOnly(uint  *outKeys, 
                                    uint2 *keys, 
                                    uint  *blockOffsets, 
                                    uint  *offsets, 
                                    uint  *sizes, 
                                    uint   numElements,
                                    uint   totalBlocks,
                                    uint   startBlock,
                                    PostProcess postprocess)
{
    __shared__ uint2 sKeys2[RadixSort::cta_size];
    __shared__ uint sOffsets[16];
    __shared__ uint sBlockOffsets[16];
    
    uint *sKeys1   = (uint*)sKeys2; 

    const uint blockId = blockIdx.x + startBlock;
    const uint i = blockId * blockDim.x + threadIdx.x;

    // handle non-full last block if array is not multiple of 1024 numElements
    if(!fullBlocks && (((i + 1) << 1) > numElements))
    {
        uint *keys1   = (uint*)keys;
        uint j = i << 1; 

        sKeys1[threadIdx.x << 1]   = (j < numElements) ? keys1[j]   : UINT_MAX; 
        j++; 
        sKeys1[(threadIdx.x << 1) + 1]   = (j < numElements) ? keys1[j]   : UINT_MAX; 
    }
    else
    {
        sKeys2[threadIdx.x]   = keys[i];
    }

    if (!manualCoalesce)
    {
        if(threadIdx.x < 16)  
        {
            sOffsets[threadIdx.x]      = offsets[threadIdx.x * totalBlocks + blockId];
            sBlockOffsets[threadIdx.x] = blockOffsets[blockId * 16 + threadIdx.x];
        }
        __syncthreads();

        uint radix = (sKeys1[threadIdx.x] >> startbit) & 0xF;
	    uint globalOffset = sOffsets[radix] + threadIdx.x - sBlockOffsets[radix];
	    
        if (fullBlocks || globalOffset < numElements)
        {
	        outKeys[globalOffset]   = postprocess(sKeys1[threadIdx.x]);
        }

        radix = (sKeys1[threadIdx.x + RadixSort::cta_size] >> startbit) & 0xF;
	    globalOffset = sOffsets[radix] + threadIdx.x + RadixSort::cta_size - sBlockOffsets[radix];
	    
        if (fullBlocks || globalOffset < numElements)
        {
	        outKeys[globalOffset]   = postprocess(sKeys1[threadIdx.x + RadixSort::cta_size]);
        }
    }
    else
    {
        __shared__ uint sSizes[16];

        if(threadIdx.x < 16)  
        {
            sOffsets[threadIdx.x]      = offsets[threadIdx.x * totalBlocks + blockId];
            sBlockOffsets[threadIdx.x] = blockOffsets[blockId * 16 + threadIdx.x];
            sSizes[threadIdx.x]        = sizes[threadIdx.x * totalBlocks + blockId];
        }
        __syncthreads();

        // 1 half-warp is responsible for writing out all values for 1 radix. 
        // Loops if there are more than 16 values to be written out. 
        // All start indices are rounded down to the nearest multiple of 16, and
        // all end indices are rounded up to the nearest multiple of 16.
        // Thus it can do extra work if the start and end indices are not multiples of 16
        // This is bounded by a factor of 2 (it can do 2X more work at most).

        const uint halfWarpID     = threadIdx.x >> 4;

        const uint halfWarpOffset = threadIdx.x & 0xF;
        const uint leadingInvalid = sOffsets[halfWarpID] & 0xF;

        uint startPos = sOffsets[halfWarpID] & 0xFFFFFFF0;
        uint endPos   = (sOffsets[halfWarpID] + sSizes[halfWarpID]) + 15 - 
                        ((sOffsets[halfWarpID] + sSizes[halfWarpID] - 1) & 0xF);
        uint numIterations = endPos - startPos;

        uint outOffset = startPos + halfWarpOffset;
        uint inOffset  = sBlockOffsets[halfWarpID] - leadingInvalid + halfWarpOffset;

        for(uint j = 0; j < numIterations; j += 16, outOffset += 16, inOffset += 16)
        {       
            if( (outOffset >= sOffsets[halfWarpID]) && 
                (inOffset - sBlockOffsets[halfWarpID] < sSizes[halfWarpID])) 
            {
                if(blockId < totalBlocks - 1 || outOffset < numElements) 
                {
                    outKeys[outOffset] = postprocess(sKeys1[inOffset]);
                }
            }       
        }
    }
}

//----------------------------------------------------------------------------
// Perform one step of the radix sort.  Sorts by nbits key bits per step, 
// starting at startbit.
//----------------------------------------------------------------------------
template<uint nbits, uint startbit, class PreProcess, class PostProcess>
void radixSortStepKeysOnly(uint *keys, 
                           uint *tempKeys, 
                           uint *counters, 
                           uint *countersSum, 
                           uint *blockOffsets, 
                           uint numElements, 
                           bool manualCoalesce,
                           PreProcess  preprocess,
                           PostProcess postprocess)
{
    const uint eltsPerBlock = RadixSort::cta_size * 4;
    const uint eltsPerBlock2 = RadixSort::cta_size * 2;

    bool fullBlocks = ((numElements % eltsPerBlock) == 0);
    uint numBlocks = (fullBlocks) ? 
        (numElements / eltsPerBlock) : 
        (numElements / eltsPerBlock + 1);
    uint numBlocks2 = ((numElements % eltsPerBlock2) == 0) ?
        (numElements / eltsPerBlock2) : 
        (numElements / eltsPerBlock2 + 1);

    const uint max1DBlocks = 65535;

    
    for (uint block = 0; block < numBlocks; block += max1DBlocks)
    {
        uint blocks   = min(max1DBlocks, numBlocks - block);
        
        if (blocks < max1DBlocks && !fullBlocks)
        {
            radixSortBlocksKeysOnly<nbits, startbit, false>
                <<<blocks, RadixSort::cta_size, 4 * RadixSort::cta_size * sizeof(uint)>>>
                    ((uint4*)tempKeys, (uint4*)keys, numElements, block, preprocess);
            synchronize_if_enabled("radixSortBlocksKeysOnly");
        }
        else
        {
            radixSortBlocksKeysOnly<nbits, startbit, true>
                <<<blocks, RadixSort::cta_size, 4 * RadixSort::cta_size * sizeof(uint)>>>
                    ((uint4*)tempKeys, (uint4*)keys, numElements, block, preprocess);
            synchronize_if_enabled("radixSortBlocksKeysOnly");
        }
    }

    for (uint block = 0; block < numBlocks2; block += max1DBlocks)
    {
        uint blocks   = min(max1DBlocks, numBlocks2 - block);

        if (blocks < max1DBlocks && !fullBlocks)
        {
            findRadixOffsets<startbit, false>
                <<<blocks, RadixSort::cta_size, 3 * RadixSort::cta_size * sizeof(uint)>>>
                    ((uint2*)tempKeys, counters, blockOffsets, numElements, numBlocks2, block);
            synchronize_if_enabled("findRadixOffsets");
        }
        else
        {
            findRadixOffsets<startbit, true>
                <<<blocks, RadixSort::cta_size, 3 * RadixSort::cta_size * sizeof(uint)>>>
                    ((uint2*)tempKeys, counters, blockOffsets, numElements, numBlocks2, block);
            synchronize_if_enabled("findRadixOffsets");
        }
    }

    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(counters), 
                            thrust::device_ptr<unsigned int>(counters + 16*numBlocks2),
                            thrust::device_ptr<unsigned int>(countersSum));

    for (uint block = 0; block < numBlocks2; block += max1DBlocks)
    {
        uint blocks   = min(max1DBlocks, numBlocks2 - block);
        
        if (blocks < max1DBlocks && !fullBlocks)
        {
            if (manualCoalesce)
            {
                reorderDataKeysOnly<startbit, false, true><<<blocks, RadixSort::cta_size>>>
                    (keys, (uint2*)tempKeys, blockOffsets, countersSum, counters, 
                     numElements, numBlocks2, block, postprocess);
                synchronize_if_enabled("reorderDataKeysOnly");
            }
            else
            {
                reorderDataKeysOnly<startbit, false, false><<<blocks, RadixSort::cta_size>>>
                    (keys, (uint2*)tempKeys, blockOffsets, countersSum, counters, 
                     numElements, numBlocks2, block, postprocess);
                synchronize_if_enabled("reorderDataKeysOnly");
            }
        }
        else
        {
            if (manualCoalesce)
            {
                reorderDataKeysOnly<startbit, true, true><<<blocks, RadixSort::cta_size>>>
                    (keys, (uint2*)tempKeys, blockOffsets, countersSum, counters, 
                     numElements, numBlocks2, block, postprocess);
                synchronize_if_enabled("reorderDataKeysOnly");
            }
            else
            {
                reorderDataKeysOnly<startbit, true, false><<<blocks, RadixSort::cta_size>>>
                    (keys, (uint2*)tempKeys, blockOffsets, countersSum, counters, 
                     numElements, numBlocks2, block, postprocess);
                synchronize_if_enabled("reorderDataKeysOnly");
            }
        }
    }    

    checkCudaError("radixSortStepKeysOnly");
}

//----------------------------------------------------------------------------
// Main radix sort function.  Sorts in place in the keys and values arrays,
// but uses the other device arrays as temporary storage.  All pointer 
// parameters are device pointers.  Uses exclusive_scan() for the prefix 
// sum of radix counters.
//----------------------------------------------------------------------------
template <class PreProcess, class PostProcess>
void radixSortKeysOnly(uint *keys, 
                       uint *tempKeys, 
                       uint *counters,
                       uint *countersSum,
                       uint *blockOffsets,
                       uint numElements, 
                       uint keyBits,
                       bool manualCoalesce,
                       PreProcess  preprocess,
                       PostProcess postprocess)
{
#define RS_KeyOnly(bit,pre)                                                                           \
    if (bit + 4 < keyBits)                                                                            \
        radixSortStepKeysOnly<4,(bit)>(keys, tempKeys,                                                \
                                       counters, countersSum, blockOffsets,                           \
                                       numElements, manualCoalesce,                                   \
                                       pre,                                                           \
                                       thrust::identity<uint>());                                     \
    else if (bit < keyBits)                                                                           \
        radixSortStepKeysOnly<4,(bit)>(keys, tempKeys,                                                \
                                       counters, countersSum, blockOffsets,                           \
                                       numElements, manualCoalesce,                                   \
                                       pre,                                                           \
                                       postprocess);

    RS_KeyOnly( 0, preprocess);
    RS_KeyOnly( 4, thrust::identity<uint>());
    RS_KeyOnly( 8, thrust::identity<uint>());
    RS_KeyOnly(12, thrust::identity<uint>());
    RS_KeyOnly(16, thrust::identity<uint>());
    RS_KeyOnly(20, thrust::identity<uint>());
    RS_KeyOnly(24, thrust::identity<uint>());
    RS_KeyOnly(28, thrust::identity<uint>());

#undef RS_KeyOnly

    checkCudaError("radixSortKeysOnly");
}

/////////////////////////////////////
// NEW CODE
/////////////////////////////////////

#define BLOCKING(N,B) ( ((N) + ((B) - 1))/(B) )

inline
bool radix_sort_use_manual_coalescing(void)
{
    int deviceID = -1;
    bool manualCoalesce = true;
    if (cudaSuccess == cudaGetDevice(&deviceID))
    {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, deviceID);
        manualCoalesce = (devprop.major < 2 && devprop.minor < 2); // sm_12 and later devices don't need help with coalesce
    }

    return manualCoalesce;
}



template <class PreProcess, class PostProcess>
void radix_sort(unsigned int * keys, 
                unsigned int numElements, 
                PreProcess preprocess,
                PostProcess postprocess,
                unsigned int keyBits = UINT_MAX)
{
    if (numElements == 0 || keyBits == 0)
        return;

    if (!thrust::detail::util::is_aligned(keys, sizeof(uint4)))
    {
        // keys is misaligned, copy to temp array and try again
        thrust::detail::raw_cuda_device_buffer<unsigned int> aligned_keys(thrust::device_ptr<unsigned int>(keys),
                                                                          thrust::device_ptr<unsigned int>(keys) + numElements);
        
        assert(thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&aligned_keys[0]), sizeof(uint4)));

        radix_sort(thrust::raw_pointer_cast(&aligned_keys[0]), numElements, preprocess, postprocess, keyBits);
        
        thrust::copy(aligned_keys.begin(), aligned_keys.end(), thrust::device_ptr<unsigned int>(keys));

        return;
    }

    unsigned int numBlocks  = BLOCKING(numElements, RadixSort::cta_size * 4);
        
    thrust::detail::raw_cuda_device_buffer<unsigned int> temp_keys(numElements);
    thrust::detail::raw_cuda_device_buffer<unsigned int> counters(RadixSort::warp_size * numBlocks);
    thrust::detail::raw_cuda_device_buffer<unsigned int> histogram(RadixSort::warp_size * numBlocks);
    thrust::detail::raw_cuda_device_buffer<unsigned int> block_offsets(RadixSort::warp_size * numBlocks);

    bool manualCoalesce = radix_sort_use_manual_coalescing();

    unsigned int min_value = 0;

    if (keyBits == UINT_MAX)
    {
        //determine number of keyBits dynamically 
        thrust::pair<unsigned int, unsigned int> minmax_value = compute_minmax(keys, numElements, preprocess);
        keyBits = compute_keyBits(minmax_value);
        min_value = minmax_value.first;
    }

    radixSortKeysOnly(keys,
                      thrust::raw_pointer_cast(&temp_keys[0]), 
                      thrust::raw_pointer_cast(&counters[0]),
                      thrust::raw_pointer_cast(&histogram[0]),
                      thrust::raw_pointer_cast(&block_offsets[0]),
                      numElements, keyBits, manualCoalesce,
                      modified_preprocess<PreProcess, unsigned int>(preprocess, min_value),
                      modified_postprocess<PostProcess, unsigned int>(postprocess, min_value));
}



template <class PreProcess, class PostProcess>
void radix_sort_by_key(unsigned int * keys, 
                       unsigned int * values, 
                       unsigned int numElements, 
                       PreProcess preprocess,
                       PostProcess postprocess,
                       unsigned int keyBits = UINT_MAX)
{
    if (numElements == 0 || keyBits == 0)
        return;

    if (!thrust::detail::util::is_aligned(keys, sizeof(uint4)))
    {
        // keys is misaligned, copy to temp array and try again
        thrust::detail::raw_cuda_device_buffer<unsigned int> aligned_keys(thrust::device_ptr<unsigned int>(keys),
                                                                          thrust::device_ptr<unsigned int>(keys) + numElements);

        assert(thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&aligned_keys[0]), sizeof(uint4)));

        radix_sort_by_key(thrust::raw_pointer_cast(&aligned_keys[0]), values, numElements, preprocess, postprocess, keyBits);
        
        thrust::copy(aligned_keys.begin(), aligned_keys.end(), thrust::device_ptr<unsigned int>(keys));

        return;
    }
    
    if (!thrust::detail::util::is_aligned(values, sizeof(uint4)))
    {
        // values is misaligned, copy to temp array and try again
        thrust::detail::raw_cuda_device_buffer<unsigned int> aligned_values(thrust::device_ptr<unsigned int>(values),
                                                                            thrust::device_ptr<unsigned int>(values) + numElements);

        assert(thrust::detail::util::is_aligned(thrust::raw_pointer_cast(&aligned_values[0]), sizeof(uint4)));
        
        radix_sort_by_key(keys, thrust::raw_pointer_cast(&aligned_values[0]), numElements, preprocess, postprocess, keyBits);
        
        thrust::copy(aligned_values.begin(), aligned_values.end(), thrust::device_ptr<unsigned int>(values));

        return;
    }
    
    unsigned int numBlocks  = BLOCKING(numElements, RadixSort::cta_size * 4);

    thrust::detail::raw_cuda_device_buffer<unsigned int> temp_keys(numElements);
    thrust::detail::raw_cuda_device_buffer<unsigned int> temp_values(numElements);
    thrust::detail::raw_cuda_device_buffer<unsigned int> counters(RadixSort::warp_size * numBlocks);
    thrust::detail::raw_cuda_device_buffer<unsigned int> histogram(RadixSort::warp_size * numBlocks);
    thrust::detail::raw_cuda_device_buffer<unsigned int> block_offsets(RadixSort::warp_size * numBlocks);

    bool manualCoalesce = radix_sort_use_manual_coalescing();
    
    unsigned int min_value = 0;

    if (keyBits == UINT_MAX)
    {
        //determine number of keyBits dynamically 
        thrust::pair<unsigned int, unsigned int> minmax_value = compute_minmax(keys, numElements, preprocess);
        keyBits = compute_keyBits(minmax_value);
        min_value = minmax_value.first;
    }

    radixSort(keys, values,
              thrust::raw_pointer_cast(&temp_keys[0]), 
              thrust::raw_pointer_cast(&temp_values[0]),
              thrust::raw_pointer_cast(&counters[0]),
              thrust::raw_pointer_cast(&histogram[0]),
              thrust::raw_pointer_cast(&block_offsets[0]),
              numElements, keyBits, manualCoalesce,
              modified_preprocess<PreProcess, unsigned int>(preprocess, min_value),
              modified_postprocess<PostProcess, unsigned int>(postprocess, min_value));
}
#undef BLOCKING

} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END


#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

