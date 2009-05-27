/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


// do not attempt to compile this file with any other compiler
#ifdef __CUDACC__

#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace thrust
{

namespace sorting
{

namespace detail
{

namespace device
{

namespace cuda
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
    const unsigned int CTA_SIZE  = 256;
    const unsigned int WARP_SIZE = 32;
}


#ifdef __DEVICE_EMULATION__
#define __SYNC __syncthreads();
#else
#define __SYNC
#endif

typedef unsigned int uint;

//----------------------------------------------------------------------------
// Scans each warp in parallel ("warp-scan"), one element per thread.
// uses 2 numElements of shared memory per thread (64 numElements per warp)
//----------------------------------------------------------------------------
template<class T, int maxlevel>
__device__ T scanwarp(T val, T* sData)
{
    // The following is the same as 2 * RadixSort::WARP_SIZE * warpId + threadInWarp = 
    // 64*(threadIdx.x >> 5) + (threadIdx.x & (RadixSort::WARP_SIZE - 1))
    int idx = 2 * threadIdx.x - (threadIdx.x & (RadixSort::WARP_SIZE - 1));
    sData[idx] = 0;
    idx += RadixSort::WARP_SIZE;
    sData[idx] = val;          __SYNC

#ifdef __DEVICE_EMULATION__
        T t = sData[idx -  1]; __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx -  2];   __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx -  4];   __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx -  8];   __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx - 16];   __SYNC 
        sData[idx] += t;       __SYNC
#else
        if (0 <= maxlevel) { sData[idx] += sData[idx - 1]; } __SYNC
        if (1 <= maxlevel) { sData[idx] += sData[idx - 2]; } __SYNC
        if (2 <= maxlevel) { sData[idx] += sData[idx - 4]; } __SYNC
        if (3 <= maxlevel) { sData[idx] += sData[idx - 8]; } __SYNC
        if (4 <= maxlevel) { sData[idx] += sData[idx -16]; } __SYNC
#endif

    return sData[idx] - val;  // convert inclusive -> exclusive
}

//----------------------------------------------------------------------------
// scan4 scans 4*RadixSort::CTA_SIZE numElements in a block (4 per thread), using 
// a warp-scan algorithm
//----------------------------------------------------------------------------
template <typename T>
__device__ uint4 scan4(T idata)  //T = uint4
{    
    extern  __shared__  uint ptr[];
    
    uint idx = threadIdx.x;

    uint4 val4 = idata;
    uint sum[3];
    sum[0] = val4.x;
    sum[1] = val4.y + sum[0];
    sum[2] = val4.z + sum[1];
    
    uint val = val4.w + sum[2];
    
    val = scanwarp<uint, 4>(val, ptr);
    __syncthreads();

    if ((idx & (RadixSort::WARP_SIZE - 1)) == RadixSort::WARP_SIZE - 1)
    {
        ptr[idx >> 5] = val + val4.w + sum[2];
    }
    __syncthreads();

#ifndef __DEVICE_EMULATION__
    if (idx < RadixSort::WARP_SIZE)
#endif
    {
        ptr[idx] = scanwarp<uint, 2>(ptr[idx], ptr);
    }
    __syncthreads();

    val += ptr[idx >> 5];

    val4.x = val;
    val4.y = val + sum[0];
    val4.z = val + sum[1];
    val4.w = val + sum[2];

    return val4;
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
template<uint nbits, uint startbit, bool floatFlip>
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

        uint4 r = rank4<RadixSort::CTA_SIZE>(lsb);

#if 1
        // This arithmetic strides the ranks across 4 CTA_SIZE regions
        sMem1[(r.x & 3) * RadixSort::CTA_SIZE + (r.x >> 2)] = key.x;
        sMem1[(r.y & 3) * RadixSort::CTA_SIZE + (r.y >> 2)] = key.y;
        sMem1[(r.z & 3) * RadixSort::CTA_SIZE + (r.z >> 2)] = key.z;
        sMem1[(r.w & 3) * RadixSort::CTA_SIZE + (r.w >> 2)] = key.w;
        __syncthreads();

        // The above allows us to read without 4-way bank conflicts:
        key.x = sMem1[threadIdx.x];
        key.y = sMem1[threadIdx.x +     RadixSort::CTA_SIZE];
        key.z = sMem1[threadIdx.x + 2 * RadixSort::CTA_SIZE];
        key.w = sMem1[threadIdx.x + 3 * RadixSort::CTA_SIZE];

        __syncthreads();

        sMem1[(r.x & 3) * RadixSort::CTA_SIZE + (r.x >> 2)] = value.x;
        sMem1[(r.y & 3) * RadixSort::CTA_SIZE + (r.y >> 2)] = value.y;
        sMem1[(r.z & 3) * RadixSort::CTA_SIZE + (r.z >> 2)] = value.z;
        sMem1[(r.w & 3) * RadixSort::CTA_SIZE + (r.w >> 2)] = value.w;
        __syncthreads();

        value.x = sMem1[threadIdx.x];
        value.y = sMem1[threadIdx.x +     RadixSort::CTA_SIZE];
        value.z = sMem1[threadIdx.x + 2 * RadixSort::CTA_SIZE];
        value.w = sMem1[threadIdx.x + 3 * RadixSort::CTA_SIZE];
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
                                const PreProcess preprocess)
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
    radixSortBlock<nbits, startbit, false>(key, value);
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
    uint  *sStartPointers  = (uint*)(sMem2 + RadixSort::CTA_SIZE);

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
    if(sRadix1[threadIdx.x + RadixSort::CTA_SIZE] != sRadix1[threadIdx.x + RadixSort::CTA_SIZE - 1]) 
    {
        sStartPointers[sRadix1[threadIdx.x + RadixSort::CTA_SIZE]] = threadIdx.x + RadixSort::CTA_SIZE;
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
    if(sRadix1[threadIdx.x + RadixSort::CTA_SIZE] != sRadix1[threadIdx.x + RadixSort::CTA_SIZE - 1] ) 
    {
        sStartPointers[sRadix1[threadIdx.x + RadixSort::CTA_SIZE - 1]] = 
            threadIdx.x + RadixSort::CTA_SIZE - sStartPointers[sRadix1[threadIdx.x + RadixSort::CTA_SIZE - 1]];
    }

    if(threadIdx.x == RadixSort::CTA_SIZE - 1) 
    {
        sStartPointers[sRadix1[2 * RadixSort::CTA_SIZE - 1]] = 
            2 * RadixSort::CTA_SIZE - sStartPointers[sRadix1[2 * RadixSort::CTA_SIZE - 1]];
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
// have been found. Depends on RadixSort::CTA_SIZE being 16 * number of radices 
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
                            const PostProcess postprocess)
{
    __shared__ uint2 sKeys2[RadixSort::CTA_SIZE];
    __shared__ uint2 sValues2[RadixSort::CTA_SIZE];
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

        radix = (sKeys1[threadIdx.x + RadixSort::CTA_SIZE] >> startbit) & 0xF;
	    globalOffset = sOffsets[radix] + threadIdx.x + RadixSort::CTA_SIZE - sBlockOffsets[radix];
	    
        if (fullBlocks || globalOffset < numElements)
        {
	        outKeys[globalOffset]   = postprocess(sKeys1[threadIdx.x + RadixSort::CTA_SIZE]);
	        outValues[globalOffset] = sValues1[threadIdx.x + RadixSort::CTA_SIZE];
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
                   const PreProcess&  preprocess,
                   const PostProcess& postprocess)
{
    const uint eltsPerBlock  = RadixSort::CTA_SIZE * 4;
    const uint eltsPerBlock2 = RadixSort::CTA_SIZE * 2;

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
                <<<blocks, RadixSort::CTA_SIZE, 4 * RadixSort::CTA_SIZE * sizeof(uint)>>>
                    ((uint4*)tempKeys, (uint4*)tempValues, (uint4*)keys, (uint4*)values, numElements, block, preprocess);
        }
        else
        {
            radixSortBlocks<nbits, startbit, true>
                <<<blocks, RadixSort::CTA_SIZE, 4 * RadixSort::CTA_SIZE * sizeof(uint)>>>
                    ((uint4*)tempKeys, (uint4*)tempValues, (uint4*)keys, (uint4*)values, numElements, block, preprocess);
        }
    }

    for (uint block = 0; block < numBlocks2; block += max1DBlocks)
    {
        uint blocks   = min(max1DBlocks, numBlocks2 - block);

        if (blocks < max1DBlocks && !fullBlocks)
        {
            findRadixOffsets<startbit, false>
                <<<blocks, RadixSort::CTA_SIZE, 3 * RadixSort::CTA_SIZE * sizeof(uint)>>>
                    ((uint2*)tempKeys, counters, blockOffsets, numElements, numBlocks2, block);
        }
        else
        {
            findRadixOffsets<startbit, true>
                <<<blocks, RadixSort::CTA_SIZE, 3 * RadixSort::CTA_SIZE * sizeof(uint)>>>
                    ((uint2*)tempKeys, counters, blockOffsets, numElements, numBlocks2, block);
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
                reorderData<startbit, false, true><<<blocks, RadixSort::CTA_SIZE>>>
                    (keys, values, (uint2*)tempKeys, (uint2*)tempValues, 
                     blockOffsets, countersSum, counters, numElements, numBlocks2, block, postprocess);
            }
            else
            {
                reorderData<startbit, false, false><<<blocks, RadixSort::CTA_SIZE>>>
                    (keys, values, (uint2*)tempKeys, (uint2*)tempValues, 
                     blockOffsets, countersSum, counters, numElements, numBlocks2, block, postprocess);
            }
        }
        else
        {
            if (manualCoalesce)
            {
                reorderData<startbit, true, true><<<blocks, RadixSort::CTA_SIZE>>>
                    (keys, values, (uint2*)tempKeys, (uint2*)tempValues, 
                     blockOffsets, countersSum, counters, numElements, numBlocks2, block, postprocess);
            }
            else
            {
                reorderData<startbit, true, false><<<blocks, RadixSort::CTA_SIZE>>>
                    (keys, values, (uint2*)tempKeys, (uint2*)tempValues, 
                     blockOffsets, countersSum, counters, numElements, numBlocks2, block, postprocess);
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
               const PreProcess&  preprocess,
               const PostProcess& postprocess)
{
#define RS_KeyValue(bit,pre,post)                                            \
    if (keyBits > (bit))                                                     \
        radixSortStep<4,(bit)>(keys, values, tempKeys, tempValues,           \
                               counters, countersSum, blockOffsets,          \
                               numElements, manualCoalesce,                  \
                               (pre), (post))                                

    RS_KeyValue( 0, preprocess,                thrust::identity<uint>());
    RS_KeyValue( 4, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyValue( 8, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyValue(12, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyValue(16, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyValue(20, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyValue(24, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyValue(28, thrust::identity<uint>(), postprocess              );

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
        // This arithmetic strides the ranks across 4 CTA_SIZE regions
        sMem1[(r.x & 3) * RadixSort::CTA_SIZE + (r.x >> 2)] = key.x;
        sMem1[(r.y & 3) * RadixSort::CTA_SIZE + (r.y >> 2)] = key.y;
        sMem1[(r.z & 3) * RadixSort::CTA_SIZE + (r.z >> 2)] = key.z;
        sMem1[(r.w & 3) * RadixSort::CTA_SIZE + (r.w >> 2)] = key.w;
        __syncthreads();

        // The above allows us to read without 4-way bank conflicts:
        key.x = sMem1[threadIdx.x];
        key.y = sMem1[threadIdx.x +     RadixSort::CTA_SIZE];
        key.z = sMem1[threadIdx.x + 2 * RadixSort::CTA_SIZE];
        key.w = sMem1[threadIdx.x + 3 * RadixSort::CTA_SIZE];
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
__global__ void radixSortBlocksKeysOnly(uint4* keysOut, uint4* keysIn, uint numElements, uint startBlock, const PreProcess preprocess)
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
// have been found. Depends on RadixSort::CTA_SIZE being 16 * number of radices 
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
                                    const PostProcess postprocess)
{
    __shared__ uint2 sKeys2[RadixSort::CTA_SIZE];
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

        radix = (sKeys1[threadIdx.x + RadixSort::CTA_SIZE] >> startbit) & 0xF;
	    globalOffset = sOffsets[radix] + threadIdx.x + RadixSort::CTA_SIZE - sBlockOffsets[radix];
	    
        if (fullBlocks || globalOffset < numElements)
        {
	        outKeys[globalOffset]   = postprocess(sKeys1[threadIdx.x + RadixSort::CTA_SIZE]);
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
                           const PreProcess&  preprocess,
                           const PostProcess& postprocess)
{
    const uint eltsPerBlock = RadixSort::CTA_SIZE * 4;
    const uint eltsPerBlock2 = RadixSort::CTA_SIZE * 2;

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
                <<<blocks, RadixSort::CTA_SIZE, 4 * RadixSort::CTA_SIZE * sizeof(uint)>>>
                    ((uint4*)tempKeys, (uint4*)keys, numElements, block, preprocess);
        }
        else
        {
            radixSortBlocksKeysOnly<nbits, startbit, true>
                <<<blocks, RadixSort::CTA_SIZE, 4 * RadixSort::CTA_SIZE * sizeof(uint)>>>
                    ((uint4*)tempKeys, (uint4*)keys, numElements, block, preprocess);
        }
    }

    for (uint block = 0; block < numBlocks2; block += max1DBlocks)
    {
        uint blocks   = min(max1DBlocks, numBlocks2 - block);

        if (blocks < max1DBlocks && !fullBlocks)
        {
            findRadixOffsets<startbit, false>
                <<<blocks, RadixSort::CTA_SIZE, 3 * RadixSort::CTA_SIZE * sizeof(uint)>>>
                    ((uint2*)tempKeys, counters, blockOffsets, numElements, numBlocks2, block);
        }
        else
        {
            findRadixOffsets<startbit, true>
                <<<blocks, RadixSort::CTA_SIZE, 3 * RadixSort::CTA_SIZE * sizeof(uint)>>>
                    ((uint2*)tempKeys, counters, blockOffsets, numElements, numBlocks2, block);
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
                reorderDataKeysOnly<startbit, false, true><<<blocks, RadixSort::CTA_SIZE>>>
                    (keys, (uint2*)tempKeys, blockOffsets, countersSum, counters, 
                     numElements, numBlocks2, block, postprocess);
            }
            else
            {
                reorderDataKeysOnly<startbit, false, false><<<blocks, RadixSort::CTA_SIZE>>>
                    (keys, (uint2*)tempKeys, blockOffsets, countersSum, counters, 
                     numElements, numBlocks2, block, postprocess);
            }
        }
        else
        {
            if (manualCoalesce)
            {
                reorderDataKeysOnly<startbit, true, true><<<blocks, RadixSort::CTA_SIZE>>>
                    (keys, (uint2*)tempKeys, blockOffsets, countersSum, counters, 
                     numElements, numBlocks2, block, postprocess);
            }
            else
            {
                reorderDataKeysOnly<startbit, true, false><<<blocks, RadixSort::CTA_SIZE>>>
                    (keys, (uint2*)tempKeys, blockOffsets, countersSum, counters, 
                     numElements, numBlocks2, block, postprocess);
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
                       const PreProcess&  preprocess,
                       const PostProcess& postprocess)
{
#define RS_KeyOnly(bit,pre,post)                                                     \
    if (keyBits > (bit))                                                             \
        radixSortStepKeysOnly<4,(bit)>(keys, tempKeys,                               \
                                       counters, countersSum, blockOffsets,          \
                                       numElements, manualCoalesce,                  \
                                       (pre), (post))                                

    RS_KeyOnly( 0, preprocess,                thrust::identity<uint>());
    RS_KeyOnly( 4, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyOnly( 8, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyOnly(12, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyOnly(16, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyOnly(20, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyOnly(24, thrust::identity<uint>(), thrust::identity<uint>());
    RS_KeyOnly(28, thrust::identity<uint>(), postprocess              );

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



template <class PreProcess>
unsigned int determine_keyBits(const unsigned int * keys,
                               const unsigned int numElements,
                               const PreProcess& preprocess)
{
    unsigned int max_val = thrust::transform_reduce(thrust::device_ptr<const unsigned int>(keys),
                                                     thrust::device_ptr<const unsigned int>(keys + numElements),
                                                     preprocess,
                                                     (unsigned int) 0,
                                                     thrust::maximum<unsigned int>());

    // compute index of most significant bit
    unsigned int keyBits = 0;     
    while(max_val){
        keyBits++;
        max_val >>= 1;
    }

    return keyBits;
}



template <class PreProcess, class PostProcess>
void radix_sort(unsigned int * keys, 
                unsigned int numElements, 
                const PreProcess& preprocess,
                const PostProcess& postprocess,
                unsigned int keyBits = UINT_MAX)
{
    //determine number of keyBits dynamically 
    if (keyBits == UINT_MAX)
        keyBits = determine_keyBits(keys, numElements, preprocess);

    if (keyBits == 0)
        return;

    unsigned int numBlocks  = BLOCKING(numElements, RadixSort::CTA_SIZE * 4);

    thrust::device_ptr<unsigned int> temp_keys     = thrust::device_malloc<unsigned int>(numElements);
    thrust::device_ptr<unsigned int> counters      = thrust::device_malloc<unsigned int>(RadixSort::WARP_SIZE * numBlocks);
    thrust::device_ptr<unsigned int> histogram     = thrust::device_malloc<unsigned int>(RadixSort::WARP_SIZE * numBlocks);
    thrust::device_ptr<unsigned int> block_offsets = thrust::device_malloc<unsigned int>(RadixSort::WARP_SIZE * numBlocks);

    bool manualCoalesce = radix_sort_use_manual_coalescing();

    radixSortKeysOnly(keys, temp_keys.get(), 
                      counters.get(), histogram.get(), block_offsets.get(),
                      numElements, keyBits, manualCoalesce,
                      preprocess, postprocess);

    thrust::device_free(temp_keys);
    thrust::device_free(counters);
    thrust::device_free(histogram);
    thrust::device_free(block_offsets);
}



template <class PreProcess, class PostProcess>
void radix_sort_by_key(unsigned int * keys, 
                       unsigned int * values, 
                       unsigned int numElements, 
                       const PreProcess& preprocess,
                       const PostProcess& postprocess,
                       unsigned int keyBits = UINT_MAX)
{
    //determine number of keyBits dynamically 
    if (keyBits == UINT_MAX)
        keyBits = determine_keyBits(keys, numElements, preprocess);
    
    if (keyBits == 0)
        return;

    unsigned int numBlocks  = BLOCKING(numElements, RadixSort::CTA_SIZE * 4);
    
    thrust::device_ptr<unsigned int> temp_keys     = thrust::device_malloc<unsigned int>(numElements);
    thrust::device_ptr<unsigned int> temp_values   = thrust::device_malloc<unsigned int>(numElements);
    thrust::device_ptr<unsigned int> counters      = thrust::device_malloc<unsigned int>(RadixSort::WARP_SIZE * numBlocks);
    thrust::device_ptr<unsigned int> histogram     = thrust::device_malloc<unsigned int>(RadixSort::WARP_SIZE * numBlocks);
    thrust::device_ptr<unsigned int> block_offsets = thrust::device_malloc<unsigned int>(RadixSort::WARP_SIZE * numBlocks);

    bool manualCoalesce = radix_sort_use_manual_coalescing();
    
    radixSort(keys, values, temp_keys.get(), temp_values.get(),
              counters.get(), histogram.get(), block_offsets.get(),
              numElements, keyBits, manualCoalesce,
              preprocess, postprocess);
    
    thrust::device_free(temp_keys);
    thrust::device_free(temp_values);
    thrust::device_free(counters);
    thrust::device_free(histogram);
    thrust::device_free(block_offsets);
}
#undef BLOCKING


} // end namespace cuda

} // end namespace device

} // end sorting detail

} // end sorting sorting

} // end sorting thrust

#endif // __CUDACC__

