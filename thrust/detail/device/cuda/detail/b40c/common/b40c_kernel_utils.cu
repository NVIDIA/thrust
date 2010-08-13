/**
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 */


//------------------------------------------------------------------------------
// Common B40C Defines, Properties, and Routines 
//------------------------------------------------------------------------------


#pragma once

#include <cuda.h>

namespace b40c {

//------------------------------------------------------------------------------
// Device properties 
//------------------------------------------------------------------------------


#ifndef __CUDA_ARCH__
	#define __CUDA_ARCH__ 0
#endif

#define FERMI(version)								(version >= 200)
#define LOG_WARP_THREADS							5									// 32 threads in a warp
#define WARP_THREADS								(1 << LOG_WARP_THREADS)
#define LOG_MEM_BANKS(version) 						((version >= 200) ? 5 : 4)			// 32 banks on fermi, 16 on tesla
#define MEM_BANKS(version)							(1 << LOG_MEM_BANKS(version))

#if __CUDA_ARCH__ >= 200
	#define FastMul(a, b) (a * b)
#else
	#define FastMul(a, b) (__umul24(a, b))
#endif	


#if __CUDA_ARCH__ >= 120
	#define WarpVoteAll(active_threads, predicate) (__all(predicate))
#else 
	#define WarpVoteAll(active_threads, predicate) (EmulatedWarpVoteAll<active_threads>(predicate))
#endif

#if __CUDA_ARCH__ >= 200
	#define TallyWarpVote(active_threads, predicate, storage) (__popc(__ballot(predicate)))
#else 
	#define TallyWarpVote(active_threads, predicate, storage) (TallyWarpVoteSm10<active_threads>(predicate, storage))
#endif

#ifdef __LP64__
	#define _B40C_LP64_ true
#else
	#define _B40C_LP64_ false
#endif

#define _B40C_REG_MISER_QUALIFIER_ __shared__


//------------------------------------------------------------------------------
// Handy routines 
//------------------------------------------------------------------------------


/**
 * Select maximum
 */
#define MAX(a, b) ((a > b) ? a : b)


/**
 * Perform a swap
 */
template <typename T> 
void __host__ __device__ __forceinline__ Swap(T &a, T &b) {
	T temp = a;
	a = b;
	b = temp;
}


/**
 * MagnitudeShift().  Allows you to shift left for positive magnitude values, 
 * right for negative.   
 * 
 * N.B. This code is a little strange; we are using this meta-programming 
 * pattern of partial template specialization for structures in order to 
 * decide whether to shift left or right.  Normally we would just use a 
 * conditional to decide if something was negative or not and then shift 
 * accordingly, knowing that the compiler will elide the untaken branch, 
 * i.e., the out-of-bounds shift during dead code elimination. However, 
 * the pass for bounds-checking shifts seems to happen before the DCE 
 * phase, which results in a an unsightly number of compiler warnings, so 
 * we force the issue earlier using structural template specialization.
 */

template <typename K, int magnitude, bool shift_left> struct MagnitudeShiftOp;

template <typename K, int magnitude> 
struct MagnitudeShiftOp<K, magnitude, true> {
	__device__ __forceinline__ static K Shift(K key) {
		return key << magnitude;
	}
};

template <typename K, int magnitude> 
struct MagnitudeShiftOp<K, magnitude, false> {
	__device__ __forceinline__ static K Shift(K key) {
		return key >> magnitude;
	}
};

template <typename K, int magnitude> 
__device__ __forceinline__ K MagnitudeShift(K key) {
	return MagnitudeShiftOp<K, (magnitude > 0) ? magnitude : magnitude * -1, (magnitude > 0)>::Shift(key);
}


/**
 * Supress warnings for unused constants
 */
template <typename T>
__device__ __forceinline__ void SuppressUnusedConstantWarning(const T) {}




//------------------------------------------------------------------------------
// Common device routines
//------------------------------------------------------------------------------


/**
 * Perform a warp-synchrounous prefix scan.  Allows for diverting a warp's
 * threads into separate scan problems (multi-scan). 
 */
template <unsigned int NUM_ELEMENTS, bool MULTI_SCAN> 
__device__ __forceinline__ unsigned int WarpScan(
	volatile unsigned int warpscan[][NUM_ELEMENTS],
	unsigned int partial_reduction,
	unsigned int copy_section) {
	
	unsigned int warpscan_idx;
	if (MULTI_SCAN) {
		warpscan_idx = threadIdx.x & (NUM_ELEMENTS - 1);
	} else {
		warpscan_idx = threadIdx.x;
	}

	warpscan[1][warpscan_idx] = partial_reduction;

	if (NUM_ELEMENTS > 1) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 1];
	if (NUM_ELEMENTS > 2) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 2];
	if (NUM_ELEMENTS > 4) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 4];
	if (NUM_ELEMENTS > 8) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 8];
	if (NUM_ELEMENTS > 16) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 16];
	
	if (copy_section > 0) {
		warpscan[1 + copy_section][warpscan_idx] = partial_reduction;
	}
	
	return warpscan[1][warpscan_idx - 1];
}

/**
 * Perform a warp-synchronous reduction
 */
template <unsigned int NUM_ELEMENTS>
__device__ __forceinline__ void WarpReduce(
	unsigned int idx, 
	volatile unsigned int *storage, 
	unsigned int partial_reduction) 
{
	storage[idx] = partial_reduction;

	if (NUM_ELEMENTS > 16) storage[idx] = partial_reduction = partial_reduction + storage[idx + 16];
	if (NUM_ELEMENTS > 8) storage[idx] = partial_reduction = partial_reduction + storage[idx + 8];
	if (NUM_ELEMENTS > 4) storage[idx] = partial_reduction = partial_reduction + storage[idx + 4];
	if (NUM_ELEMENTS > 2) storage[idx] = partial_reduction = partial_reduction + storage[idx + 2];
	if (NUM_ELEMENTS > 1) storage[idx] = partial_reduction = partial_reduction + storage[idx + 1];
}


/**
 * Tally a warp-vote regarding the given predicate using the supplied storage
 */
template <unsigned int ACTIVE_THREADS>
__device__ __forceinline__ unsigned int TallyWarpVoteSm10(unsigned int predicate, unsigned int storage[]) {
	WarpReduce<ACTIVE_THREADS>(threadIdx.x, storage, predicate);
	return storage[0];
}


__shared__ unsigned int vote_reduction[WARP_THREADS];

/**
 * Tally a warp-vote regarding the given predicate
 */
template <unsigned int ACTIVE_THREADS>
__device__ __forceinline__ unsigned int TallyWarpVoteSm10(unsigned int predicate) {
	return TallyWarpVoteSm10<ACTIVE_THREADS>(predicate, vote_reduction);
}

/**
 * Emulate the __all() warp vote instruction
 */
template <unsigned int ACTIVE_THREADS>
__device__ __forceinline__ unsigned int EmulatedWarpVoteAll(unsigned int predicate) {
	return (TallyWarpVoteSm10<ACTIVE_THREADS>(predicate) == ACTIVE_THREADS);
}


/**
 * Have each thread concurrently perform a serial reduction over its specified segment 
 */
template <unsigned int LENGTH>
__device__ __forceinline__ unsigned int 
SerialReduce(unsigned int segment[]) {
	
	unsigned int reduce = segment[0];

	#pragma unroll
	for (int i = 1; i < (int) LENGTH; i++) {
		reduce += segment[i];
	}
	
	return reduce;
}


/**
 * Have each thread concurrently perform a serial scan over its specified segment
 */
template <unsigned int LENGTH>
__device__ __forceinline__
void SerialScan(unsigned int segment[], unsigned int seed0) {
	
	unsigned int seed1;

	#pragma unroll	
	for (int i = 0; i < (int) LENGTH; i += 2) {
		seed1 = segment[i] + seed0;
		segment[i] = seed0;
		seed0 = seed1 + segment[i + 1];
		segment[i + 1] = seed1;
	}
}




//------------------------------------------------------------------------------
// Empty Kernels
//------------------------------------------------------------------------------

template <typename T>
__global__ void FlushKernel(void)
{
}

} // namespace b40c

