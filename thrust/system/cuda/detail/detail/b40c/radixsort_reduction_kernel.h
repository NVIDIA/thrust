/******************************************************************************
 * 
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
 * 
 * 
 * 
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Bottom-level digit-reduction/counting kernel
 ******************************************************************************/

#pragma once

#include "radixsort_kernel_common.h"

namespace thrust  {
namespace system  {
namespace cuda    {
namespace detail  {
namespace detail  {
namespace b40c_thrust   {

/******************************************************************************
 * Defines
 ******************************************************************************/

const int BYTE_ENCODE_SHIFT = 0x3;


/******************************************************************************
 * Cycle-processing Routines
 ******************************************************************************/

__device__ __forceinline__ int DecodeInt(int encoded, int quad_byte){
	return (encoded >> quad_byte) & 0xff;		// shift right 8 bits per digit and return rightmost 8 bits
}


__device__ __forceinline__ int EncodeInt(int count, int quad_byte) {
	return count << quad_byte;					// shift left 8 bits per digit
}


template <typename K, long long RADIX_DIGITS, int BIT>
__device__ __forceinline__ void DecodeDigit(
	K key, 
	int &lane, 
	int &quad_shift) 
{
	const K DIGIT_MASK = RADIX_DIGITS - 1;
	lane = (key & (DIGIT_MASK << BIT)) >> (BIT + 2);
	
	const K QUAD_MASK = (RADIX_DIGITS < 4) ? 0x1 : 0x3;
	if (BIT == 32) {
		// N.B.: This takes one more instruction than the code below it, but 
		// otherwise the compiler goes nuts and shoves hundreds of bytes 
		// to lmem when bit = 32 on 64-bit keys.		
		quad_shift = ((key >> BIT) & QUAD_MASK) << BYTE_ENCODE_SHIFT;	
	} else {
		quad_shift = MagnitudeShift<K, BYTE_ENCODE_SHIFT - BIT>(key & (QUAD_MASK << BIT));
	}
}


template <int RADIX_DIGITS, int SCAN_LANES, int LANES_PER_WARP, int BIT, bool FINAL_REDUCE>
__device__ __forceinline__ void ReduceEncodedCounts(
	int local_counts[LANES_PER_WARP][4],
	int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS]) 
{
	const int LOG_PARTIALS_PER_THREAD = B40C_RADIXSORT_LOG_THREADS - B40C_LOG_WARP_THREADS;
	const int PARTIALS_PER_THREAD = 1 << LOG_PARTIALS_PER_THREAD;
	
	int encoded;
	int idx = threadIdx.x & (B40C_WARP_THREADS - 1);
	
	
	__syncthreads();

	#pragma unroll
	for (int j = 0; j < (int) LANES_PER_WARP; j++) {
		
		int warp_id = (threadIdx.x >> B40C_LOG_WARP_THREADS) + (j * B40C_RADIXSORT_WARPS);
		if (warp_id < SCAN_LANES) {

			// rest of my elements
			#pragma unroll
			for (int i = 0; i < (int) PARTIALS_PER_THREAD; i++) {
				encoded = encoded_carry[warp_id][idx + (i * B40C_WARP_THREADS)];		
				local_counts[j][0] += DecodeInt(encoded, 0u << BYTE_ENCODE_SHIFT);
				local_counts[j][1] += DecodeInt(encoded, 1u << BYTE_ENCODE_SHIFT);
				local_counts[j][2] += DecodeInt(encoded, 2u << BYTE_ENCODE_SHIFT);
				local_counts[j][3] += DecodeInt(encoded, 3u << BYTE_ENCODE_SHIFT);
			}
			
			if (FINAL_REDUCE) {
				// reduce all four packed fields, leaving them in the first four elements of our row
				WarpReduce<B40C_WARP_THREADS>(idx, &encoded_carry[warp_id][0], local_counts[j][0]);
				WarpReduce<B40C_WARP_THREADS>(idx, &encoded_carry[warp_id][1], local_counts[j][1]);
				WarpReduce<B40C_WARP_THREADS>(idx, &encoded_carry[warp_id][2], local_counts[j][2]);
				WarpReduce<B40C_WARP_THREADS>(idx, &encoded_carry[warp_id][3], local_counts[j][3]);
			}
		}
	}	

	__syncthreads();
	
}
	

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
__device__ __forceinline__ void Bucket(
	K input, 
	int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS],
	PreprocessFunctor preprocess = PreprocessFunctor()) 
{
	int lane, quad_shift;
	preprocess(input);
	DecodeDigit<K, RADIX_DIGITS, BIT>(input, lane, quad_shift);
	encoded_carry[lane][threadIdx.x] += EncodeInt(1, quad_shift);
}


template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor, int CYCLES>
struct LoadOp;

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 1> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int offset, int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
	{
		K key = d_in_keys[offset + threadIdx.x];
		Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(key, encoded_carry);
	}
};

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 2> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int offset, int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
	{
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 0), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 1), encoded_carry);
	}
};

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 4> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int offset, int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
	{
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 0), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 2), encoded_carry);
	}
};

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 8> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int offset, int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
	{
			K keys[8];
				
			keys[0] = d_in_keys[offset + (B40C_RADIXSORT_THREADS * 0) + threadIdx.x];
			keys[1] = d_in_keys[offset + (B40C_RADIXSORT_THREADS * 1) + threadIdx.x];
			keys[2] = d_in_keys[offset + (B40C_RADIXSORT_THREADS * 2) + threadIdx.x];
			keys[3] = d_in_keys[offset + (B40C_RADIXSORT_THREADS * 3) + threadIdx.x];

			if (B40C_FERMI(__CUDA_ARCH__)) __syncthreads();
			
			keys[4] = d_in_keys[offset + (B40C_RADIXSORT_THREADS * 4) + threadIdx.x];
			keys[5] = d_in_keys[offset + (B40C_RADIXSORT_THREADS * 5) + threadIdx.x];
			keys[6] = d_in_keys[offset + (B40C_RADIXSORT_THREADS * 6) + threadIdx.x];
			keys[7] = d_in_keys[offset + (B40C_RADIXSORT_THREADS * 7) + threadIdx.x];
			
			Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(keys[0], encoded_carry);
			Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(keys[1], encoded_carry);
			Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(keys[2], encoded_carry);
			Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(keys[3], encoded_carry);
			Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(keys[4], encoded_carry);
			Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(keys[5], encoded_carry);
			Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(keys[6], encoded_carry);
			Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(keys[7], encoded_carry);
	}
};

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 16> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int offset, int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
	{
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 0), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 8), encoded_carry);
	}
};

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 32> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int offset, int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
	{
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 0), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 16), encoded_carry);
	}
};

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 64> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int offset, int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
	{
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 0), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 32), encoded_carry);
	}
};

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 128> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int offset, int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
	{
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 0), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 64), encoded_carry);
	}
};

template <typename K, int RADIX_DIGITS, int SCAN_LANES, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 252> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int offset, int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
	{
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 128>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 0), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 128), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 192), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 224), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 240), encoded_carry);
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 4>::BlockOfLoads(d_in_keys, offset + (B40C_RADIXSORT_THREADS * 248), encoded_carry);
	}
};


template <int SCAN_LANES>
__device__ __forceinline__ void ResetEncodedCarry(
	int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS])
{
	#pragma unroll
	for (int SCAN_LANE = 0; SCAN_LANE < (int) SCAN_LANES; SCAN_LANE++) {
		encoded_carry[SCAN_LANE][threadIdx.x] = 0;
	}
}


template <typename K, int RADIX_DIGITS, int SCAN_LANES, int LANES_PER_WARP, int BIT, typename PreprocessFunctor>
__device__ __forceinline__ int ProcessLoads(
	K *d_in_keys,
	int loads,
	int &offset,
	int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS],
	int local_counts[LANES_PER_WARP][4])
{
	// Unroll batches of loads with occasional reduction to avoid overflow
	while (loads >= 32) {
	
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, offset, encoded_carry);
		offset += B40C_RADIXSORT_THREADS * 32;
		loads -= 32;

		// Reduce int local count registers to prevent overflow
		ReduceEncodedCounts<RADIX_DIGITS, SCAN_LANES, LANES_PER_WARP, BIT, false>(
				local_counts, 
				encoded_carry);
		
		// Reset encoded counters
		ResetEncodedCarry<SCAN_LANES>(encoded_carry);
	} 
	
	int retval = loads;
	
	// Wind down loads in decreasing batch sizes

	while (loads >= 4) {
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 4>::BlockOfLoads(d_in_keys, offset, encoded_carry);
		offset += B40C_RADIXSORT_THREADS * 4;
		loads -= 4;
	} 

	while (loads) {
		LoadOp<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, offset, encoded_carry);
		offset += B40C_RADIXSORT_THREADS * 1;
		loads--;
	}
	
	return retval;
}


/******************************************************************************
 * Reduction/counting Kernel Entry Point
 ******************************************************************************/

template <typename K, typename V, int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor>
__launch_bounds__ (B40C_RADIXSORT_THREADS, B40C_RADIXSORT_REDUCE_CTA_OCCUPANCY(__CUDA_ARCH__))
__global__ 
void RakingReduction(
	bool *d_from_alt_storage,
	int *d_spine,
	K *d_in_keys,
	K *d_out_keys,
	CtaDecomposition work_decomposition)
{
	const int RADIX_DIGITS 		= 1 << RADIX_BITS;

	const int LOG_SCAN_LANES 		= (RADIX_BITS >= 2) ? RADIX_BITS - 2 : 0;	// Always at least one fours group
	const int SCAN_LANES 			= 1 << LOG_SCAN_LANES;

	const int LOG_LANES_PER_WARP 	= (SCAN_LANES > B40C_RADIXSORT_WARPS) ? LOG_SCAN_LANES - B40C_RADIXSORT_LOG_WARPS : 0;	// Always at least one fours group per warp
	const int LANES_PER_WARP 		= 1 << LOG_LANES_PER_WARP;
	
	
	// Each thread gets its own column of fours-groups (for conflict-free updates)
	__shared__ int encoded_carry[SCAN_LANES][B40C_RADIXSORT_THREADS];			

	// Each thread is also responsible for aggregating an unencoded segment of a fours-group
	int local_counts[LANES_PER_WARP][4];								

	// Determine where to read our input
	bool from_alt_storage = (PASS == 0) ? false : d_from_alt_storage[PASS & 0x1];
	if (from_alt_storage) d_in_keys = d_out_keys;
	
	// Calculate our threadblock's range
	int offset, block_elements;
	if (blockIdx.x < work_decomposition.num_big_blocks) {
		offset = work_decomposition.big_block_elements * blockIdx.x;
		block_elements = work_decomposition.big_block_elements;
	} else {
		offset = (work_decomposition.normal_block_elements * blockIdx.x) + (work_decomposition.num_big_blocks * B40C_RADIXSORT_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V));
		block_elements = work_decomposition.normal_block_elements;
	}
	
	// Initialize local counts
	#pragma unroll 
	for (int LANE = 0; LANE < (int) LANES_PER_WARP; LANE++) {
		local_counts[LANE][0] = 0;
		local_counts[LANE][1] = 0;
		local_counts[LANE][2] = 0;
		local_counts[LANE][3] = 0;
	}
	
	// Reset encoded counters
	ResetEncodedCarry<SCAN_LANES>(encoded_carry);
	
	// Process loads
	int loads = block_elements >> B40C_RADIXSORT_LOG_THREADS;
	int unreduced_loads = ProcessLoads<K, RADIX_DIGITS, SCAN_LANES, LANES_PER_WARP, BIT, PreprocessFunctor>(
		d_in_keys,
		loads,
		offset,
		encoded_carry,
		local_counts);
	
	// Cleanup if we're the last block  
	if ((blockIdx.x == gridDim.x - 1) && (work_decomposition.extra_elements_last_block)) {

		const int LOADS_PER_CYCLE = B40C_RADIXSORT_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V) / B40C_RADIXSORT_THREADS;
		
		// If extra guarded loads may cause overflow, reduce now and reset counters
		if (unreduced_loads + LOADS_PER_CYCLE > 255) {
		
			ReduceEncodedCounts<RADIX_DIGITS, SCAN_LANES, LANES_PER_WARP, BIT, false>(
					local_counts, 
					encoded_carry);
			
			ResetEncodedCarry<SCAN_LANES>(encoded_carry);
		}
		
		// perform up to LOADS_PER_CYCLE extra guarded loads
		#pragma unroll
		for (int EXTRA_LOAD = 0; EXTRA_LOAD < (int) LOADS_PER_CYCLE; EXTRA_LOAD++) {
			if (threadIdx.x + (B40C_RADIXSORT_THREADS * EXTRA_LOAD) < work_decomposition.extra_elements_last_block) {
				K key = d_in_keys[offset + (B40C_RADIXSORT_THREADS * EXTRA_LOAD) + threadIdx.x];
				Bucket<K, RADIX_DIGITS, SCAN_LANES, BIT, PreprocessFunctor>(key, encoded_carry);
			}
		}
	}
	
	// Aggregate 
	ReduceEncodedCounts<RADIX_DIGITS, SCAN_LANES, LANES_PER_WARP, BIT, true>(
		local_counts, 
		encoded_carry);

	// Write carry in parallel (carries per row are in the first four bytes of each row) 
	if (threadIdx.x < RADIX_DIGITS) {

		int row = threadIdx.x >> 2;		
		int col = threadIdx.x & 3;			 
		d_spine[(gridDim.x * threadIdx.x) + blockIdx.x] = encoded_carry[row][col];
	}
} 

} // end namespace b40c_thrust
} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

