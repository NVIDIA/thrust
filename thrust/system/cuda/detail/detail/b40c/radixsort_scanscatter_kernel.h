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
// Bottom-level digit scanning/scattering kernel
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
 * Appropriate substitutes to use for out-of-bounds key (and value) offsets 
 ******************************************************************************/

template <typename T> 
__device__ __forceinline__ T DefaultextraValue() {
	return T();
}

template <> 
__device__ __forceinline__ unsigned char DefaultextraValue<unsigned char>() {
	return (unsigned char) -1;
}

template <> 
__device__ __forceinline__ unsigned short DefaultextraValue<unsigned short>() {
	return (unsigned short) -1;
}

template <> 
__device__ __forceinline__ unsigned int DefaultextraValue<unsigned int>() {
	return (unsigned int) -1u;
}

template <> 
__device__ __forceinline__ unsigned long DefaultextraValue<unsigned long>() {
	return (unsigned long) -1ul;
}

template <> 
__device__ __forceinline__ unsigned long long DefaultextraValue<unsigned long long>() {
	return (unsigned long long) -1ull;
}


/******************************************************************************
 * Cycle-processing Routines
 ******************************************************************************/

template <typename K, long long RADIX_DIGITS, int BIT>
__device__ __forceinline__ int DecodeDigit(K key) 
{
	const K DIGIT_MASK = RADIX_DIGITS - 1;
	return (key >> BIT) & DIGIT_MASK;
}


template <typename K, long long RADIX_DIGITS, int BIT, int PADDED_PARTIALS_PER_LANE>
__device__ __forceinline__ void DecodeDigit(
	K key, 
	int &digit, 
	int &flag_offset,		// in bytes
	const int SET_OFFSET)
{
	const int PADDED_BYTES_PER_LANE 	= PADDED_PARTIALS_PER_LANE * 4;
	const int SET_OFFSET_BYTES 		= SET_OFFSET * 4;
	const K QUAD_MASK 							= (RADIX_DIGITS < 4) ? 0x1 : 0x3;
	
	digit = DecodeDigit<K, RADIX_DIGITS, BIT>(key);
	int lane = digit >> 2;
	int quad_byte = digit & QUAD_MASK;

	flag_offset = SET_OFFSET_BYTES + FastMul(lane, PADDED_BYTES_PER_LANE) + quad_byte;
}


template <typename K, long long RADIX_DIGITS, int BIT, int SETS_PER_PASS, int SCAN_LANES_PER_SET, int PADDED_PARTIALS_PER_LANE>
__device__ __forceinline__ void DecodeDigits(
	typename VecType<K, 2>::Type keypairs[SETS_PER_PASS],
	int2 digits[SETS_PER_PASS],
	int2 flag_offsets[SETS_PER_PASS])		// in bytes 
{

	#pragma unroll
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		
		const int SET_OFFSET = SET * SCAN_LANES_PER_SET * PADDED_PARTIALS_PER_LANE;

		DecodeDigit<K, RADIX_DIGITS, BIT, PADDED_PARTIALS_PER_LANE>(
				keypairs[SET].x, digits[SET].x, flag_offsets[SET].x, SET_OFFSET);
		
		DecodeDigit<K, RADIX_DIGITS, BIT, PADDED_PARTIALS_PER_LANE>(
				keypairs[SET].y, digits[SET].y, flag_offsets[SET].y, SET_OFFSET);
	}
}


template <typename T, typename PreprocessFunctor>
__device__ __forceinline__ void GuardedReadSet(
	T *in, 
	typename VecType<T, 2>::Type &pair,
	int offset,
	int extra[1],
	PreprocessFunctor preprocess = PreprocessFunctor())				
{
	if (offset - extra[0] < 0) {
		pair.x = in[offset];
		preprocess(pair.x);
	} else {
		pair.x = DefaultextraValue<T>();
	}
	
	if (offset + 1 - extra[0] < 0) {
		pair.y = in[offset + 1];
		preprocess(pair.y);
	} else {
		pair.y = DefaultextraValue<T>();
	}
}


template <typename T, bool UNGUARDED_IO, int SETS_PER_PASS, typename PreprocessFunctor>
__device__ __forceinline__ void ReadSets(
	typename VecType<T, 2>::Type *d_in, 
	typename VecType<T, 2>::Type pairs[SETS_PER_PASS],
	const int BASE2,
	int extra[1],
	PreprocessFunctor preprocess = PreprocessFunctor())				
{
	if (UNGUARDED_IO) {

		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler makes it 1% slower
		if (SETS_PER_PASS > 0) pairs[0] = d_in[threadIdx.x + BASE2 + (B40C_RADIXSORT_THREADS * 0)];
		if (SETS_PER_PASS > 1) pairs[1] = d_in[threadIdx.x + BASE2 + (B40C_RADIXSORT_THREADS * 1)];
		if (SETS_PER_PASS > 2) pairs[2] = d_in[threadIdx.x + BASE2 + (B40C_RADIXSORT_THREADS * 2)];
		if (SETS_PER_PASS > 3) pairs[3] = d_in[threadIdx.x + BASE2 + (B40C_RADIXSORT_THREADS * 3)];

		#pragma unroll 
		for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
			preprocess(pairs[SET].x);
			preprocess(pairs[SET].y);
		}
		
	} else {

		T* in = (T*) d_in;
		
		// N.B. --  I wish we could do some pragma unrolling here, but the compiler won't let 
		// us with user-defined value types (e.g., Fribbitz): "Advisory: Loop was not unrolled, cannot deduce loop trip count"
		
		if (SETS_PER_PASS > 0) GuardedReadSet<T, PreprocessFunctor>(in, pairs[0], (threadIdx.x << 1) + (BASE2 << 1) + (B40C_RADIXSORT_THREADS * 2 * 0), extra);
		if (SETS_PER_PASS > 1) GuardedReadSet<T, PreprocessFunctor>(in, pairs[1], (threadIdx.x << 1) + (BASE2 << 1) + (B40C_RADIXSORT_THREADS * 2 * 1), extra);
		if (SETS_PER_PASS > 2) GuardedReadSet<T, PreprocessFunctor>(in, pairs[2], (threadIdx.x << 1) + (BASE2 << 1) + (B40C_RADIXSORT_THREADS * 2 * 2), extra);
		if (SETS_PER_PASS > 3) GuardedReadSet<T, PreprocessFunctor>(in, pairs[3], (threadIdx.x << 1) + (BASE2 << 1) + (B40C_RADIXSORT_THREADS * 2 * 3), extra);
	}
}


template <int SETS_PER_PASS>
__device__ __forceinline__ void PlacePartials(
	unsigned char * base_partial,
	int2 digits[SETS_PER_PASS],
	int2 flag_offsets[SETS_PER_PASS]) 
{
	#pragma unroll
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		base_partial[flag_offsets[SET].x] = 1;
		base_partial[flag_offsets[SET].y] = 1 + (digits[SET].x == digits[SET].y);
	}
}


template <int SETS_PER_PASS>
__device__ __forceinline__ void ExtractRanks(
	unsigned char * base_partial,
	int2 digits[SETS_PER_PASS],
	int2 flag_offsets[SETS_PER_PASS],
	int2 ranks[SETS_PER_PASS]) 
{
	#pragma unroll
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		ranks[SET].x = base_partial[flag_offsets[SET].x];
		ranks[SET].y = base_partial[flag_offsets[SET].y] + (digits[SET].x == digits[SET].y);
	}
}


template <int RADIX_DIGITS, int SETS_PER_PASS>
__device__ __forceinline__ void UpdateRanks(
	int2 digits[SETS_PER_PASS],
	int2 ranks[SETS_PER_PASS],
	int digit_counts[SETS_PER_PASS][RADIX_DIGITS])
{
	// N.B.: I wish we could pragma unroll here, but doing so currently 
	// results in the 3.1 compilier on 64-bit platforms generating bad
	// code for SM1.3, resulting in incorrect sorting (e.g., problem size 16)
	
	if (SETS_PER_PASS > 0) {
		ranks[0].x += digit_counts[0][digits[0].x];
		ranks[0].y += digit_counts[0][digits[0].y]; 
	}	
	if (SETS_PER_PASS > 1) {
		ranks[1].x += digit_counts[1][digits[1].x];
		ranks[1].y += digit_counts[1][digits[1].y]; 
	}	
	if (SETS_PER_PASS > 2) {
		ranks[2].x += digit_counts[2][digits[2].x];
		ranks[2].y += digit_counts[2][digits[2].y]; 
	}	
	if (SETS_PER_PASS > 3) {
		ranks[3].x += digit_counts[3][digits[3].x];
		ranks[3].y += digit_counts[3][digits[3].y]; 
	}	
}

template <int RADIX_DIGITS, int PASSES_PER_CYCLE, int SETS_PER_PASS>
__device__ __forceinline__ void UpdateRanks(
	int2 digits[PASSES_PER_CYCLE][SETS_PER_PASS],
	int2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS],
	int digit_counts[PASSES_PER_CYCLE][SETS_PER_PASS][RADIX_DIGITS])
{
	// N.B.: I wish we could pragma unroll here, but doing so currently 
	// results in the 3.1 compilier on 64-bit platforms generating bad
	// code for SM1.3, resulting in incorrect sorting (e.g., problem size 16)
	
	if (PASSES_PER_CYCLE > 0) UpdateRanks<RADIX_DIGITS, SETS_PER_PASS>(digits[0], ranks[0], digit_counts[0]);
	if (PASSES_PER_CYCLE > 1) UpdateRanks<RADIX_DIGITS, SETS_PER_PASS>(digits[1], ranks[1], digit_counts[1]);
	if (PASSES_PER_CYCLE > 2) UpdateRanks<RADIX_DIGITS, SETS_PER_PASS>(digits[2], ranks[2], digit_counts[2]);
	if (PASSES_PER_CYCLE > 3) UpdateRanks<RADIX_DIGITS, SETS_PER_PASS>(digits[3], ranks[3], digit_counts[3]);
}



template <int SCAN_LANES_PER_PASS, int LOG_RAKING_THREADS_PER_LANE, int RAKING_THREADS_PER_LANE, int PARTIALS_PER_SEG>
__device__ __forceinline__ void PrefixScanOverLanes(
	int 	raking_segment[],
	int 	warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE],
	int 	copy_section)
{
	// Upsweep rake
	int partial_reduction = SerialReduce<PARTIALS_PER_SEG>(raking_segment);

	// Warpscan reduction in digit warpscan_lane
	int warpscan_lane = threadIdx.x >> LOG_RAKING_THREADS_PER_LANE;
	int group_prefix = WarpScan<RAKING_THREADS_PER_LANE, true>(
		warpscan[warpscan_lane], 
		partial_reduction,
		copy_section);

	// Downsweep rake
	SerialScan<PARTIALS_PER_SEG>(raking_segment, group_prefix);
	
}


template <int SCAN_LANES_PER_PASS, int RAKING_THREADS_PER_LANE, int SETS_PER_PASS, int SCAN_LANES_PER_SET>
__device__ __forceinline__ void RecoverDigitCounts(
	int warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE],
	int counts[SETS_PER_PASS],
	int copy_section)
{
	int my_lane = threadIdx.x >> 2;
	int my_quad_byte = threadIdx.x & 3;
	
	#pragma unroll
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		unsigned char *warpscan_count = (unsigned char *) &warpscan[my_lane + (SCAN_LANES_PER_SET * SET)][1 + copy_section][RAKING_THREADS_PER_LANE - 1];
		counts[SET] = warpscan_count[my_quad_byte];
	}
}

template<int RADIX_DIGITS>
__device__ __forceinline__ void CorrectUnguardedSetOverflow(
	int2 			set_digits,
	int 	&set_count)				
{
	if (WarpVoteAll(RADIX_DIGITS, set_count <= 1)) {
		// All first-pass, first set keys have same digit. 
		set_count = (threadIdx.x == set_digits.x) ? 256 : 0;
	}
}

template <int RADIX_DIGITS, int SETS_PER_PASS>
__device__ __forceinline__ void CorrectUnguardedPassOverflow(
	int2 			pass_digits[SETS_PER_PASS],
	int 	pass_counts[SETS_PER_PASS])				
{
	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, unexpected call OPs"

	if (SETS_PER_PASS > 0) CorrectUnguardedSetOverflow<RADIX_DIGITS>(pass_digits[0], pass_counts[0]);
	if (SETS_PER_PASS > 1) CorrectUnguardedSetOverflow<RADIX_DIGITS>(pass_digits[1], pass_counts[1]);
	if (SETS_PER_PASS > 2) CorrectUnguardedSetOverflow<RADIX_DIGITS>(pass_digits[2], pass_counts[2]);
	if (SETS_PER_PASS > 3) CorrectUnguardedSetOverflow<RADIX_DIGITS>(pass_digits[3], pass_counts[3]);
}


template <int RADIX_DIGITS, int PASSES_PER_CYCLE, int SETS_PER_PASS>
__device__ __forceinline__ void CorrectUnguardedCycleOverflow(
	int2 			cycle_digits[PASSES_PER_CYCLE][SETS_PER_PASS],
	int 	cycle_counts[PASSES_PER_CYCLE][SETS_PER_PASS])
{
	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, unexpected call OPs"

	if (PASSES_PER_CYCLE > 0) CorrectUnguardedPassOverflow<RADIX_DIGITS, SETS_PER_PASS>(cycle_digits[0], cycle_counts[0]);
	if (PASSES_PER_CYCLE > 1) CorrectUnguardedPassOverflow<RADIX_DIGITS, SETS_PER_PASS>(cycle_digits[1], cycle_counts[1]);
}


template <int RADIX_DIGITS>
__device__ __forceinline__ void CorrectLastLaneOverflow(int &count, int extra[1]) 
{
	if (WarpVoteAll(RADIX_DIGITS, count == 0) && (threadIdx.x == RADIX_DIGITS - 1)) {
		// We're 'f' and we overflowed b/c of invalid 'f' placemarkers; the number of valid items in this set is the count of valid f's 
		count = extra[0] & 255;
	}
}
		

template <int RADIX_DIGITS, int PASSES_PER_CYCLE, int SETS_PER_PASS, int SETS_PER_CYCLE, bool UNGUARDED_IO>
__device__ __forceinline__ void CorrectForOverflows(
	int2 digits[PASSES_PER_CYCLE][SETS_PER_PASS],
	int counts[PASSES_PER_CYCLE][SETS_PER_PASS], 
	int extra[1])				
{
	if (!UNGUARDED_IO) {

		// Correct any overflow in the partially-filled last lane
		int *linear_counts = (int *) counts;
		CorrectLastLaneOverflow<RADIX_DIGITS>(linear_counts[SETS_PER_CYCLE - 1], extra);
	}

	CorrectUnguardedCycleOverflow<RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS>(digits, counts);
}


template <
	typename K,
	int BIT, 
	int RADIX_DIGITS,
	int SCAN_LANES_PER_SET,
	int SETS_PER_PASS,
	int RAKING_THREADS_PER_PASS,
	int SCAN_LANES_PER_PASS,
	int LOG_RAKING_THREADS_PER_LANE,
	int RAKING_THREADS_PER_LANE,
	int PARTIALS_PER_SEG,
	int PADDED_PARTIALS_PER_LANE,
	int PASSES_PER_CYCLE>
__device__ __forceinline__ void ScanPass(
	int *base_partial,
	int	*raking_partial,
	int warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE],
	typename VecType<K, 2>::Type keypairs[SETS_PER_PASS],
	int2 digits[SETS_PER_PASS],
	int2 flag_offsets[SETS_PER_PASS],
	int2 ranks[SETS_PER_PASS],
	int copy_section)
{
	// Reset smem
	#pragma unroll
	for (int SCAN_LANE = 0; SCAN_LANE < (int) SCAN_LANES_PER_PASS; SCAN_LANE++) {
		base_partial[SCAN_LANE * PADDED_PARTIALS_PER_LANE] = 0;
	}
	
	// Decode digits for first pass
	DecodeDigits<K, RADIX_DIGITS, BIT, SETS_PER_PASS, SCAN_LANES_PER_SET, PADDED_PARTIALS_PER_LANE>(
		keypairs, digits, flag_offsets);
	
	// Encode counts into smem for first pass
	PlacePartials<SETS_PER_PASS>(
		(unsigned char *) base_partial,
		digits,
		flag_offsets); 
	
	__syncthreads();
	
	// Intra-group prefix scans for first pass
	if (threadIdx.x < RAKING_THREADS_PER_PASS) {
	
		PrefixScanOverLanes<SCAN_LANES_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG>(		// first pass is offset right by one
			raking_partial,
			warpscan, 
			copy_section);
	}
	
	__syncthreads();

	// Extract ranks
	ExtractRanks<SETS_PER_PASS>(
		(unsigned char *) base_partial, 
		digits, 
		flag_offsets, 
		ranks); 	
}	
	

/******************************************************************************
 * SM1.3 Local Exchange Routines
 * 
 * Routines for exchanging keys (and values) in shared memory (i.e., local 
 * scattering) in order to to facilitate coalesced global scattering
 ******************************************************************************/

template <typename T, bool UNGUARDED_IO, int PASSES_PER_CYCLE, int SETS_PER_PASS, typename PostprocessFunctor>
__device__ __forceinline__ void ScatterSets(
	T *d_out, 
	typename VecType<T, 2>::Type pairs[SETS_PER_PASS],
	int2 offsets[SETS_PER_PASS],
	const int BASE4,
	int extra[1],
	PostprocessFunctor postprocess = PostprocessFunctor())				
{
	#pragma unroll 
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		postprocess(pairs[SET].x);
		postprocess(pairs[SET].y);
	}

	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler makes it 1% slower 
		
	if (SETS_PER_PASS > 0) { 
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * 0) < extra[0])) 
			d_out[offsets[0].x] = pairs[0].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * 1) < extra[0])) 
			d_out[offsets[0].y] = pairs[0].y;
	}

	if (SETS_PER_PASS > 1) { 
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * 2) < extra[0])) 
			d_out[offsets[1].x] = pairs[1].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * 3) < extra[0])) 
			d_out[offsets[1].y] = pairs[1].y;
	}

	if (SETS_PER_PASS > 2) { 
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * 4) < extra[0])) 
			d_out[offsets[2].x] = pairs[2].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * 5) < extra[0])) 
			d_out[offsets[2].y] = pairs[2].y;
	}

	if (SETS_PER_PASS > 3) { 
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * 6) < extra[0])) 
			d_out[offsets[3].x] = pairs[3].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * 7) < extra[0])) 
			d_out[offsets[3].y] = pairs[3].y;
	}
}

template <typename T, int PASSES_PER_CYCLE, int SETS_PER_PASS>
__device__ __forceinline__ void PushPairs(
	T *swap, 
	typename VecType<T, 2>::Type pairs[PASSES_PER_CYCLE][SETS_PER_PASS],
	int2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS])				
{
	#pragma unroll 
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
	
		#pragma unroll 
		for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
			swap[ranks[PASS][SET].x] = pairs[PASS][SET].x;
			swap[ranks[PASS][SET].y] = pairs[PASS][SET].y;
		}
	}
}
	
template <typename T, int PASSES_PER_CYCLE, int SETS_PER_PASS>
__device__ __forceinline__ void ExchangePairs(
	T *swap, 
	typename VecType<T, 2>::Type pairs[PASSES_PER_CYCLE][SETS_PER_PASS],
	int2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS])				
{
	// Push in Pairs
	PushPairs<T, PASSES_PER_CYCLE, SETS_PER_PASS>(swap, pairs, ranks);
	
	__syncthreads();
	
	// Extract pairs
	#pragma unroll 
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
		
		#pragma unroll 
		for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
			const int BLOCK = ((PASS * SETS_PER_PASS) + SET) * 2;
			pairs[PASS][SET].x = swap[threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0))];
			pairs[PASS][SET].y = swap[threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1))];
		}
	}
}


template <
	typename K,
	typename V,	
	int RADIX_DIGITS, 
	int BIT, 
	int PASSES_PER_CYCLE,
	int SETS_PER_PASS,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
__device__ __forceinline__ void SwapAndScatterSm13(
	typename VecType<K, 2>::Type keypairs[PASSES_PER_CYCLE][SETS_PER_PASS], 
	int2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS],
	int4 *exchange,
	typename VecType<V, 2>::Type *d_in_values, 
	K *d_out_keys, 
	V *d_out_values, 
	int carry[RADIX_DIGITS], 
	int extra[1])				
{
	int2 offsets[PASSES_PER_CYCLE][SETS_PER_PASS];
	
	// Swap keys according to ranks
	ExchangePairs<K, PASSES_PER_CYCLE, SETS_PER_PASS>((K*) exchange, keypairs, ranks);				
	
	// Calculate scatter offsets (re-decode digits from keys: it's less work than making a second exchange of digits) 
	#pragma unroll 
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
		
		#pragma unroll 
		for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
			const int BLOCK = ((PASS * SETS_PER_PASS) + SET) * 2;
			offsets[PASS][SET].x = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0)) + carry[DecodeDigit<K, RADIX_DIGITS, BIT>(keypairs[PASS][SET].x)];
			offsets[PASS][SET].y = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1)) + carry[DecodeDigit<K, RADIX_DIGITS, BIT>(keypairs[PASS][SET].y)];
		}
	}
	
	// Scatter keys
	#pragma unroll 
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
		const int BLOCK = PASS * SETS_PER_PASS * 2;
		ScatterSets<K, UNGUARDED_IO, PASSES_PER_CYCLE, SETS_PER_PASS, PostprocessFunctor>(d_out_keys, keypairs[PASS], offsets[PASS], B40C_RADIXSORT_THREADS * BLOCK, extra);
	}

	if (!IsKeysOnly<V>()) {
	
		__syncthreads();

		// Read input data
		typename VecType<V, 2>::Type datapairs[PASSES_PER_CYCLE][SETS_PER_PASS];

		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
		// telling me "Advisory: Loop was not unrolled, unexpected control flow"

		if (PASSES_PER_CYCLE > 0) ReadSets<V, UNGUARDED_IO, SETS_PER_PASS, NopFunctor<V> >(d_in_values, datapairs[0], B40C_RADIXSORT_THREADS * SETS_PER_PASS * 0, extra);
		if (PASSES_PER_CYCLE > 1) ReadSets<V, UNGUARDED_IO, SETS_PER_PASS, NopFunctor<V> >(d_in_values, datapairs[1], B40C_RADIXSORT_THREADS * SETS_PER_PASS * 1, extra);
		
		// Swap data according to ranks
		ExchangePairs<V, PASSES_PER_CYCLE, SETS_PER_PASS>((V*) exchange, datapairs, ranks);
		
		// Scatter data
		#pragma unroll 
		for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
			const int BLOCK = PASS * SETS_PER_PASS * 2;
			ScatterSets<V, UNGUARDED_IO, PASSES_PER_CYCLE, SETS_PER_PASS, NopFunctor<V> >(d_out_values, datapairs[PASS], offsets[PASS], B40C_RADIXSORT_THREADS * BLOCK, extra);
		}
	}
}


/******************************************************************************
 * SM1.0 Local Exchange Routines
 *
 * Routines for exchanging keys (and values) in shared memory (i.e., local 
 * scattering) in order to to facilitate coalesced global scattering
 ******************************************************************************/

template <
	typename T, 
	int RADIX_DIGITS,
	bool UNGUARDED_IO,
	typename PostprocessFunctor> 
__device__ __forceinline__ void ScatterPass(
	T *swapmem,
	T *d_out, 
	int digit_scan[2][RADIX_DIGITS], 
	int carry[RADIX_DIGITS], 
	int extra[1],
	int base_digit,				
	PostprocessFunctor postprocess = PostprocessFunctor())				
{
	const int LOG_STORE_TXN_THREADS = B40C_LOG_MEM_BANKS(__CUDA_ARCH__);
	const int STORE_TXN_THREADS = 1 << LOG_STORE_TXN_THREADS;
	
	int store_txn_idx = threadIdx.x & (STORE_TXN_THREADS - 1);
	int store_txn_digit = threadIdx.x >> LOG_STORE_TXN_THREADS;
	
	int my_digit = base_digit + store_txn_digit;
	if (my_digit < RADIX_DIGITS) {
	
		int my_exclusive_scan = digit_scan[1][my_digit - 1];
		int my_inclusive_scan = digit_scan[1][my_digit];
		int my_digit_count = my_inclusive_scan - my_exclusive_scan;

		int my_carry = carry[my_digit] + my_exclusive_scan;
		int my_aligned_offset = store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));
		
		while (my_aligned_offset < my_digit_count) {

			if ((my_aligned_offset >= 0) && (UNGUARDED_IO || (my_exclusive_scan + my_aligned_offset < extra[0]))) { 
			
				T datum = swapmem[my_exclusive_scan + my_aligned_offset];
				postprocess(datum);
				d_out[my_carry + my_aligned_offset] = datum;
			}
			my_aligned_offset += STORE_TXN_THREADS;
		}
	}
}

template <
	typename T,
	int RADIX_DIGITS, 
	int PASSES_PER_CYCLE,
	int SETS_PER_PASS,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
__device__ __forceinline__ void SwapAndScatterPairs(
	typename VecType<T, 2>::Type pairs[PASSES_PER_CYCLE][SETS_PER_PASS], 
	int2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS],
	T *exchange,
	T *d_out, 
	int carry[RADIX_DIGITS], 
	int digit_scan[2][RADIX_DIGITS], 
	int extra[1])				
{
	const int SCATTER_PASS_DIGITS = B40C_RADIXSORT_WARPS * (B40C_WARP_THREADS / B40C_MEM_BANKS(__CUDA_ARCH__));
	const int SCATTER_PASSES = RADIX_DIGITS / SCATTER_PASS_DIGITS;

	// Push in pairs
	PushPairs<T, PASSES_PER_CYCLE, SETS_PER_PASS>(exchange, pairs, ranks);

	__syncthreads();

	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, not an innermost loop"

	if (SCATTER_PASSES > 0) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 0);
	if (SCATTER_PASSES > 1) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 1);
	if (SCATTER_PASSES > 2) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 2);
	if (SCATTER_PASSES > 3) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 3);
	if (SCATTER_PASSES > 4) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 4);
	if (SCATTER_PASSES > 5) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 5);
	if (SCATTER_PASSES > 6) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 6);
	if (SCATTER_PASSES > 7) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 7);
}


template <
	typename K,
	typename V,	
	int RADIX_DIGITS, 
	int PASSES_PER_CYCLE,
	int SETS_PER_PASS,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
__device__ __forceinline__ void SwapAndScatterSm10(
	typename VecType<K, 2>::Type keypairs[PASSES_PER_CYCLE][SETS_PER_PASS], 
	int2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS],
	int4 *exchange,
	typename VecType<V, 2>::Type *d_in_values, 
	K *d_out_keys, 
	V *d_out_values, 
	int carry[RADIX_DIGITS], 
	int digit_scan[2][RADIX_DIGITS], 
	int extra[1])				
{
	// Swap and scatter keys
	SwapAndScatterPairs<K, RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, ranks, (K*) exchange, d_out_keys, carry, digit_scan, extra);				
	
	if (!IsKeysOnly<V>()) {

		__syncthreads();
		
		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
		// telling me "Advisory: Loop was not unrolled, unexpected control flow"

		// Read input data
		typename VecType<V, 2>::Type datapairs[PASSES_PER_CYCLE][SETS_PER_PASS];
		if (PASSES_PER_CYCLE > 0) ReadSets<V, UNGUARDED_IO, SETS_PER_PASS, NopFunctor<V> >(d_in_values, datapairs[0], B40C_RADIXSORT_THREADS * SETS_PER_PASS * 0, extra);
		if (PASSES_PER_CYCLE > 1) ReadSets<V, UNGUARDED_IO, SETS_PER_PASS, NopFunctor<V> >(d_in_values, datapairs[1], B40C_RADIXSORT_THREADS * SETS_PER_PASS * 1, extra);

		// Swap and scatter data
		SwapAndScatterPairs<V, RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS, UNGUARDED_IO, NopFunctor<V> >(
			datapairs, ranks, (V*) exchange, d_out_values, carry, digit_scan, extra);				
	}
}


/******************************************************************************
 * Cycle of RADIXSORT_CYCLE_ELEMENTS keys (and values)
 ******************************************************************************/

template <
	typename K,
	typename V,	
	int BIT, 
	bool UNGUARDED_IO,
	int RADIX_DIGITS,
	int LOG_SCAN_LANES_PER_SET,
	int SCAN_LANES_PER_SET,
	int SETS_PER_PASS,
	int PASSES_PER_CYCLE,
	int LOG_SCAN_LANES_PER_PASS,
	int SCAN_LANES_PER_PASS,
	int LOG_PARTIALS_PER_LANE,
	int LOG_PARTIALS_PER_PASS,
	int LOG_RAKING_THREADS_PER_PASS,
	int RAKING_THREADS_PER_PASS,
	int LOG_RAKING_THREADS_PER_LANE,
	int RAKING_THREADS_PER_LANE,
	int LOG_PARTIALS_PER_SEG,
	int PARTIALS_PER_SEG,
	int LOG_PARTIALS_PER_ROW,
	int PARTIALS_PER_ROW,
	int LOG_SEGS_PER_ROW,	
	int SEGS_PER_ROW,
	int LOG_ROWS_PER_SET,
	int LOG_ROWS_PER_LANE,
	int ROWS_PER_LANE,
	int LOG_ROWS_PER_PASS,
	int ROWS_PER_PASS,
	int MAX_EXCHANGE_BYTES,
	typename PreprocessFunctor,
	typename PostprocessFunctor>

__device__ __forceinline__ void SrtsScanDigitCycle(
	typename VecType<K, 2>::Type *d_in_keys, 
	typename VecType<V, 2>::Type *d_in_values, 
	K *d_out_keys, 
	V *d_out_values, 
	int4 *exchange,								
	int	warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE],
	int	carry[RADIX_DIGITS],
	int	digit_scan[2][RADIX_DIGITS],						 
	int	digit_counts[PASSES_PER_CYCLE][SETS_PER_PASS][RADIX_DIGITS],
	int	extra[1],
	int	*base_partial,
	int	*raking_partial)		
{
	
	const int PADDED_PARTIALS_PER_LANE 		= ROWS_PER_LANE * (PARTIALS_PER_ROW + 1);	 
	const int SETS_PER_CYCLE 				= PASSES_PER_CYCLE * SETS_PER_PASS;

	// N.B.: We use the following voodoo incantations to elide the compiler's miserable 
	// "declared but never referenced" warnings for these (which are actually used for 
	// template instantiation)	
	SuppressUnusedConstantWarning(PADDED_PARTIALS_PER_LANE);
	SuppressUnusedConstantWarning(SETS_PER_CYCLE);
	
	typename VecType<K, 2>::Type 	keypairs[PASSES_PER_CYCLE][SETS_PER_PASS];
	int2 							digits[PASSES_PER_CYCLE][SETS_PER_PASS];
	int2 							flag_offsets[PASSES_PER_CYCLE][SETS_PER_PASS];		// a byte offset
	int2 							ranks[PASSES_PER_CYCLE][SETS_PER_PASS];

	
	//-------------------------------------------------------------------------
	// Read keys
	//-------------------------------------------------------------------------

	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, unexpected control flow construct"
	
	// Read Keys
	if (PASSES_PER_CYCLE > 0) ReadSets<K, UNGUARDED_IO, SETS_PER_PASS, PreprocessFunctor>(d_in_keys, keypairs[0], B40C_RADIXSORT_THREADS * SETS_PER_PASS * 0, extra);		 
	if (PASSES_PER_CYCLE > 1) ReadSets<K, UNGUARDED_IO, SETS_PER_PASS, PreprocessFunctor>(d_in_keys, keypairs[1], B40C_RADIXSORT_THREADS * SETS_PER_PASS * 1, extra); 	
	
	//-------------------------------------------------------------------------
	// Lane-scanning Passes
	//-------------------------------------------------------------------------

	#pragma unroll
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
	
		// First Pass
		ScanPass<K, BIT, RADIX_DIGITS, SCAN_LANES_PER_SET, SETS_PER_PASS, RAKING_THREADS_PER_PASS, SCAN_LANES_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PADDED_PARTIALS_PER_LANE, PASSES_PER_CYCLE>(
			base_partial,
			raking_partial,
			warpscan,
			keypairs[PASS],
			digits[PASS],
			flag_offsets[PASS],
			ranks[PASS],
			PASSES_PER_CYCLE - PASS - 1);		// lower passes get copied right
	}
	
	//-------------------------------------------------------------------------
	// Digit-scanning 
	//-------------------------------------------------------------------------

	// Recover second-half digit-counts, scan across all digit-counts
	if (threadIdx.x < RADIX_DIGITS) {

		int counts[PASSES_PER_CYCLE][SETS_PER_PASS];

		// Recover digit-counts

		#pragma unroll
		for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
			RecoverDigitCounts<SCAN_LANES_PER_PASS, RAKING_THREADS_PER_LANE, SETS_PER_PASS, SCAN_LANES_PER_SET>(		// first pass, offset by 1			
				warpscan, 
				counts[PASS],
				PASSES_PER_CYCLE - PASS - 1);		// lower passes get copied right
		}
		
		// Check for overflows
		CorrectForOverflows<RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS, SETS_PER_CYCLE, UNGUARDED_IO>(
				digits, counts, extra);

		// Scan across my digit counts for each set 
		int exclusive_total = 0;
		int inclusive_total = 0;
		
		#pragma unroll
		for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
		
			#pragma unroll
			for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
				inclusive_total += counts[PASS][SET];
				counts[PASS][SET] = exclusive_total;
				exclusive_total = inclusive_total;
			}
		}

		// second half of carry update
		int my_carry = carry[threadIdx.x] + digit_scan[1][threadIdx.x];

		// Perform overflow-free SIMD Kogge-Stone across digits
		int digit_prefix = WarpScan<RADIX_DIGITS, false>(
				digit_scan, 
				inclusive_total,
				0);

		// first-half of carry update 
		carry[threadIdx.x] = my_carry - digit_prefix;
		
		#pragma unroll
		for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {

			#pragma unroll
			for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
				digit_counts[PASS][SET][threadIdx.x] = counts[PASS][SET] + digit_prefix;
			}
		}
	}
	
	__syncthreads();

	//-------------------------------------------------------------------------
	// Update Ranks
	//-------------------------------------------------------------------------

	UpdateRanks<RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS>(digits, ranks, digit_counts);
	
	
	//-------------------------------------------------------------------------
	// Scatter 
	//-------------------------------------------------------------------------

#if ((__CUDA_ARCH__ < 130) || FERMI_ECC)		

	SwapAndScatterSm10<K, V, RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, 
		ranks,
		exchange,
		d_in_values, 
		d_out_keys, 
		d_out_values, 
		carry, 
		digit_scan,
		extra);
	
#else 

	SwapAndScatterSm13<K, V, RADIX_DIGITS, BIT, PASSES_PER_CYCLE, SETS_PER_PASS, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, 
		ranks,
		exchange,
		d_in_values, 
		d_out_keys, 
		d_out_values, 
		carry, 
		extra);
	
#endif

	__syncthreads();

}



/******************************************************************************
 * Scan/Scatter Kernel Entry Point
 ******************************************************************************/

template <
	typename K, 
	typename V, 
	int PASS, 
	int RADIX_BITS, 
	int BIT, 
	typename PreprocessFunctor, 
	typename PostprocessFunctor>
__launch_bounds__ (B40C_RADIXSORT_THREADS, B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(__CUDA_ARCH__))
__global__ 
void ScanScatterDigits(
	bool *d_from_alt_storage,
	int* d_spine,
	K* d_in_keys,
	K* d_out_keys,
	V* d_in_values,
	V* d_out_values,
	CtaDecomposition work_decomposition)
{

	const int RADIX_DIGITS 				= 1 << RADIX_BITS;
	
	const int LOG_SCAN_LANES_PER_SET	= (RADIX_BITS > 2) ? RADIX_BITS - 2 : 0;					// Always at one lane per set
	const int SCAN_LANES_PER_SET		= 1 << LOG_SCAN_LANES_PER_SET;								// N.B.: we have "declared but never referenced" warnings for these, but they're actually used for template instantiation
	
	const int LOG_SETS_PER_PASS			= B40C_RADIXSORT_LOG_SETS_PER_PASS(__CUDA_ARCH__);			
	const int SETS_PER_PASS				= 1 << LOG_SETS_PER_PASS;
	
	const int LOG_PASSES_PER_CYCLE		= B40C_RADIXSORT_LOG_PASSES_PER_CYCLE(__CUDA_ARCH__, K, V);			
	const int PASSES_PER_CYCLE			= 1 << LOG_PASSES_PER_CYCLE;

	const int LOG_SCAN_LANES_PER_PASS	= LOG_SETS_PER_PASS + LOG_SCAN_LANES_PER_SET;
	const int SCAN_LANES_PER_PASS		= 1 << LOG_SCAN_LANES_PER_PASS;
	
	const int LOG_PARTIALS_PER_LANE 	= B40C_RADIXSORT_LOG_THREADS;
	
	const int LOG_PARTIALS_PER_PASS		= LOG_SCAN_LANES_PER_PASS + LOG_PARTIALS_PER_LANE;

	const int LOG_RAKING_THREADS_PER_PASS 		= B40C_RADIXSORT_LOG_RAKING_THREADS_PER_PASS(__CUDA_ARCH__);
	const int RAKING_THREADS_PER_PASS			= 1 << LOG_RAKING_THREADS_PER_PASS;

	const int LOG_RAKING_THREADS_PER_LANE 		= LOG_RAKING_THREADS_PER_PASS - LOG_SCAN_LANES_PER_PASS;
	const int RAKING_THREADS_PER_LANE 			= 1 << LOG_RAKING_THREADS_PER_LANE;

	const int LOG_PARTIALS_PER_SEG 		= LOG_PARTIALS_PER_LANE - LOG_RAKING_THREADS_PER_LANE;
	const int PARTIALS_PER_SEG 			= 1 << LOG_PARTIALS_PER_SEG;

	const int LOG_PARTIALS_PER_ROW		= (LOG_PARTIALS_PER_SEG < B40C_LOG_MEM_BANKS(__CUDA_ARCH__)) ? B40C_LOG_MEM_BANKS(__CUDA_ARCH__) : LOG_PARTIALS_PER_SEG;		// floor of MEM_BANKS partials per row
	const int PARTIALS_PER_ROW			= 1 << LOG_PARTIALS_PER_ROW;
	const int PADDED_PARTIALS_PER_ROW 	= PARTIALS_PER_ROW + 1;

	const int LOG_SEGS_PER_ROW 			= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG;	
	const int SEGS_PER_ROW				= 1 << LOG_SEGS_PER_ROW;

	const int LOG_ROWS_PER_SET 			= LOG_PARTIALS_PER_PASS - LOG_PARTIALS_PER_ROW;

	const int LOG_ROWS_PER_LANE 		= LOG_PARTIALS_PER_LANE - LOG_PARTIALS_PER_ROW;
	const int ROWS_PER_LANE 			= 1 << LOG_ROWS_PER_LANE;

	const int LOG_ROWS_PER_PASS 		= LOG_SCAN_LANES_PER_PASS + LOG_ROWS_PER_LANE;
	const int ROWS_PER_PASS 			= 1 << LOG_ROWS_PER_PASS;
	
	const int SCAN_LANE_BYTES			= ROWS_PER_PASS * PADDED_PARTIALS_PER_ROW * sizeof(int);
	const int MAX_EXCHANGE_BYTES		= (sizeof(K) > sizeof(V)) ? 
													B40C_RADIXSORT_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V) * sizeof(K) : 
													B40C_RADIXSORT_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V) * sizeof(V);
	const int SCAN_LANE_INT4S         = (B40C_MAX(MAX_EXCHANGE_BYTES, SCAN_LANE_BYTES) + sizeof(int4) - 1) / sizeof(int4);


	// N.B.: We use the following voodoo incantations to elide the compiler's miserable 
	// "declared but never referenced" warnings for these (which are actually used for 
	// template instantiation)	
	SuppressUnusedConstantWarning(SCAN_LANES_PER_SET);
	SuppressUnusedConstantWarning(PARTIALS_PER_SEG);
	SuppressUnusedConstantWarning(LOG_ROWS_PER_SET);
	SuppressUnusedConstantWarning(ROWS_PER_LANE);

    // scan_lanes is a int4[] to avoid alignment issues when casting to (K *) and/or (V *)
	__shared__ int4		scan_lanes[SCAN_LANE_INT4S];
	__shared__ int 		warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE];		// One warpscan per fours-group
	__shared__ int 		carry[RADIX_DIGITS];
	__shared__ int 		digit_scan[2][RADIX_DIGITS];						 
	__shared__ int 		digit_counts[PASSES_PER_CYCLE][SETS_PER_PASS][RADIX_DIGITS];
	__shared__ bool 	non_trivial_digit_pass;
	__shared__ bool		from_alt_storage;
	
	_B40C_REG_MISER_QUALIFIER_ int extra[1];
	_B40C_REG_MISER_QUALIFIER_ int oob[1];

	extra[0] = (blockIdx.x == gridDim.x - 1) ? work_decomposition.extra_elements_last_block : 0;

	// calculate our threadblock's range
	int block_elements, block_offset;
	if (blockIdx.x < work_decomposition.num_big_blocks) {
		block_offset = work_decomposition.big_block_elements * blockIdx.x;
		block_elements = work_decomposition.big_block_elements;
	} else {
		block_offset = (work_decomposition.normal_block_elements * blockIdx.x) + (work_decomposition.num_big_blocks * B40C_RADIXSORT_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V));
		block_elements = work_decomposition.normal_block_elements;
	}
	oob[0] = block_offset + block_elements;	// out-of-bounds

	
	// location for placing 2-element partial reductions in the first lane of a pass	
	int row = threadIdx.x >> LOG_PARTIALS_PER_ROW; 
	int col = threadIdx.x & (PARTIALS_PER_ROW - 1); 
	int *base_partial = reinterpret_cast<int *>(scan_lanes) + (row * PADDED_PARTIALS_PER_ROW) + col; 								
	
	// location for raking across all sets within a pass
	int *raking_partial = 0;										

	if (threadIdx.x < RAKING_THREADS_PER_PASS) {

		// initalize lane warpscans
		if (threadIdx.x < RAKING_THREADS_PER_LANE) {
			
			#pragma unroll
			for (int SCAN_LANE = 0; SCAN_LANE < (int) SCAN_LANES_PER_PASS; SCAN_LANE++) {
				warpscan[SCAN_LANE][0][threadIdx.x] = 0;
			}
		}

		// initialize digit warpscans
		if (threadIdx.x < RADIX_DIGITS) {

			// Initialize digit_scan
			digit_scan[0][threadIdx.x] = 0;
			digit_scan[1][threadIdx.x] = 0;

			// Determine where to read our input
			from_alt_storage = (PASS == 0) ? false : d_from_alt_storage[PASS & 0x1];

			// Read carry in parallel 
			int spine_digit_offset = FastMul(gridDim.x, threadIdx.x);
			int my_digit_carry = d_spine[spine_digit_offset + blockIdx.x];
			carry[threadIdx.x] = my_digit_carry;

			// Determine whether or not we have work to do and setup the next round 
			// accordingly.  Everybody but the first threadblock can determine this 
			// from the number of non-zero-and-non-oob digit carries.  First block 
			// needs someone else's because he always writes the zero offset.
			
			int predicate;
			if (PreprocessFunctor::MustApply() || PostprocessFunctor::MustApply()) {

				non_trivial_digit_pass = true;

			} else {

				if (blockIdx.x > 0) {
					// Non-first CTA : use digit-carry from first block
					my_digit_carry = d_spine[spine_digit_offset];
				}
				
				predicate = ((my_digit_carry > 0) && (my_digit_carry < work_decomposition.num_elements));
				non_trivial_digit_pass = (TallyWarpVote(RADIX_DIGITS, predicate, reinterpret_cast<int *>(scan_lanes)) > 0);
			}

			// Let the next round know which set of buffers to use
			if (blockIdx.x == 0) d_from_alt_storage[(PASS + 1) & 0x1] = from_alt_storage ^ non_trivial_digit_pass;
		}

		// initialize raking segment
		row = threadIdx.x >> LOG_SEGS_PER_ROW;
		col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		raking_partial = reinterpret_cast<int *>(scan_lanes) + (row * PADDED_PARTIALS_PER_ROW) + col; 
	}

	// Sync to acquire non_trivial_digit_pass and from_temp_storage
	__syncthreads();
	
	// Short-circuit this entire pass
	if (!non_trivial_digit_pass) return; 

	if (!from_alt_storage) {
	
		// Scan in tiles of cycle_elements
		while (block_offset < oob[0]) {
	
			SrtsScanDigitCycle<K, V, BIT, true, RADIX_DIGITS, LOG_SCAN_LANES_PER_SET, SCAN_LANES_PER_SET, SETS_PER_PASS, PASSES_PER_CYCLE, LOG_SCAN_LANES_PER_PASS, SCAN_LANES_PER_PASS, LOG_PARTIALS_PER_LANE, LOG_PARTIALS_PER_PASS, LOG_RAKING_THREADS_PER_PASS, RAKING_THREADS_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, LOG_PARTIALS_PER_SEG, PARTIALS_PER_SEG, LOG_PARTIALS_PER_ROW, PARTIALS_PER_ROW, LOG_SEGS_PER_ROW, SEGS_PER_ROW, LOG_ROWS_PER_SET, LOG_ROWS_PER_LANE, ROWS_PER_LANE, LOG_ROWS_PER_PASS, ROWS_PER_PASS, MAX_EXCHANGE_BYTES, PreprocessFunctor, PostprocessFunctor>(	
				reinterpret_cast<typename VecType<K, 2>::Type *>((void *) &d_in_keys[block_offset]), 
				reinterpret_cast<typename VecType<V, 2>::Type *>((void *) &d_in_values[block_offset]), 
				d_out_keys, 
				d_out_values, 
				scan_lanes,
				warpscan,
				carry,
				digit_scan,						 
				digit_counts,
				extra,
				base_partial,
				raking_partial);		
	
			block_offset += B40C_RADIXSORT_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V);
		}
	
		if (extra[0]) {
			
			SrtsScanDigitCycle<K, V, BIT, false, RADIX_DIGITS, LOG_SCAN_LANES_PER_SET, SCAN_LANES_PER_SET, SETS_PER_PASS, PASSES_PER_CYCLE, LOG_SCAN_LANES_PER_PASS, SCAN_LANES_PER_PASS, LOG_PARTIALS_PER_LANE, LOG_PARTIALS_PER_PASS, LOG_RAKING_THREADS_PER_PASS, RAKING_THREADS_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, LOG_PARTIALS_PER_SEG, PARTIALS_PER_SEG, LOG_PARTIALS_PER_ROW, PARTIALS_PER_ROW, LOG_SEGS_PER_ROW, SEGS_PER_ROW, LOG_ROWS_PER_SET, LOG_ROWS_PER_LANE, ROWS_PER_LANE, LOG_ROWS_PER_PASS, ROWS_PER_PASS, MAX_EXCHANGE_BYTES, PreprocessFunctor, PostprocessFunctor>(	
				reinterpret_cast<typename VecType<K, 2>::Type *>((void *) &d_in_keys[block_offset]), 
				reinterpret_cast<typename VecType<V, 2>::Type *>((void *) &d_in_values[block_offset]), 
				d_out_keys, 
				d_out_values, 
				scan_lanes,
				warpscan,
				carry,
				digit_scan,						 
				digit_counts,
				extra,
				base_partial,
				raking_partial);		
		}

	} else {
		
		// Scan in tiles of cycle_elements
		while (block_offset < oob[0]) {

			SrtsScanDigitCycle<K, V, BIT, true, RADIX_DIGITS, LOG_SCAN_LANES_PER_SET, SCAN_LANES_PER_SET, SETS_PER_PASS, PASSES_PER_CYCLE, LOG_SCAN_LANES_PER_PASS, SCAN_LANES_PER_PASS, LOG_PARTIALS_PER_LANE, LOG_PARTIALS_PER_PASS, LOG_RAKING_THREADS_PER_PASS, RAKING_THREADS_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, LOG_PARTIALS_PER_SEG, PARTIALS_PER_SEG, LOG_PARTIALS_PER_ROW, PARTIALS_PER_ROW, LOG_SEGS_PER_ROW, SEGS_PER_ROW, LOG_ROWS_PER_SET, LOG_ROWS_PER_LANE, ROWS_PER_LANE, LOG_ROWS_PER_PASS, ROWS_PER_PASS, MAX_EXCHANGE_BYTES, PreprocessFunctor, PostprocessFunctor>(	
				reinterpret_cast<typename VecType<K, 2>::Type *>((void *) &d_out_keys[block_offset]), 
				reinterpret_cast<typename VecType<V, 2>::Type *>((void *) &d_out_values[block_offset]), 
				d_in_keys, 
				d_in_values, 
				scan_lanes,
				warpscan,
				carry,
				digit_scan,						 
				digit_counts,
				extra,
				base_partial,
				raking_partial);		

			block_offset += B40C_RADIXSORT_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V);
		}

		if (extra[0]) {
			
			SrtsScanDigitCycle<K, V, BIT, false, RADIX_DIGITS, LOG_SCAN_LANES_PER_SET, SCAN_LANES_PER_SET, SETS_PER_PASS, PASSES_PER_CYCLE, LOG_SCAN_LANES_PER_PASS, SCAN_LANES_PER_PASS, LOG_PARTIALS_PER_LANE, LOG_PARTIALS_PER_PASS, LOG_RAKING_THREADS_PER_PASS, RAKING_THREADS_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, LOG_PARTIALS_PER_SEG, PARTIALS_PER_SEG, LOG_PARTIALS_PER_ROW, PARTIALS_PER_ROW, LOG_SEGS_PER_ROW, SEGS_PER_ROW, LOG_ROWS_PER_SET, LOG_ROWS_PER_LANE, ROWS_PER_LANE, LOG_ROWS_PER_PASS, ROWS_PER_PASS, MAX_EXCHANGE_BYTES, PreprocessFunctor, PostprocessFunctor>(	
				reinterpret_cast<typename VecType<K, 2>::Type *>((void *) &d_out_keys[block_offset]), 
				reinterpret_cast<typename VecType<V, 2>::Type *>((void *) &d_out_values[block_offset]), 
				d_in_keys, 
				d_in_values, 
				scan_lanes,
				warpscan,
				carry,
				digit_scan,						 
				digit_counts,
				extra,
				base_partial,
				raking_partial);		
		}
		
	}
}

} // end namespace b40c_thrust
} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

