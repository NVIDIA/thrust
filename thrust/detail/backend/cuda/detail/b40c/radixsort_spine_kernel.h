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
 * Top-level histogram/spine scanning kernel
 ******************************************************************************/

#pragma once

#include "radixsort_kernel_common.h"

namespace thrust  {
namespace detail  {
namespace backend {
namespace cuda    {
namespace detail  {
namespace b40c_thrust   {

/******************************************************************************
 * Scans a cycle of RADIXSORT_CYCLE_ELEMENTS elements 
 ******************************************************************************/

template<int PARTIALS_PER_SEG>
__device__ __forceinline__ void SrtsScanCycle(
	int *smem_offset,
	int *smem_segment,
	int warpscan[2][B40C_WARP_THREADS],
	int4 *in, 
	int4 *out,
	int &carry)
{
	int4 datum; 

	// read input data
	datum = in[threadIdx.x];

	smem_offset[0] = datum.x + datum.y + datum.z + datum.w;

	__syncthreads();

	if (threadIdx.x < B40C_WARP_THREADS) {

		int partial_reduction = SerialReduce<PARTIALS_PER_SEG>(smem_segment);

		int seed = WarpScan<B40C_WARP_THREADS, false>(warpscan, partial_reduction, 0);
		seed += carry;		
		
		SerialScan<PARTIALS_PER_SEG>(smem_segment, seed);

		carry += warpscan[1][B40C_WARP_THREADS - 1];	
	}

	__syncthreads();

	int part0 = smem_offset[0];
	int part1;

	part1 = datum.x + part0;
	datum.x = part0;
	part0 = part1 + datum.y;
	datum.y = part1;

	part1 = datum.z + part0;
	datum.z = part0;
	part0 = part1 + datum.w;
	datum.w = part1;
	
	out[threadIdx.x] = datum;
}


/******************************************************************************
 * Spine/histogram Scan Kernel Entry Point
 ******************************************************************************/

template <typename T>
__global__ void SrtsScanSpine(
	int *d_ispine,
	int *d_ospine,
	int normal_block_elements)
{
	const int LOG_PARTIALS				= B40C_RADIXSORT_LOG_THREADS;				
	const int PARTIALS			 		= 1 << LOG_PARTIALS;
	
	const int LOG_PARTIALS_PER_SEG 		= LOG_PARTIALS - B40C_LOG_WARP_THREADS;
	const int PARTIALS_PER_SEG 			= 1 << LOG_PARTIALS_PER_SEG;

	const int LOG_PARTIALS_PER_ROW		= (LOG_PARTIALS_PER_SEG < B40C_LOG_MEM_BANKS(__CUDA_ARCH__)) ? B40C_LOG_MEM_BANKS(__CUDA_ARCH__) : LOG_PARTIALS_PER_SEG;		// floor of 32 elts per row
	const int PARTIALS_PER_ROW			= 1 << LOG_PARTIALS_PER_ROW;
	
	const int LOG_SEGS_PER_ROW 			= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG;	
	const int SEGS_PER_ROW				= 1 << LOG_SEGS_PER_ROW;

	const int SMEM_ROWS 				= PARTIALS / PARTIALS_PER_ROW;
	
	__shared__ int smem[SMEM_ROWS][PARTIALS_PER_ROW + 1];
	__shared__ int warpscan[2][B40C_WARP_THREADS];

  // WAR spurious unused constant warning
  SuppressUnusedConstantWarning(PARTIALS_PER_SEG);

	int *smem_segment = 0;
	int carry = 0;

	int row = threadIdx.x >> LOG_PARTIALS_PER_ROW;		
	int col = threadIdx.x & (PARTIALS_PER_ROW - 1);			
	int *smem_offset = &smem[row][col];

	if (blockIdx.x > 0) {
		return;
	}
	
	if (threadIdx.x < B40C_WARP_THREADS) {
		
		// two segs per row, odd segs are offset by 8
		row = threadIdx.x >> LOG_SEGS_PER_ROW;
		col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		smem_segment = &smem[row][col];
	
		if (threadIdx.x < B40C_WARP_THREADS) {
			carry = 0;
			warpscan[0][threadIdx.x] = 0;
		}
	}

	// scan the spine in blocks of cycle_elements
	int block_offset = 0;
	while (block_offset < normal_block_elements) {
		
		SrtsScanCycle<PARTIALS_PER_SEG>(	
			smem_offset, 
			smem_segment, 
			warpscan,
			reinterpret_cast<int4 *>((void *) &d_ispine[block_offset]), 
			reinterpret_cast<int4 *>((void *) &d_ospine[block_offset]), 
			carry);

		block_offset += B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
	}
} 


} // end namespace b40c_thrust
} // end namespace detail
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

