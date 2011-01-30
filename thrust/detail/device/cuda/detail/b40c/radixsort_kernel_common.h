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
 * Configuration management for B40C radix sorting kernels  
 ******************************************************************************/

#pragma once

#include "kernel_utils.h"
#include "vector_types.h"
#include "radixsort_key_conversion.h"

namespace thrust {
namespace detail {
namespace device {
namespace cuda   {
namespace detail {
namespace b40c_thrust   {


/******************************************************************************
 * Radix sorting configuration  
 ******************************************************************************/

// 128 threads
#define B40C_RADIXSORT_LOG_THREADS						7								
#define B40C_RADIXSORT_THREADS							(1 << B40C_RADIXSORT_LOG_THREADS)	

// Target threadblock occupancy for counting/reduction kernel
#define B40C_SM20_REDUCE_CTA_OCCUPANCY()					(8)			// 8 threadblocks on GF100
#define B40C_SM12_REDUCE_CTA_OCCUPANCY()					(5)			// 5 threadblocks on GT200
#define B40C_SM10_REDUCE_CTA_OCCUPANCY()					(3)			// 4 threadblocks on G80
#define B40C_RADIXSORT_REDUCE_CTA_OCCUPANCY(version)		((version >= 200) ? B40C_SM20_REDUCE_CTA_OCCUPANCY() : 	\
			        										 (version >= 120) ? B40C_SM12_REDUCE_CTA_OCCUPANCY() : 	\
					        													B40C_SM10_REDUCE_CTA_OCCUPANCY())		
													                    
// Target threadblock occupancy for bulk scan/scatter kernel
#define B40C_SM20_SCAN_SCATTER_CTA_OCCUPANCY()				(7)			// 7 threadblocks on GF100
#define B40C_SM12_SCAN_SCATTER_CTA_OCCUPANCY()				(5)			// 5 threadblocks on GT200
#define B40C_SM10_SCAN_SCATTER_CTA_OCCUPANCY()				(2)			// 2 threadblocks on G80
#define B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(version)	((version >= 200) ? B40C_SM20_SCAN_SCATTER_CTA_OCCUPANCY() : 	\
			    											 (version >= 120) ? B40C_SM12_SCAN_SCATTER_CTA_OCCUPANCY() : 	\
				    															B40C_SM10_SCAN_SCATTER_CTA_OCCUPANCY())		

// Number of 256-element sets to rake per raking pass
#define B40C_SM20_LOG_SETS_PER_PASS()					(1)			// 2 sets on GF100
#define B40C_SM12_LOG_SETS_PER_PASS()					(0)			// 1 set on GT200
#define B40C_SM10_LOG_SETS_PER_PASS()					(1)			// 2 sets on G80
#define B40C_RADIXSORT_LOG_SETS_PER_PASS(version)		((version >= 200) ? B40C_SM20_LOG_SETS_PER_PASS() : 	\
			     										 (version >= 120) ? B40C_SM12_LOG_SETS_PER_PASS() : 	\
				    														B40C_SM10_LOG_SETS_PER_PASS())		

// Number of raking passes per cycle
#define B40C_SM20_LOG_PASSES_PER_CYCLE(K, V)					(((B40C_MAX(sizeof(K), sizeof(V)) > 4) || _B40C_LP64_) ? 0 : 1)	// 2 passes on GF100 (only one for large keys/values, or for 64-bit device pointers)
#define B40C_SM12_LOG_PASSES_PER_CYCLE(K, V)					(B40C_MAX(sizeof(K), sizeof(V)) > 4 ? 0 : 1)					// 2 passes on GT200 (only for large keys/values)
#define B40C_SM10_LOG_PASSES_PER_CYCLE(K, V)					(0)																// 1 pass on G80
#define B40C_RADIXSORT_LOG_PASSES_PER_CYCLE(version, K, V)	((version >= 200) ? B40C_SM20_LOG_PASSES_PER_CYCLE(K, V) : 	\
				    										 (version >= 120) ? B40C_SM12_LOG_PASSES_PER_CYCLE(K, V) : 	\
					    														B40C_SM10_LOG_PASSES_PER_CYCLE(K, V))		


// Number of raking threads per raking pass
#define B40C_SM20_LOG_RAKING_THREADS_PER_PASS()				(B40C_LOG_WARP_THREADS + 1)		// 2 raking warps on GF100
#define B40C_SM12_LOG_RAKING_THREADS_PER_PASS()				(B40C_LOG_WARP_THREADS)			// 1 raking warp on GT200
#define B40C_SM10_LOG_RAKING_THREADS_PER_PASS()				(B40C_LOG_WARP_THREADS + 2)		// 4 raking warps on G80
#define B40C_RADIXSORT_LOG_RAKING_THREADS_PER_PASS(version)	((version >= 200) ? B40C_SM20_LOG_RAKING_THREADS_PER_PASS() : 	\
				    										 (version >= 120) ? B40C_SM12_LOG_RAKING_THREADS_PER_PASS() : 	\
					    														B40C_SM10_LOG_RAKING_THREADS_PER_PASS())		


// Number of elements per cycle
#define B40C_RADIXSORT_LOG_CYCLE_ELEMENTS(version, K, V)		(B40C_RADIXSORT_LOG_SETS_PER_PASS(version) + B40C_RADIXSORT_LOG_PASSES_PER_CYCLE(version, K, V) + B40C_RADIXSORT_LOG_THREADS + 1)
#define B40C_RADIXSORT_CYCLE_ELEMENTS(version, K, V)			(1 << B40C_RADIXSORT_LOG_CYCLE_ELEMENTS(version, K, V))

// Number of warps per CTA
#define B40C_RADIXSORT_LOG_WARPS								(B40C_RADIXSORT_LOG_THREADS - B40C_LOG_WARP_THREADS)
#define B40C_RADIXSORT_WARPS									(1 << B40C_RADIXSORT_LOG_WARPS)

// Number of threads for spine-scanning kernel
#define B40C_RADIXSORT_LOG_SPINE_THREADS						7		// 128 threads
#define B40C_RADIXSORT_SPINE_THREADS							(1 << B40C_RADIXSORT_LOG_SPINE_THREADS)	

// Number of elements per spine-scanning cycle
#define B40C_RADIXSORT_LOG_SPINE_CYCLE_ELEMENTS  				9		// 512 elements
#define B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS		    			(1 << B40C_RADIXSORT_LOG_SPINE_CYCLE_ELEMENTS)



/******************************************************************************
 * SRTS Control Structures
 ******************************************************************************/


/**
 * Value-type structure denoting keys-only sorting
 */
struct KeysOnlyType {};

/**
 * Returns whether or not the templated type indicates keys-only sorting
 */
template <typename V>
inline __host__ __device__ bool IsKeysOnly() {return false;}


/**
 * Returns whether or not the templated type indicates keys-only sorting
 */
template <>
inline __host__ __device__ bool IsKeysOnly<KeysOnlyType>() {return true;}


/**
 * A given threadblock may receive one of three different amounts of 
 * work: "big", "normal", and "last".  The big workloads are one
 * cycle_elements greater than the normal, and the last workload 
 * does the extra (problem-size % cycle_elements) work.
 */
struct CtaDecomposition {
	unsigned int num_big_blocks;
	unsigned int big_block_elements;
	unsigned int normal_block_elements;
	unsigned int extra_elements_last_block;
	unsigned int num_elements;
};


} // end namespace b40c_thrust
} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

