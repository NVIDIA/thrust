/******************************************************************************
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
 ******************************************************************************/



/******************************************************************************
 * Radix Sorting API
 *
 * USAGE:
 * 
 * Using the B40C radix sorting implementation is easy.  Just #include this API 
 * file and its kernel include dependencies within your source.  Below are two
 * examples for using: 
 *
 * (1) A keys-only example for sorting floats:
 * 
 *		// Create storage-management structure
 * 		RadixSortStorage<float> device_storage(d_float_keys);			
 *
 *		// Create and enact sorter
 * 		RadixSortingEnactor sorter<float>(d_float_keys_len);
 *		sorter.EnactSort(device_storage);
 *
 *		// Re-acquire pointer to sorted keys, free unused/temp storage 
 *		d_float_keys = device_storage.d_keys;
 *		device_storage.CleanupTempStorage();
 *
 * (2) And a key-value example for sorting ints paired with doubles:
 *
 *		// Create storage-management structure
 * 		RadixSortStorage<int, double> device_storage(d_int_keys, d_double_values);			
 *
 *		// Create and enact sorter
 * 		RadixSortingEnactor sorter<int, double>(d_int_keys_len);
 *		sorter.EnactSort(device_storage);
 *
 *		// Re-acquire pointer to sorted keys and values, free unused/temp storage 
 *		d_int_keys = device_storage.d_keys;
 *		d_double_values = device_storage.d_values;
 *		device_storage.CleanupTempStorage();
 *
 *
 ******************************************************************************/

#pragma once

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>

#include "radixsort_reduction_kernel.h"
#include "radixsort_spine_kernel.h"
#include "radixsort_scanscatter_kernel.h"

#include <thrust/swap.h>

namespace thrust {
namespace detail {
namespace device {
namespace cuda   {
namespace detail {
namespace b40c_thrust   {


/******************************************************************************
 * Debugging options
 ******************************************************************************/

static bool RADIXSORT_DEBUG = false;



/******************************************************************************
 * Structures for mananging device-side sorting state
 ******************************************************************************/

/**
 * Sorting storage-management structure for device vectors
 */
template <typename K, typename V = KeysOnlyType>
struct RadixSortStorage {

	// Device vector of keys to sort
	K* d_keys;
	
	// Device vector of values to sort
	V* d_values;

	// Ancillary device vector for key storage 
	K* d_alt_keys;

	// Ancillary device vector for value storage
	V* d_alt_values;

	// Temporary device storage needed for radix sorting histograms
	int *d_spine;
	
	// Flip-flopping temporary device storage denoting which digit place 
	// pass should read from which input source (i.e., false if reading from 
	// keys, true if reading from alternate_keys
	bool *d_from_alt_storage;

	// Host-side boolean whether or not an odd number of sorting passes left the 
	// results in alternate storage.  If so, the d_keys (and d_values) pointers 
	// will have been swapped with the d_alt_keys (and d_alt_values) pointers in order to 
	// point to the final results.
	bool using_alternate_storage;
	
	// Constructor
	RadixSortStorage(K* keys = NULL, V* values = NULL) 
	{ 
		d_keys = keys; 
		d_values = values; 
		d_alt_keys = NULL; 
		d_alt_values = NULL; 
		d_spine = NULL;
		d_from_alt_storage = NULL;
		
		using_alternate_storage = false;
	}

	// Clean up non-results storage (may include freeing original storage if 
	// primary pointers were swizzled as per using_alternate_storage) 
	cudaError_t CleanupTempStorage() 
	{
		if (d_alt_keys) cudaFree(d_alt_keys);
		if (d_alt_values) cudaFree(d_alt_values);
		if (d_spine) cudaFree(d_spine);
		if (d_from_alt_storage) cudaFree(d_from_alt_storage);
		
		return cudaSuccess;
	}
};



/******************************************************************************
 * Base class for sorting enactors
 ******************************************************************************/


/**
 * Base class for SRTS radix sorting enactors.
 */
template <typename K, typename V>
class BaseRadixSortingEnactor 
{
public:
	
	// Unsigned integer type suitable for radix sorting of keys
	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;

protected:

	//
	// Information about our problem configuration
	//
	
	bool				_keys_only;
	unsigned int 		_num_elements;
	int 				_cycle_elements;
	int 				_spine_elements;
	int 				_grid_size;
	CtaDecomposition 	_work_decomposition;
	int 				_passes;
	bool 				_swizzle_pointers_for_odd_passes;

	// Information about our target device
	cudaDeviceProp 		_device_props;
	int 				_device_sm_version;
	
	// Information about our kernel assembly
	int 				_kernel_ptx_version;
	cudaFuncAttributes 	_spine_scan_kernel_attrs;
	
protected:
	
	/**
	 * Constructor.
	 */
	BaseRadixSortingEnactor(int passes, int radix_bits, unsigned int num_elements, int max_grid_size, bool swizzle_pointers_for_odd_passes = true); 
	
	/**
	 * Heuristic for determining the number of CTAs to launch.
	 *   
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  A value of 0 indicates 
	 * 		that the default value should be used.
	 * 
	 * @return The actual number of CTAs that should be launched
	 */
	int GridSize(int max_grid_size);

	/**
	 * Performs a distribution sorting pass over a single digit place
	 */
	template <int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
	cudaError_t DigitPlacePass(const RadixSortStorage<ConvertedKeyType, V> &converted_storage); 
	
	/**
	 * Enacts a sorting operation by performing the the appropriate 
	 * digit-place passes.  To be overloaded by specialized subclasses.
	 */
	virtual cudaError_t EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage) = 0;
	
public:
	
	/**
	 * Returns the length (in unsigned ints) of the device vector needed for  
	 * temporary storage of the reduction spine.  Useful if pre-allocating 
	 * your own device storage (as opposed to letting EnactSort() allocate it
	 * for you).
	 */
	int SpineElements() { return _spine_elements; }

	/**
	 * Returns whether or not the problem will fit on the device.
	 */
	bool CanFit();

	/**
	 * Enacts a radix sorting operation on the specified device data.
	 * 
	 * IMPORTANT NOTES: The device storage backing the specified input vectors of 
	 * keys (and data) will be modified.  (I.e., treat this as an in-place sort.)  
	 * 
	 * Additionally, the pointers in the problem_storage structure may be updated 
	 * (a) depending upon the number of digit-place sorting passes needed, and (b) 
	 * whether or not the caller has already allocated temporary storage.  
	 * 
	 * The sorted results will always be referenced by problem_storage.d_keys (and 
	 * problem_storage.d_values).  However, for an odd number of sorting passes (uncommon)
	 * these results will actually be backed by the storage initially allocated for 
	 * by problem_storage.d_alt_keys (and problem_storage.d_alt_values).  If so, 
	 * problem_storage.d_alt_keys and problem_storage.d_alt_keys will be updated to 
	 * reference the original problem_storage.d_keys and problem_storage.d_values in order 
	 * to facilitate cleanup.  
	 * 
	 * This means it is important to avoid keeping stale copies of device pointers 
	 * to keys/data; you will want to re-reference the pointers in problem_storage.
	 * 
	 * @param[in/out] 	problem_storage 
	 * 		Device vectors of keys and values to sort, and ancillary storage 
	 * 		needed by the sorting kernels. See the IMPORTANT NOTES above. 
	 * 
	 * 		The problem_storage.[alternate_keys|alternate_values|d_spine] fields are 
	 * 		temporary storage needed by the sorting kernels.  To facilitate 
	 * 		speed, callers are welcome to re-use this storage for same-sized 
	 * 		(or smaller) sortign problems. If NULL, these storage vectors will be 
	 *      allocated by this routine (and must be subsequently cuda-freed by 
	 *      the caller).
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	cudaError_t EnactSort(RadixSortStorage<K, V> &problem_storage);	

    /*
     * Destructor
     */
    virtual ~BaseRadixSortingEnactor() {}
};



template <typename K, typename V>
BaseRadixSortingEnactor<K, V>::BaseRadixSortingEnactor(
	int passes, 
	int max_radix_bits, 
	unsigned int num_elements, 
	int max_grid_size,
	bool swizzle_pointers_for_odd_passes) 
{
	//
	// Get current device properties 
	//

	int current_device;
	cudaGetDevice(&current_device);
	cudaGetDeviceProperties(&_device_props, current_device);
	_device_sm_version = _device_props.major * 100 + _device_props.minor * 10;

	
	//
	// Get SM version of compiled kernel assembly
	//
	cudaFuncGetAttributes(&_spine_scan_kernel_attrs, SrtsScanSpine<void>);
	_kernel_ptx_version = _spine_scan_kernel_attrs.ptxVersion * 10;
	

	//
	// Determine number of CTAs to launch, shared memory, cycle elements, etc.
	//

	_passes								= passes;
	_num_elements 						= num_elements;
	_keys_only 							= IsKeysOnly<V>();
	_cycle_elements 					= B40C_RADIXSORT_CYCLE_ELEMENTS(_kernel_ptx_version , ConvertedKeyType, V);
	_grid_size 							= GridSize(max_grid_size);
	_swizzle_pointers_for_odd_passes	= swizzle_pointers_for_odd_passes;
	
	int total_cycles 			= _num_elements / _cycle_elements;
	int cycles_per_block 		= total_cycles / _grid_size;						
	int extra_cycles 			= total_cycles - (cycles_per_block * _grid_size);

	CtaDecomposition work_decomposition = {
		extra_cycles,										// num_big_blocks
		(cycles_per_block + 1) * _cycle_elements,			// big_block_elements
		cycles_per_block * _cycle_elements,					// normal_block_elements
		_num_elements - (total_cycles * _cycle_elements),	// extra_elements_last_block
		_num_elements};										// num_elements
	
	_work_decomposition = work_decomposition;
	
	int spine_cycles = ((_grid_size * (1 << max_radix_bits)) + B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS - 1) / B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
	_spine_elements = spine_cycles * B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
}



template <typename K, typename V>
int BaseRadixSortingEnactor<K, V>::GridSize(int max_grid_size)
{
	const int SINGLE_CTA_CUTOFF = 0;		// right now zero; we have no single-cta sorting

	// find maximum number of threadblocks if "use-default"
	if (max_grid_size == 0) {

		if (_num_elements <= static_cast<unsigned int>(SINGLE_CTA_CUTOFF)) {

			// The problem size is too small to warrant a two-level reduction: 
			// use only one stream-processor
			max_grid_size = 1;

		} else {

			if (_device_sm_version <= 120) {
				
				// G80/G90
				max_grid_size = _device_props.multiProcessorCount * 4;
				
			} else if (_device_sm_version < 200) {
				
				// GT200 (has some kind of TLB or icache drama)
				int orig_max_grid_size = _device_props.multiProcessorCount * B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(_kernel_ptx_version);
				if (_keys_only) { 
					orig_max_grid_size *= (_num_elements + (1024 * 1024 * 96) - 1) / (1024 * 1024 * 96);
				} else {
					orig_max_grid_size *= (_num_elements + (1024 * 1024 * 64) - 1) / (1024 * 1024 * 64);
				}
				max_grid_size = orig_max_grid_size;

				if (_num_elements / _cycle_elements > static_cast<unsigned int>(max_grid_size)) {
	
					double multiplier1 = 4.0;
					double multiplier2 = 16.0;

					double delta1 = 0.068;
					double delta2 = 0.127;	
	
					int dividend = (_num_elements + _cycle_elements - 1) / _cycle_elements;
	
					while(true) {
	
						double quotient = ((double) dividend) / (multiplier1 * max_grid_size);
						quotient -= (int) quotient;

						if ((quotient > delta1) && (quotient < 1 - delta1)) {

							quotient = ((double) dividend) / (multiplier2 * max_grid_size / 3.0);
							quotient -= (int) quotient;

							if ((quotient > delta2) && (quotient < 1 - delta2)) {
								break;
							}
						}
						
						if (max_grid_size == orig_max_grid_size - 2) {
							max_grid_size = orig_max_grid_size - 30;
						} else {
							max_grid_size -= 1;
						}
					}
				}
			} else {
				
				// GF100
				max_grid_size = 418;
			}
		}
	}

	// Calculate the actual number of threadblocks to launch.  Initially
	// assume that each threadblock will do only one cycle_elements worth 
	// of work, but then clamp it by the "max" restriction derived above
	// in order to accomodate the "single-sp" and "saturated" cases.

	int grid_size = _num_elements / _cycle_elements;
	if (grid_size == 0) {
		grid_size = 1;
	}
	if (grid_size > max_grid_size) {
		grid_size = max_grid_size;
	} 

	return grid_size;
}



template <typename K, typename V>
bool BaseRadixSortingEnactor<K, V>::
CanFit() 
{
	long long bytes = (_num_elements * sizeof(K) * 2) + (_spine_elements * sizeof(int));
	if (!_keys_only) bytes += _num_elements * sizeof(V) * 2;

	if (_device_props.totalGlobalMem < 1024 * 1024 * 513) {
		return (bytes < ((double) _device_props.totalGlobalMem) * 0.81); 	// allow up to 81% capacity for 512MB   
	}
	
	return (bytes < ((double) _device_props.totalGlobalMem) * 0.89); 	// allow up to 90% capacity 
}



template <typename K, typename V>
template <int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
cudaError_t BaseRadixSortingEnactor<K, V>::
DigitPlacePass(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
{
	int threads = B40C_RADIXSORT_THREADS;
	int dynamic_smem;

	cudaFuncAttributes reduce_kernel_attrs, scan_scatter_attrs;
	cudaFuncGetAttributes(&reduce_kernel_attrs, RakingReduction<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor>);
	cudaFuncGetAttributes(&scan_scatter_attrs, ScanScatterDigits<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor>);
	
	//
	// Counting Reduction
	//

	// Run tesla flush kernel if we have two or more threadblocks for each of the SMs
	if ((_device_sm_version == 130) && (_work_decomposition.num_elements > static_cast<unsigned int>(_device_props.multiProcessorCount * _cycle_elements * 2))) { 
		FlushKernel<void><<<_grid_size, B40C_RADIXSORT_THREADS, scan_scatter_attrs.sharedSizeBytes>>>();
		synchronize_if_enabled("FlushKernel");
	}

	// GF100 and GT200 get the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels)
	dynamic_smem = (_kernel_ptx_version >= 130) ? scan_scatter_attrs.sharedSizeBytes - reduce_kernel_attrs.sharedSizeBytes : 0;

	RakingReduction<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor> <<<_grid_size, threads, dynamic_smem>>>(
		converted_storage.d_from_alt_storage,
		converted_storage.d_spine,
		converted_storage.d_keys,
		converted_storage.d_alt_keys,
		_work_decomposition);
    synchronize_if_enabled("RakingReduction");

	
	//
	// Spine
	//
	
	// GF100 and GT200 get the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels)
	dynamic_smem = (_kernel_ptx_version >= 130) ? scan_scatter_attrs.sharedSizeBytes - _spine_scan_kernel_attrs.sharedSizeBytes : 0;
	
	SrtsScanSpine<void><<<_grid_size, B40C_RADIXSORT_SPINE_THREADS, dynamic_smem>>>(
		converted_storage.d_spine,
		converted_storage.d_spine,
		_spine_elements);
    synchronize_if_enabled("SrtsScanSpine");

	
	//
	// Scanning Scatter
	//
	
	// Run tesla flush kernel if we have two or more threadblocks for each of the SMs
	if ((_device_sm_version == 130) && (_work_decomposition.num_elements > static_cast<unsigned int>(_device_props.multiProcessorCount * _cycle_elements * 2))) { 
		FlushKernel<void><<<_grid_size, B40C_RADIXSORT_THREADS, scan_scatter_attrs.sharedSizeBytes>>>();
		synchronize_if_enabled("FlushKernel");
	}

	ScanScatterDigits<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor> <<<_grid_size, threads, 0>>>(
		converted_storage.d_from_alt_storage,
		converted_storage.d_spine,
		converted_storage.d_keys,
		converted_storage.d_alt_keys,
		converted_storage.d_values,
		converted_storage.d_alt_values,
		_work_decomposition);
    synchronize_if_enabled("ScanScatterDigits");

	return cudaSuccess;
}



template <typename K, typename V>
cudaError_t BaseRadixSortingEnactor<K, V>::
EnactSort(RadixSortStorage<K, V> &problem_storage) 
{
	//
	// Allocate device memory for temporary storage (if necessary)
	//

	if (problem_storage.d_alt_keys == NULL) {
		cudaMalloc((void**) &problem_storage.d_alt_keys, _num_elements * sizeof(K));
	}
	if (!_keys_only && (problem_storage.d_alt_values == NULL)) {
		cudaMalloc((void**) &problem_storage.d_alt_values, _num_elements * sizeof(V));
	}
	if (problem_storage.d_spine == NULL) {
		cudaMalloc((void**) &problem_storage.d_spine, _spine_elements * sizeof(int));
	}
	if (problem_storage.d_from_alt_storage == NULL) {
		cudaMalloc((void**) &problem_storage.d_from_alt_storage, 2 * sizeof(bool));
	}

	// Determine suitable type of unsigned byte storage to use for keys 
	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;
	
	// Copy storage pointers to an appropriately typed stucture 
	RadixSortStorage<ConvertedKeyType, V> converted_storage;
	memcpy(&converted_storage, &problem_storage, sizeof(RadixSortStorage<K, V>));

	// 
	// Enact the sorting operation
	//
	
	if (RADIXSORT_DEBUG) {
		
		printf("_device_sm_version: %d, _kernel_ptx_version: %d\n", _device_sm_version, _kernel_ptx_version);
		printf("Bottom-level reduction & scan kernels:\n\tgrid_size: %d, \n\tthreads: %d, \n\tcycle_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d\n\textra_elements_last_block: %d\n\n",
			_grid_size, B40C_RADIXSORT_THREADS, _cycle_elements, _work_decomposition.num_big_blocks, _work_decomposition.big_block_elements, _work_decomposition.normal_block_elements, _work_decomposition.extra_elements_last_block);
		printf("Top-level spine scan:\n\tgrid_size: %d, \n\tthreads: %d, \n\tspine_block_elements: %d\n\n", 
			_grid_size, B40C_RADIXSORT_SPINE_THREADS, _spine_elements);
	}	

	cudaError_t retval = EnactDigitPlacePasses(converted_storage);

	
	//
	// Swizzle pointers if we left our sorted output in temp storage 
	//
	
	if (_swizzle_pointers_for_odd_passes) {
	
		cudaMemcpy(
			&problem_storage.using_alternate_storage, 
			&problem_storage.d_from_alt_storage[_passes & 0x1], 
			sizeof(bool), 
			cudaMemcpyDeviceToHost);
	
		if (problem_storage.using_alternate_storage) {
            thrust::swap<K*>(problem_storage.d_keys, problem_storage.d_alt_keys);
			if (!_keys_only) {
                thrust::swap<V*>(problem_storage.d_values, problem_storage.d_alt_values);
			}
		}
	}
	
	return retval;
}





/******************************************************************************
 * Sorting enactor classes
 ******************************************************************************/

/**
 * Generic sorting enactor class.  Simply create an instance of this class
 * with your key-type K (and optionally value-type V if sorting with satellite 
 * values).
 * 
 * Template specialization provides the appropriate enactor instance to handle 
 * the specified data types. 
 * 
 * @template-param K
 * 		Type of keys to be sorted
 *
 * @template-param V
 * 		Type of values to be sorted.
 *
 * @template-param ConvertedKeyType
 * 		Leave as default to effect necessary enactor specialization.
 */
template <typename K, typename V = KeysOnlyType, typename ConvertedKeyType = typename KeyConversion<K>::UnsignedBits>
class RadixSortingEnactor;



/**
 * Sorting enactor that is specialized for for 8-bit key types
 */
template <typename K, typename V>
class RadixSortingEnactor<K, V, unsigned char> : public BaseRadixSortingEnactor<K, V>
{
protected:

	typedef BaseRadixSortingEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

	cudaError_t EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{
		Base::template DigitPlacePass<0, 4, 0, PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
		Base::template DigitPlacePass<1, 4, 4, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >    (converted_storage); 

		return cudaSuccess;
	}

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		num_elements 
	 * 		Length (in elements) of the input to a sorting operation
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	RadixSortingEnactor(unsigned int num_elements, int max_grid_size = 0) : Base::BaseRadixSortingEnactor(2, 4, num_elements, max_grid_size) {}

};



/**
 * Sorting enactor that is specialized for for 16-bit key types
 */
template <typename K, typename V>
class RadixSortingEnactor<K, V, unsigned short> : public BaseRadixSortingEnactor<K, V>
{
protected:

	typedef BaseRadixSortingEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

	cudaError_t EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{
		Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
		Base::template DigitPlacePass<1, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<2, 4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<3, 4, 12, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >    (converted_storage); 

		return cudaSuccess;
	}

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		num_elements 
	 * 		Length (in elements) of the input to a sorting operation
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	RadixSortingEnactor(unsigned int num_elements, int max_grid_size = 0) : Base::BaseRadixSortingEnactor(4, 4, num_elements, max_grid_size) {}

};


/**
 * Sorting enactor that is specialized for for 32-bit key types
 */
template <typename K, typename V>
class RadixSortingEnactor<K, V, unsigned int> : public BaseRadixSortingEnactor<K, V>
{
protected:

	typedef BaseRadixSortingEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

	cudaError_t EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{
		Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
		Base::template DigitPlacePass<1, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<2, 4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<3, 4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<4, 4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<5, 4, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<6, 4, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<7, 4, 28, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >    (converted_storage); 

		return cudaSuccess;
	}

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		num_elements 
	 * 		Length (in elements) of the input to a sorting operation
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	RadixSortingEnactor(unsigned int num_elements, int max_grid_size = 0) : Base::BaseRadixSortingEnactor(8, 4, num_elements, max_grid_size) {}

};



/**
 * Sorting enactor that is specialized for for 64-bit key types
 */
template <typename K, typename V>
class RadixSortingEnactor<K, V, unsigned long long> : public BaseRadixSortingEnactor<K, V>
{
protected:

	typedef BaseRadixSortingEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

	cudaError_t EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{
		Base::template DigitPlacePass<0,  4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(converted_storage);
		Base::template DigitPlacePass<1,  4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<2,  4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<3,  4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<4,  4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<5,  4, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<6,  4, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<7,  4, 28, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<8,  4, 32, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage);
		Base::template DigitPlacePass<9,  4, 36, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<10, 4, 40, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<11, 4, 44, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<12, 4, 48, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<13, 4, 52, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<14, 4, 56, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(converted_storage); 
		Base::template DigitPlacePass<15, 4, 60, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >    (converted_storage); 

		return cudaSuccess;
	}

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		num_elements 
	 * 		Length (in elements) of the input to a sorting operation
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	RadixSortingEnactor(unsigned int num_elements, int max_grid_size = 0) : Base::BaseRadixSortingEnactor(16, 4, num_elements, max_grid_size) {}

};


} // end namespace b40c_thrust
} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

