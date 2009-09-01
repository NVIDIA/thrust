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


/*! \file scan.inl
 *  \brief Inline file for scan.h.
 */

// do not attempt to compile this file with any other compiler
#ifdef __CUDACC__


#include <thrust/experimental/arch.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/raw_buffer.h>
#include <thrust/copy.h>

#include <thrust/detail/util/blocking.h>

#include <thrust/detail/device/dereference.h>

#include <thrust/detail/mpl/math.h> // for log2<N>

// warpwise method fails in some cases
//#define USE_WARPWISE_SCAN

namespace thrust
{

namespace detail
{

// forward declaration of raw_device_buffer
template<typename> class raw_device_buffer;

namespace device
{

namespace cuda
{

//namespace interval_scan
//{

/////////////    
// Kernels //
/////////////    

// XXX replace with thrust::detail::warp::scan() after perf testing
//template<typename T, 
//         typename AssociativeOperator>
//         __device__
//void scan_warp(const unsigned int& thread_lane, volatile T * sdata, AssociativeOperator binary_op)
//{
//    // the use of 'volatile' is a workaround so that nvcc doesn't reorder the following lines
//    if (thread_lane >=  1)  sdata[threadIdx.x] = binary_op((T &) sdata[threadIdx.x -  1] , (T &) sdata[threadIdx.x]);
//    if (thread_lane >=  2)  sdata[threadIdx.x] = binary_op((T &) sdata[threadIdx.x -  2] , (T &) sdata[threadIdx.x]);
//    if (thread_lane >=  4)  sdata[threadIdx.x] = binary_op((T &) sdata[threadIdx.x -  4] , (T &) sdata[threadIdx.x]);
//    if (thread_lane >=  8)  sdata[threadIdx.x] = binary_op((T &) sdata[threadIdx.x -  8] , (T &) sdata[threadIdx.x]);
//    if (thread_lane >= 16)  sdata[threadIdx.x] = binary_op((T &) sdata[threadIdx.x - 16] , (T &) sdata[threadIdx.x]);
//}

template<typename InputType, 
         typename InputIterator, 
         typename AssociativeOperator>
         __device__
void scan_warp(const unsigned int& thread_lane, InputType& val, InputIterator sdata, AssociativeOperator binary_op)
{
    sdata[threadIdx.x] = val;

    if (thread_lane >=  1)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x -  1], val);  __threadfence_block();
    if (thread_lane >=  2)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x -  2], val);  __threadfence_block(); 
    if (thread_lane >=  4)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x -  4], val);  __threadfence_block();
    if (thread_lane >=  8)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x -  8], val);  __threadfence_block();
    if (thread_lane >= 16)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x - 16], val);  __threadfence_block();
}

template<unsigned int BLOCK_SIZE,
         typename OutputIterator,
         typename AssociativeOperator>
__global__ void
inclusive_update_kernel(OutputIterator result,
                        AssociativeOperator binary_op,
                        const unsigned int n,
                        const unsigned int interval_size,
                        typename thrust::iterator_traits<OutputIterator>::value_type * carry_in)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    const unsigned int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;        // global thread index
    const unsigned int thread_lane = threadIdx.x & 31;                             // thread index within the warp
    const unsigned int warp_id     = thread_id   / 32;                             // global warp index
    
    const unsigned int interval_begin = warp_id * interval_size;                   // beginning of this warp's segment
    const unsigned int interval_end   = min(interval_begin + interval_size, n);    // end of this warp's segment
    
    if(interval_begin == 0 || interval_begin >= n) return;                         // nothing to do

    OutputType carry = carry_in[warp_id - 1];                                      // value to add to this segment

    for(unsigned int i = interval_begin + thread_lane; i < interval_end; i += 32){
        thrust::detail::device::dereference(result, i) = binary_op(carry, thrust::detail::device::dereference(result, i));
    }
}


template<unsigned int BLOCK_SIZE,
         typename OutputIterator,
         typename AssociativeOperator>
__global__ void
exclusive_update_kernel(OutputIterator result,
                        typename thrust::iterator_traits<OutputIterator>::value_type init,
                        AssociativeOperator binary_op,
                        const unsigned int n,
                        const unsigned int interval_size,
                        typename thrust::iterator_traits<OutputIterator>::value_type * carry_in)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    // XXX workaround types with constructors in __shared__ memory
    //__shared__ OutputType sdata[BLOCK_SIZE];
    __shared__ unsigned char sdata_workaround[BLOCK_SIZE * sizeof(OutputType)];
    OutputType *sdata = reinterpret_cast<OutputType*>(sdata_workaround);

#if defined(USE_WARPWISE_SCAN)
    const unsigned int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;        // global thread index
    const unsigned int thread_lane = threadIdx.x & 31;                             // thread index within the warp
    const unsigned int warp_id     = thread_id   / 32;                             // global warp index
    
    const unsigned int interval_begin = warp_id * interval_size;                   // beginning of this warp's segment
    const unsigned int interval_end   = min(interval_begin + interval_size, n);    // end of this warp's segment
    
    if(interval_begin >= n) return;                                                // nothing to do

    OutputType carry = (warp_id == 0) ? init : binary_op(init, carry_in[warp_id - 1]);  // value to add to this segment
    OutputType val   = carry;

    for(unsigned int i = interval_begin + thread_lane; i < interval_end; i += 32){
        sdata[threadIdx.x] = binary_op(carry, thrust::detail::device::dereference(result, i));

        thrust::detail::device::dereference(result, i) = (thread_lane == 0) ? val : sdata[threadIdx.x - 1]; 

        if(thread_lane == 0)
            val = sdata[threadIdx.x + 31];
    }
#else
    const unsigned int interval_begin = blockIdx.x * interval_size;                // beginning of this block's segment
    const unsigned int interval_end   = min(interval_begin + interval_size, n);    // end of this block's segment

    OutputType carry = (blockIdx.x == 0) ? init : binary_op(init, carry_in[blockIdx.x - 1]);  // value to add to this segment
    OutputType val   = carry;

    for(unsigned int base = interval_begin; base < interval_end; base += BLOCK_SIZE)
    {
        const unsigned int i = base + threadIdx.x;

        if(i < interval_end)
            sdata[threadIdx.x] = binary_op(carry, thrust::detail::device::dereference(result, i));

        __syncthreads();

        if (threadIdx.x != 0)
            val = sdata[threadIdx.x - 1];

        if (i < interval_end)
            thrust::detail::device::dereference(result, i) = val;

        if(threadIdx.x == 0)
            val = sdata[threadIdx.x + BLOCK_SIZE - 1];
    }
#endif // define(USE_WARPWISE_SCAN)
}


/* Perform an inclusive scan on separate intervals
 *
 * For intervals of length 2:
 *    [ a, b, c, d, ... ] -> [ a, a+b, c, c+d, ... ]
 *
 * Each warp is assigned an interval of [first, first + n)
 */
template<unsigned int BLOCK_SIZE,
         typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
__global__ void
scan_kernel(InputIterator first,
            const unsigned int n,
            OutputIterator result,
            AssociativeOperator binary_op,
            const unsigned int interval_size,
            typename thrust::iterator_traits<OutputIterator>::value_type * final_carry)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  // XXX warpSize exists, but is not known at compile time,
  //     so define our own constant
  const unsigned int WARP_SIZE = 32;

  //__shared__ volatile OutputType sdata[BLOCK_SIZE];
  __shared__ unsigned char sdata_workaround[BLOCK_SIZE * sizeof(OutputType)];
  OutputType *sdata = reinterpret_cast<OutputType*>(sdata_workaround);
  
  const unsigned int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const unsigned int thread_lane = threadIdx.x & (WARP_SIZE - 1);          // thread index within the warp
  const unsigned int warp_id     = thread_id   / WARP_SIZE;                // global warp index

  const unsigned int interval_begin = warp_id * interval_size;                 // beginning of this warp's segment
  const unsigned int interval_end   = min(interval_begin + interval_size, n);  // end of this warp's segment

  unsigned int i = interval_begin + thread_lane;                               // initial thread starting position

  // nothing to do
  if(i >= interval_end)
      return;

//  /// XXX BEGIN TEST
//  if(thread_lane == 0){
//    end = min(base + interval_size, n);
//
//    OutputType sum = thrust::detail::device::dereference(first, i);
//    thrust::detail::device::dereference(result, i) = sum;
//
//    i++;
//    while( i < end ){
//        sum = binary_op(sum, thrust::detail::device::dereference(first, i));
//        thrust::detail::device::dereference(result, i) = sum;
//        i++;
//    }
//    final_carry[warp_id] = sum;
//  }
//  // XXX END TEST


  // First iteration has no carry in
  if(i < interval_end){
      OutputType val = thrust::detail::device::dereference(first, i);

      scan_warp(thread_lane, val, sdata, binary_op);

      thrust::detail::device::dereference(result, i) = val;

      i += WARP_SIZE;
  }

  // Remaining iterations have carry in
  while(i < interval_end){
      OutputType val = thrust::detail::device::dereference(first, i);

      if (thread_lane == 0)
          val = binary_op(sdata[threadIdx.x + (WARP_SIZE - 1)], val);

      scan_warp(thread_lane, val, sdata, binary_op);

      thrust::detail::device::dereference(result, i) = val;

      i += WARP_SIZE;
  }

  if (i == interval_end + (WARP_SIZE - 1))
      final_carry[warp_id] = sdata[threadIdx.x];

} // end scan_kernel()


//} // end namespace interval_scan




//////////////////
// Entry Points //
//////////////////

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
  
    const size_t n = last - first;

    if( n == 0 ) 
        return result;
    
    // 16KB (max) - 1KB (upper bound on what's used for other purposes)
    const size_t MAX_SMEM_SIZE = 15 * 1024; 

    const unsigned int WARP_SIZE  = 32;

    // largest 2^N that fits in SMEM
    static const size_t BLOCKSIZE_LIMIT1 = 1 << thrust::detail::mpl::math::log2< (MAX_SMEM_SIZE/sizeof(OutputType)) >::value;
    static const size_t BLOCKSIZE_LIMIT2 = 256;
    static const size_t BLOCK_SIZE = (BLOCKSIZE_LIMIT1 < BLOCKSIZE_LIMIT2) ? BLOCKSIZE_LIMIT1 : BLOCKSIZE_LIMIT2;

    const unsigned int MAX_BLOCKS = thrust::experimental::arch::max_active_blocks(scan_kernel<BLOCK_SIZE, InputIterator, OutputIterator, AssociativeOperator>, BLOCK_SIZE, (size_t) 0);
    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE/WARP_SIZE;

    const unsigned int num_units  = thrust::detail::util::divide_ri(n, WARP_SIZE);
    const unsigned int num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
    const unsigned int num_blocks = thrust::detail::util::divide_ri(num_warps,WARPS_PER_BLOCK);
    const unsigned int num_iters  = thrust::detail::util::divide_ri(num_units, num_warps);          // number of times each warp iterates, interval length is 32*num_iters

    const unsigned int interval_size = WARP_SIZE * num_iters;

    // create a temp vector for per-warp results
    thrust::detail::raw_device_buffer<OutputType> d_carry_out(num_warps);

    //////////////////////
    // first level scan
    scan_kernel<BLOCK_SIZE> <<<num_blocks, BLOCK_SIZE>>>
        (first, n, result, binary_op, interval_size, raw_pointer_cast(&d_carry_out[0]));

    ///////////////////////
    // second level scan
    // scan carry_out on the device (use one warp of GPU method for second level scan)
    scan_kernel<WARP_SIZE> <<<1, WARP_SIZE>>>
        (raw_pointer_cast(&d_carry_out[0]), num_warps, raw_pointer_cast(&d_carry_out[0]), binary_op, num_warps, raw_pointer_cast(&*(d_carry_out.end() - 1)));

    //////////////////////
    // update intervals
    inclusive_update_kernel<BLOCK_SIZE> <<<num_blocks, BLOCK_SIZE>>>
        (result, binary_op, n, interval_size, raw_pointer_cast(&d_carry_out[0]));

    return result + n;
} // end inclusive_scan()



template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
  
    const size_t n = last - first;

    if( n == 0 )
        return result;
    
    // 16KB (max) - 1KB (upper bound on what's used for other purposes)
    const size_t MAX_SMEM_SIZE = 15 * 1024; 

    const unsigned int WARP_SIZE  = 32;

    // largest 2^N that fits in SMEM
    static const size_t BLOCKSIZE_LIMIT1 = 1 << thrust::detail::mpl::math::log2< (MAX_SMEM_SIZE/sizeof(OutputType)) >::value;
    static const size_t BLOCKSIZE_LIMIT2 = 256;
    static const size_t BLOCK_SIZE = (BLOCKSIZE_LIMIT1 < BLOCKSIZE_LIMIT2) ? BLOCKSIZE_LIMIT1 : BLOCKSIZE_LIMIT2;

    const unsigned int MAX_BLOCKS = thrust::experimental::arch::max_active_blocks(scan_kernel<BLOCK_SIZE, InputIterator, OutputIterator, AssociativeOperator>, BLOCK_SIZE, (size_t) 0);
    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE/WARP_SIZE;

    const unsigned int num_units  = thrust::detail::util::divide_ri(n, WARP_SIZE);
    const unsigned int num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
    const unsigned int num_blocks = thrust::detail::util::divide_ri(num_warps,WARPS_PER_BLOCK);
    const unsigned int num_iters  = thrust::detail::util::divide_ri(num_units, num_warps);

    const unsigned int interval_size = WARP_SIZE * num_iters;

    // create a temp vector for per-warp results
    thrust::detail::raw_device_buffer<OutputType> d_carry_out(num_warps);

    //////////////////////
    // first level scan
    scan_kernel<BLOCK_SIZE> <<<num_blocks, BLOCK_SIZE>>>
        (first, n, result, binary_op, interval_size, raw_pointer_cast(&d_carry_out[0]));

    ///////////////////////
    // second level scan
    // scan carry_out on the device (use one warp of GPU method for second level scan)
    scan_kernel<WARP_SIZE> <<<1, WARP_SIZE>>>
        (raw_pointer_cast(&d_carry_out[0]), num_warps, raw_pointer_cast(&d_carry_out[0]), binary_op, num_warps, raw_pointer_cast(&*(d_carry_out.end() - 1)));

    //////////////////////
    // update intervals
#if defined(USE_WARPWISE_SCAN)
    exclusive_update_kernel<BLOCK_SIZE> <<<num_blocks, BLOCK_SIZE>>>
        (result, OutputType(init), binary_op, n, interval_size, raw_pointer_cast(&d_carry_out[0]));
#else
    exclusive_update_kernel<BLOCK_SIZE> <<<num_warps, BLOCK_SIZE>>>
        (result, OutputType(init), binary_op, n, interval_size, raw_pointer_cast(&d_carry_out[0]));
#endif // defined(USE_WARPWISE_SCAN)

    return result + n;
} // end exclusive_scan()


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // __CUDACC__

