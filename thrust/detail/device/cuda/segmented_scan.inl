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


/*! \file segmented_scan.inl
 *  \brief Inline file for segmented_scan.h.
 */

#include <thrust/detail/config.h>

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/detail/device/cuda/arch.h>
#include <thrust/functional.h>

#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/detail/device/cuda/warp/scan.h>

#include <thrust/detail/device/dereference.h>
#include <thrust/detail/device/cuda/synchronize.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{
  
typedef unsigned int FlagType;

namespace segmented_scan
{


/////////////    
// Kernels //
/////////////    


template<typename InputType,
         typename FlagType,
         typename InputIterator,
         typename FlagIterator,
         typename AssociativeOperator>
         __device__
InputType segscan_warp1(const unsigned int thread_lane, InputType val, FlagType mindex, InputIterator sval, FlagIterator sflg, AssociativeOperator binary_op)
{
#if __CUDA_ARCH__ >= 120
    // optimization
    if (!__any(mindex))
        return thrust::detail::device::cuda::warp::scan(thread_lane, val, sval, binary_op);
#endif

    // (1) Convert head flags to min_indices form
    mindex = thrust::detail::device::cuda::warp::scan(thread_lane, mindex, sflg, thrust::maximum<FlagType>());

    // (2) Perform segmented scan across warp
    sval[threadIdx.x] = val;

    if (thread_lane >= mindex +  1)  sval[threadIdx.x] = val = binary_op(sval[threadIdx.x -  1], val);
    if (thread_lane >= mindex +  2)  sval[threadIdx.x] = val = binary_op(sval[threadIdx.x -  2], val);
    if (thread_lane >= mindex +  4)  sval[threadIdx.x] = val = binary_op(sval[threadIdx.x -  4], val);
    if (thread_lane >= mindex +  8)  sval[threadIdx.x] = val = binary_op(sval[threadIdx.x -  8], val);
    if (thread_lane >= mindex + 16)  sval[threadIdx.x] = val = binary_op(sval[threadIdx.x - 16], val);

    return val;
}

template<typename FlagType,
         typename InputIterator,
         typename FlagIterator,
         typename AssociativeOperator>
         __device__
void segscan_warp2(const unsigned int thread_lane, FlagType flg, InputIterator sval, FlagIterator sflg, AssociativeOperator binary_op)
{
  
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

//// currently broken
//#if __CUDA_ARCH__ >= 120
//    // optimization
//    if (!__any(flg))
//        thrust::detail::device::cuda::warp::scan(thread_lane, sval[threadIdx.x], sval, binary_op);
//#endif

    // (1) Convert head flags to min_indices form
    FlagType mindex = (flg) ? thread_lane : 0;
    mindex = thrust::detail::device::cuda::warp::scan(thread_lane, mindex, sflg, thrust::maximum<FlagType>());

    // (2) Perform segmented scan across warp
    if (thread_lane >= mindex +  1)  sval[threadIdx.x] = binary_op(sval[threadIdx.x -  1], sval[threadIdx.x]);
    if (thread_lane >= mindex +  2)  sval[threadIdx.x] = binary_op(sval[threadIdx.x -  2], sval[threadIdx.x]);
    if (thread_lane >= mindex +  4)  sval[threadIdx.x] = binary_op(sval[threadIdx.x -  4], sval[threadIdx.x]);
    if (thread_lane >= mindex +  8)  sval[threadIdx.x] = binary_op(sval[threadIdx.x -  8], sval[threadIdx.x]);
    if (thread_lane >= mindex + 16)  sval[threadIdx.x] = binary_op(sval[threadIdx.x - 16], sval[threadIdx.x]);
}


template<unsigned int block_size,
         typename OutputIterator,
         typename OutputType,
         typename AssociativeOperator>
__global__ void
inclusive_update_kernel(OutputIterator result,
                        AssociativeOperator binary_op,
                        const unsigned int n,
                        const unsigned int interval_size,
                        OutputType * carry_in,
                        unsigned int * segment_lengths)
{
    const unsigned int thread_id   = block_size * blockIdx.x + threadIdx.x;  // global thread index
    const unsigned int thread_lane = threadIdx.x & 31;                       // thread index within the warp
    const unsigned int warp_id     = thread_id   / 32;                       // global warp index

    const unsigned int interval_begin = warp_id * interval_size;                   // beginning of this warp's segment
    const unsigned int interval_end   = interval_begin + segment_lengths[warp_id]; // end of this warp's segment
    
    if(warp_id == 0 || interval_begin >= n) return;                         // nothing to do

    OutputType carry = carry_in[warp_id - 1];                                // value to add to this segment

    for(unsigned int i = interval_begin + thread_lane; i < interval_end; i += 32)
    {
        thrust::detail::device::dereference(result, i) = binary_op(carry, thrust::detail::device::dereference(result, i));
    }
}


template<unsigned int block_size,
         typename OutputIterator,
         typename OutputType,
         typename AssociativeOperator>
__global__ void
exclusive_update_kernel(OutputIterator result,
                        OutputType init,
                        AssociativeOperator binary_op,
                        const unsigned int n,
                        const unsigned int interval_size,
                        OutputType * carry_in,
                        unsigned int * segment_lengths)
                        
{
    const unsigned int thread_id   = block_size * blockIdx.x + threadIdx.x;  // global thread index
    const unsigned int thread_lane = threadIdx.x & 31;                       // thread index within the warp
    const unsigned int warp_id     = thread_id   / 32;                       // global warp index

    const unsigned int interval_begin = warp_id * interval_size;                   // beginning of this warp's segment
    const unsigned int interval_end   = interval_begin + segment_lengths[warp_id]; // end of this warp's segment
    
    if(warp_id == 0 || interval_begin >= n) return;                                // nothing to do

    OutputType carry = binary_op(init, carry_in[warp_id - 1]);                      // value to add to this segment

    unsigned int i = interval_begin + thread_lane;

    if(i < interval_end)
    {
        OutputType val = thrust::detail::device::dereference(result, i);
        
        if (thread_lane == 0)
            val = carry;
        else
            val = binary_op(carry, val);
        
        thrust::detail::device::dereference(result, i) = val;
    }

    for(i += 32; i < interval_end; i += 32)
    {
        OutputType val = thrust::detail::device::dereference(result, i);

        val = binary_op(carry, val);

        thrust::detail::device::dereference(result, i) = val;
    }
}



/* Perform an inclusive scan on separate intervals
 *
 * For intervals of length 2:
 *    [ a, b, c, d, ... ] -> [ a, a+b, c, c+d, ... ]
 *
 * Each warp is assigned an interval of [first, first + n)
 */
template<unsigned int block_size,
         typename InputIterator1, 
         typename InputIterator2,
         typename OutputIterator,
         typename AssociativeOperator,
         typename BinaryPredicate,
         typename OutputType>
__global__ 
void inclusive_scan_kernel(InputIterator1 first1,
                           InputIterator2 first2,
                           OutputIterator result,
                           AssociativeOperator binary_op,
                           BinaryPredicate pred,
                           const unsigned int n,
                           const unsigned int interval_size,
                           OutputType * final_val,
                           unsigned int * segment_lengths)
{
  typedef typename thrust::iterator_traits<InputIterator2>::value_type KeyType;

  // XXX warpSize exists, but is not known at compile time,
  //     so define our own constant
  const unsigned int warp_size = 32;

  //__shared__ volatile OutputType sval[block_size];
  //__shared__ volatile KeyType    skey[block_size];
  __shared__ unsigned char sval_workaround[block_size * sizeof(OutputType)];
  __shared__ unsigned char skey_workaround[block_size * sizeof(KeyType)];
  OutputType * sval = reinterpret_cast<OutputType*>(sval_workaround);
  KeyType    * skey = reinterpret_cast<KeyType*>(skey_workaround);
  __shared__ FlagType    sflg[block_size];

  const unsigned int thread_id   = block_size * blockIdx.x + threadIdx.x;      // global thread index
  const unsigned int thread_lane = threadIdx.x & (warp_size - 1);              // thread index within the warp
  const unsigned int warp_id     = thread_id   / warp_size;                    // global warp index

  const unsigned int interval_begin = warp_id * interval_size;                 // beginning of this warp's segment
  const unsigned int interval_end   = min(interval_begin + interval_size, n);  // end of this warp's segment

  unsigned int i = interval_begin + thread_lane;                               // initial thread starting position

  unsigned int first_segment_end = interval_end;                               // length of initial segment in this interval

  if(interval_begin >= interval_end)                                           // warp has nothing to do
      return;

  FlagType   mindex;

  if(i < interval_end)
  {
      OutputType val = thrust::detail::device::dereference(first1, i);
      KeyType    key = thrust::detail::device::dereference(first2, i);

      // compute head flags
      skey[threadIdx.x] = key;
      if (thread_lane == 0)
      {
          if(warp_id == 0 || !pred(thrust::detail::device::dereference(first2, i - 1), key))
              first_segment_end = i;
          mindex = thread_lane;
      }
      else if (pred(skey[threadIdx.x - 1], key))
      {
          mindex = 0;
      }
      else
      {
          first_segment_end = i;
          mindex = thread_lane;
      }

      val = segscan_warp1(thread_lane, val, mindex, sval, sflg, binary_op);

      thrust::detail::device::dereference(result, i) = val;
      
      i += 32;
  }
     
  
  while(i < interval_end)
  {
      OutputType val = thrust::detail::device::dereference(first1, i);
      KeyType    key = thrust::detail::device::dereference(first2, i);

      if (thread_lane == 0)
      {
          if(pred(skey[threadIdx.x + 31], key))
              val = binary_op(sval[threadIdx.x + 31], val);                    // segment spans warp boundary
          else
              first_segment_end = min(first_segment_end, i);                   // new segment begins here
      }

      // compute head flags
      skey[threadIdx.x] = key;
      if (thread_lane == 0 || pred(skey[threadIdx.x - 1], key))
      {
          mindex = 0;
      }
      else
      {
          first_segment_end = min(first_segment_end, i);
          mindex = thread_lane;
      }

      val = segscan_warp1(thread_lane, val, mindex, sval, sflg, binary_op);

      thrust::detail::device::dereference(result, i) = val;
      
      i += 32;
  }
   
  // write out final value
  if (i == interval_end + 31)
  {
      final_val[warp_id] = sval[threadIdx.x];
  }

  // compute first segment boundary
  first_segment_end = thrust::detail::device::cuda::warp::scan(thread_lane, first_segment_end, sflg, thrust::minimum<FlagType>());

  // write out initial segment length
  if (thread_lane == 31)
      segment_lengths[warp_id] = first_segment_end - interval_begin;

//  /// XXX BEGIN TEST
//  if(thread_lane == 0){
//    unsigned int initial_segment_length = interval_end - interval_begin;
//
//    OutputType sum = thrust::detail::device::dereference(first1, i);
//    thrust::detail::device::dereference(result, i) = sum;
//
//    i++;
//    while( i < interval_end ){
//        if (pred(thrust::detail::device::dereference(first2, i - 1), thrust::detail::device::dereference(first2, i)))
//        {
//            sum = binary_op(sum, thrust::detail::device::dereference(first1, i));
//        }
//        else 
//        {
//            sum = thrust::detail::device::dereference(first1, i);
//            initial_segment_length = min(initial_segment_length, i - interval_begin);
//        }
//
//        thrust::detail::device::dereference(result, i) = sum;
//        i++;
//    }
//
//    if (warp_id > 0 && !pred(thrust::detail::device::dereference(first2, interval_begin - 1), 
//                             thrust::detail::device::dereference(first2, interval_begin)))
//        initial_segment_length = 0; // segment does not overlap interval boundary
//    
//    final_val[warp_id] = sum;
//    segment_lengths[warp_id] = initial_segment_length;
//  }
//  // XXX END TEST

} // end kernel()



/* Perform an exclusive scan on separate intervals
 *
 * For intervals of length 3:
 *    [ a, b, c, d, ... ] -> [ init, a, a+b, init, c, ... ]
 *
 * Each warp is assigned an interval of [first, first + n)
 */
template<unsigned int block_size,
         typename InputIterator1, 
         typename InputIterator2,
         typename OutputIterator,
         typename AssociativeOperator,
         typename BinaryPredicate,
         typename OutputType>
__global__ 
void exclusive_scan_kernel(InputIterator1 first1,
                           InputIterator2 first2,
                           OutputIterator result,
                           OutputType init,
                           AssociativeOperator binary_op,
                           BinaryPredicate pred,
                           const unsigned int n,
                           const unsigned int interval_size,
                           OutputType * final_val,
                           unsigned int * segment_lengths)
{
  typedef typename thrust::iterator_traits<InputIterator2>::value_type KeyType;

  // XXX warpSize exists, but is not known at compile time,
  //     so define our own constant
  const unsigned int warp_size = 32;

  //__shared__ volatile OutputType sval[block_size];
  //__shared__ volatile KeyType    skey[block_size];
  __shared__ unsigned char sval_workaround[block_size * sizeof(OutputType)];
  __shared__ unsigned char skey_workaround[block_size * sizeof(KeyType)];
  OutputType * sval = reinterpret_cast<OutputType*>(sval_workaround);
  KeyType    * skey = reinterpret_cast<KeyType*>(skey_workaround);
  __shared__ FlagType    sflg[block_size];

  const unsigned int thread_id   = block_size * blockIdx.x + threadIdx.x;      // global thread index
  const unsigned int thread_lane = threadIdx.x & (warp_size - 1);              // thread index within the warp
  const unsigned int warp_id     = thread_id   / warp_size;                    // global warp index

  const unsigned int interval_begin = warp_id * interval_size;                 // beginning of this warp's segment
  const unsigned int interval_end   = min(interval_begin + interval_size, n);  // end of this warp's segment

  unsigned int i = interval_begin + thread_lane;                               // initial thread starting position

  unsigned int first_segment_end = interval_end;                               // length of initial segment in this interval

  if(interval_begin >= interval_end)                                           // warp has nothing to do
      return;
  
  OutputType val;
  KeyType    key;
  FlagType   flg;


  if(i < interval_end)
  {
      sval[threadIdx.x] = thrust::detail::device::dereference(first1, i);
      skey[threadIdx.x] = thrust::detail::device::dereference(first2, i);

      // compute head flags
      if (thread_lane == 0)
      {
          if(warp_id == 0 || !pred(thrust::detail::device::dereference(first2, i - 1), skey[threadIdx.x]))
              first_segment_end = i;
          flg = 1;
      }
      else if (pred(skey[threadIdx.x - 1], skey[threadIdx.x]))
      {
          flg = 0;
      }
      else
      {
          first_segment_end = i;
          flg = 1;
      }

      segscan_warp2(thread_lane, flg, sval, sflg, binary_op);
  
      first_segment_end = thrust::detail::device::cuda::warp::scan(thread_lane, first_segment_end, sflg, thrust::minimum<FlagType>());
      
      if (thread_lane != 0)
          val = sval[threadIdx.x - 1]; // value to the left

      if (flg)
          val = init;
      else if (first_segment_end < i)
          val = binary_op(init, val);

      // when thread_lane == 0 and warp_id != 0, result is bogus
      thrust::detail::device::dereference(result, i) = val;
      
      i += 32;
  }
     
  
  while(i < interval_end)
  {
      if (thread_lane == 0)
      {
          first_segment_end = sflg[threadIdx.x + 31];
          val = sval[threadIdx.x + 31];
          key = skey[threadIdx.x + 31];
      }
             
      sval[threadIdx.x] = thrust::detail::device::dereference(first1, i);
      skey[threadIdx.x] = thrust::detail::device::dereference(first2, i);

      if (thread_lane == 0 && pred(key, skey[threadIdx.x]))
          sval[threadIdx.x] = binary_op(val, sval[threadIdx.x]);           // segment spans warp boundary
      else
          key = skey[threadIdx.x - 1];

      // compute head flags
      if(pred(key, skey[threadIdx.x]))
      {
          flg = 0;
      }
      else
      {
          flg = 1;
          first_segment_end = min(first_segment_end, i);
      }

      segscan_warp2(thread_lane, flg, sval, sflg, binary_op);

      first_segment_end = thrust::detail::device::cuda::warp::scan(thread_lane, first_segment_end, sflg, thrust::minimum<FlagType>());

      if (thread_lane != 0)
          val = sval[threadIdx.x - 1]; // value to the left

      if (flg)
          val = init;
      else if (first_segment_end < i)
          val = binary_op(init, val);

      thrust::detail::device::dereference(result, i) = val;
     
      i += 32;
  }
   
  // write out final value
  if (i == interval_end + 31)
  {
      final_val[warp_id] = sval[threadIdx.x];
  }

  // compute first segment boundary
  first_segment_end = thrust::detail::device::cuda::warp::scan(thread_lane, first_segment_end, sflg, thrust::minimum<FlagType>());

  // write out initial segment length
  if (thread_lane == 31)
      segment_lengths[warp_id] = first_segment_end - interval_begin;


//  /// XXX BEGIN TEST
//  if(thread_lane == 0){
//    unsigned int initial_segment_length = interval_end - interval_begin;
//
//    OutputType temp = thrust::detail::device::dereference(first1, i);
//    OutputType next;
//    
//    if (warp_id == 0 || !pred(thrust::detail::device::dereference(first2, interval_begin - 1), 
//                              thrust::detail::device::dereference(first2, interval_begin)))
//
//    {
//        initial_segment_length = 0; // segment does not overlap interval boundary
//        next = binary_op(init, temp);
//        thrust::detail::device::dereference(result, i) = init;
//    }
//    else
//    {
//        next = temp;
//        //thrust::detail::device::dereference(result, i) = ???; // no value to put here
//    }
//      
//
//    i++;
//
//    while( i < interval_end ){
//        temp = thrust::detail::device::dereference(first1, i);
// 
//        if (!pred(thrust::detail::device::dereference(first2, i - 1), thrust::detail::device::dereference(first2, i)))
//        {
//            next = init;
//            initial_segment_length = min(initial_segment_length, i - interval_begin);
//        }
//
//        thrust::detail::device::dereference(result, i) = next;
//        
//        next = binary_op(next, temp);
//
//        i++;
//    }
//
//    
//    final_val[warp_id] = next;
//    segment_lengths[warp_id] = initial_segment_length;
//  }
//  // XXX END TEST

} // end kernel()




struct __segment_spans_interval
{
    const unsigned int interval_size;

    __segment_spans_interval(const int _interval_size) : interval_size(_interval_size) {}
    template <typename T>
    __host__ __device__
    bool operator()(const T& a, const T& b) const
    {
        return b == interval_size;
    }
};

} // end namespace segmented_scan




//////////////////
// Entry Points //
//////////////////

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename AssociativeOperator,
         typename BinaryPredicate>
  OutputIterator inclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          AssociativeOperator binary_op,
                                          BinaryPredicate pred)
{
    typedef typename thrust::iterator_traits<InputIterator2>::value_type KeyType;
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    if(first1 == last1) 
        return result;
    
    const size_t n = last1 - first1;
    
    const unsigned int warp_size  = 32;
    
    // 16KB (max) - 1KB (upper bound on what's used for other purposes)
    const size_t max_smem_size = 15 * 1025; 

    // largest 2^N that fits in SMEM
    static const size_t blocksize_limit1 = 1 << thrust::detail::mpl::math::log2< (max_smem_size/ (sizeof(OutputType) + sizeof(KeyType) + sizeof(FlagType))) >::value;
    static const size_t blocksize_limit2 = 256;

    static const size_t block_size = (blocksize_limit1 < blocksize_limit2) ? blocksize_limit1 : blocksize_limit2;

    const unsigned int max_blocks = thrust::detail::device::cuda::arch::max_active_threads()/block_size;
    const unsigned int warps_per_block = block_size/warp_size;

    const unsigned int num_units  = thrust::detail::util::divide_ri(n, warp_size);
    const unsigned int num_warps  = std::min(num_units, warps_per_block * max_blocks);
    const unsigned int num_blocks = thrust::detail::util::divide_ri(num_warps,warps_per_block);
    const unsigned int num_iters  = thrust::detail::util::divide_ri(num_units, num_warps);          // number of times each warp iterates, interval length is 32*num_iters

    const unsigned int interval_size = warp_size * num_iters;

    // create a temp vector for per-warp results
    thrust::detail::raw_cuda_device_buffer<OutputType>   d_final_val(num_warps + 1);
    thrust::detail::raw_cuda_device_buffer<unsigned int> d_segment_lengths(num_warps + 1);

    //////////////////////
    // first level scan
    segmented_scan::inclusive_scan_kernel<block_size> <<<num_blocks, block_size>>>
        (first1, first2, result, binary_op, pred, n, interval_size, raw_pointer_cast(&d_final_val[0]), raw_pointer_cast(&d_segment_lengths[0]));
    synchronize_if_enabled("inclusive_scan_kernel");

    ///////////////////////
    // second level scan
    // scan final_val on the device (use one warp of GPU method for second level scan)
    segmented_scan::inclusive_scan_kernel<warp_size> <<<1, warp_size>>>
        (raw_pointer_cast(&d_final_val[0]), raw_pointer_cast(&d_segment_lengths[0]), raw_pointer_cast(&d_final_val[0]), binary_op, segmented_scan::__segment_spans_interval(interval_size),
         num_warps, num_warps, raw_pointer_cast(&d_final_val[num_warps]), raw_pointer_cast(&d_segment_lengths[num_warps]));
    synchronize_if_enabled("inclusive_scan_kernel");
        
    //////////////////////
    // update intervals
    segmented_scan::inclusive_update_kernel<block_size> <<<num_blocks, block_size>>>
        (result, binary_op, n, interval_size, raw_pointer_cast(&d_final_val[0]), raw_pointer_cast(&d_segment_lengths[0]));
    synchronize_if_enabled("inclusive_update_kernel");

    return result + n;
} // end inclusive_segmented_scan()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator,
         typename BinaryPredicate>
  OutputIterator exclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          const T init,
                                          AssociativeOperator binary_op,
                                          BinaryPredicate pred)
{
    typedef typename thrust::iterator_traits<InputIterator2>::value_type KeyType;
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    if(first1 == last1) 
        return result;
    
    const size_t n = last1 - first1;
    
    const unsigned int warp_size  = 32;
    
    // 16KB (max) - 1KB (upper bound on what's used for other purposes)
    const size_t max_smem_size = 15 * 1025; 

    // largest 2^N that fits in SMEM
    static const size_t blocksize_limit1 = 1 << thrust::detail::mpl::math::log2< (max_smem_size/ (sizeof(OutputType) + sizeof(KeyType) + sizeof(FlagType))) >::value;
    static const size_t blocksize_limit2 = 256;
    
    static const size_t block_size = (blocksize_limit1 < blocksize_limit2) ? blocksize_limit1 : blocksize_limit2;

    const unsigned int max_blocks = thrust::detail::device::cuda::arch::max_active_threads()/block_size;
    const unsigned int warps_per_block = block_size/warp_size;

    const unsigned int num_units  = thrust::detail::util::divide_ri(n, warp_size);
    const unsigned int num_warps  = std::min(num_units, warps_per_block * max_blocks);
    const unsigned int num_blocks = thrust::detail::util::divide_ri(num_warps,warps_per_block);
    const unsigned int num_iters  = thrust::detail::util::divide_ri(num_units, num_warps);          // number of times each warp iterates, interval length is 32*num_iters

    const unsigned int interval_size = warp_size * num_iters;

    // create a temp vector for per-warp results
    thrust::detail::raw_cuda_device_buffer<OutputType>   d_final_val(num_warps + 1);
    thrust::detail::raw_cuda_device_buffer<unsigned int> d_segment_lengths(num_warps + 1);

    //////////////////////
    // first level scan
    segmented_scan::exclusive_scan_kernel<block_size> <<<num_blocks, block_size>>>
        (first1, first2, result, OutputType(init), binary_op, pred, n, interval_size, raw_pointer_cast(&d_final_val[0]), raw_pointer_cast(&d_segment_lengths[0]));
    synchronize_if_enabled("exclusive_scan_kernel");

    ///////////////////////
    // second level scan
    // scan final_val on the device (use one warp of GPU method for second level scan)
    segmented_scan::inclusive_scan_kernel<warp_size> <<<1, warp_size>>>
        (raw_pointer_cast(&d_final_val[0]), raw_pointer_cast(&d_segment_lengths[0]), raw_pointer_cast(&d_final_val[0]), binary_op, segmented_scan::__segment_spans_interval(interval_size),
         num_warps, num_warps, raw_pointer_cast(&d_final_val[num_warps]), raw_pointer_cast(&d_segment_lengths[num_warps]));
    synchronize_if_enabled("inclusive_scan_kernel");
        
    //////////////////////
    // update intervals
    segmented_scan::exclusive_update_kernel<block_size> <<<num_blocks, block_size>>>
        (result, OutputType(init), binary_op, n, interval_size, raw_pointer_cast(&d_final_val[0]), raw_pointer_cast(&d_segment_lengths[0]));
    synchronize_if_enabled("exclusive_update_kernel");
    
    return result + n;
} // end exclusive_interval_scan()


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER_NVCC

