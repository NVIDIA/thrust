/*
 *  Copyright 2008-2012 NVIDIA Corporation
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


// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/uninitialized_array.h>

#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>

#include <thrust/detail/backend/dereference.h>
#include <thrust/detail/backend/cuda/arch.h>
#include <thrust/detail/backend/cuda/synchronize.h>
#include <thrust/detail/backend/cuda/reduce_intervals.h>
#include <thrust/detail/backend/cuda/default_decomposition.h>

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN


namespace thrust
{
namespace detail
{

// forward declaration of uninitialized_array
template<typename,typename> class uninitialized_array;

namespace backend
{
namespace cuda
{
namespace detail
{
namespace fast_scan
{

template <unsigned int CTA_SIZE,
          typename SharedArray,
          typename BinaryFunction>
          __device__ __forceinline__
void scan_block(SharedArray array, BinaryFunction binary_op)
{
    typedef typename thrust::iterator_value<SharedArray>::type T;

    T val = array[threadIdx.x];

    if (CTA_SIZE >    1) { if(threadIdx.x >=    1) { T tmp = array[threadIdx.x -    1]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >    2) { if(threadIdx.x >=    2) { T tmp = array[threadIdx.x -    2]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >    4) { if(threadIdx.x >=    4) { T tmp = array[threadIdx.x -    4]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >    8) { if(threadIdx.x >=    8) { T tmp = array[threadIdx.x -    8]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   16) { if(threadIdx.x >=   16) { T tmp = array[threadIdx.x -   16]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   32) { if(threadIdx.x >=   32) { T tmp = array[threadIdx.x -   32]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   64) { if(threadIdx.x >=   64) { T tmp = array[threadIdx.x -   64]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  128) { if(threadIdx.x >=  128) { T tmp = array[threadIdx.x -  128]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  256) { if(threadIdx.x >=  256) { T tmp = array[threadIdx.x -  256]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }  
    if (CTA_SIZE >  512) { if(threadIdx.x >=  512) { T tmp = array[threadIdx.x -  512]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }  
    if (CTA_SIZE > 1024) { if(threadIdx.x >= 1024) { T tmp = array[threadIdx.x - 1024]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }  
}

template <unsigned int CTA_SIZE,
          typename SharedArray,
          typename BinaryFunction>
          __device__ __forceinline__
void scan_block_n(SharedArray array, const unsigned int n, BinaryFunction binary_op)
{
    typedef typename thrust::iterator_value<SharedArray>::type T;

    T val = array[threadIdx.x];

    if (CTA_SIZE >    1) { if(threadIdx.x < n && threadIdx.x >=    1) { T tmp = array[threadIdx.x -    1]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >    2) { if(threadIdx.x < n && threadIdx.x >=    2) { T tmp = array[threadIdx.x -    2]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >    4) { if(threadIdx.x < n && threadIdx.x >=    4) { T tmp = array[threadIdx.x -    4]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >    8) { if(threadIdx.x < n && threadIdx.x >=    8) { T tmp = array[threadIdx.x -    8]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   16) { if(threadIdx.x < n && threadIdx.x >=   16) { T tmp = array[threadIdx.x -   16]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   32) { if(threadIdx.x < n && threadIdx.x >=   32) { T tmp = array[threadIdx.x -   32]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >   64) { if(threadIdx.x < n && threadIdx.x >=   64) { T tmp = array[threadIdx.x -   64]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  128) { if(threadIdx.x < n && threadIdx.x >=  128) { T tmp = array[threadIdx.x -  128]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  256) { if(threadIdx.x < n && threadIdx.x >=  256) { T tmp = array[threadIdx.x -  256]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE >  512) { if(threadIdx.x < n && threadIdx.x >=  512) { T tmp = array[threadIdx.x -  512]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
    if (CTA_SIZE > 1024) { if(threadIdx.x < n && threadIdx.x >= 1024) { T tmp = array[threadIdx.x - 1024]; val = binary_op(tmp, val); } __syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool FullBlock,
          typename InputIterator,
          typename ValueType>
__device__ __forceinline__
void load_block(const unsigned int n,
                InputIterator input,
                ValueType (&sdata)[K][CTA_SIZE + 1])
{
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = k*CTA_SIZE + threadIdx.x;

    if (FullBlock || offset < n)
    {
      InputIterator temp = input + offset;
      sdata[offset % K][offset / K] = thrust::detail::backend::dereference(temp);
    }
  }

  __syncthreads();
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool Inclusive,
          bool FullBlock,
          typename OutputIterator,
          typename ValueType>
__device__ __forceinline__
void store_block(const unsigned int n,
                 OutputIterator output,
                 ValueType (&sdata)[K][CTA_SIZE + 1],
                 ValueType& carry)
{
  if (Inclusive)
  {
    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int offset = k*CTA_SIZE + threadIdx.x;

      if (FullBlock || offset < n)
      {
        OutputIterator temp = output + offset;
        thrust::detail::backend::dereference(temp) = sdata[offset % K][offset / K];
      }
    }   
  }
  else
  {
    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int offset = k*CTA_SIZE + threadIdx.x;

      if (FullBlock || offset < n)
      {
        OutputIterator temp = output + offset;
        thrust::detail::backend::dereference(temp) = (offset == 0) ? carry : sdata[(offset - 1) % K][(offset - 1) / K];
      }
    }   
  }
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool FullBlock,
          typename InputIterator,
          typename BinaryFunction,
          typename ValueType>
__device__ __forceinline__
void upsweep_body(const unsigned int n,
                  const bool carry_in,
                  InputIterator input,
                  BinaryFunction binary_op,
                  ValueType (&sdata)[K][CTA_SIZE + 1],
                  ValueType& carry)
{
  // read data
  load_block<CTA_SIZE,K,FullBlock>(n, input, sdata);
 
  // copy into local array
  ValueType ldata[K];
  for (unsigned int k = 0; k < K; k++)
    ldata[k] = sdata[k][threadIdx.x];

  // carry in
  if (threadIdx.x == 0 && carry_in)
  {
    // XXX WAR sm_10 issue
    ValueType tmp = carry;
    ldata[0] = binary_op(tmp, ldata[0]);
  }

  // scan local values
  for(unsigned int k = 1; k < K; k++)
  {
    const unsigned int offset = K * threadIdx.x + k;

    if (FullBlock || offset < n)
      ldata[k] = binary_op(ldata[k-1],ldata[k]);
  }

  sdata[K - 1][threadIdx.x] = ldata[K - 1];

  __syncthreads();

  // second level scan
  if (FullBlock && sizeof(ValueType) > 1) // TODO investigate why this WAR is necessary
    scan_block<CTA_SIZE>(sdata[K - 1], binary_op); 
  else
    scan_block_n<CTA_SIZE>(sdata[K - 1], n / K, binary_op);

  // store carry out
  if (FullBlock)
  {
     if (threadIdx.x == CTA_SIZE - 1)
        carry = sdata[K - 1][threadIdx.x];
  }
  else
  {
    if (threadIdx.x == (n - 1) / K)
    {
      ValueType sum;

      for (unsigned int k = 0; k < K; k++)
          if ((n - 1) % K == k)
              sum = ldata[k];

      if (threadIdx.x > 0)
      {
        // WAR sm_10 issue
        ValueType tmp = sdata[K - 1][threadIdx.x - 1];
        sum = binary_op(tmp, sum);
      }

      carry = sum;
    }
  }

  __syncthreads();
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool Inclusive,
          bool FullBlock,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename ValueType>
__device__ __forceinline__
void scan_body(const unsigned int n,
               const bool carry_in,
               InputIterator input,
               OutputIterator output,
               BinaryFunction binary_op,
               ValueType (&sdata)[K][CTA_SIZE + 1],
               ValueType& carry)
{
  // read data
  load_block<CTA_SIZE,K,FullBlock>(n, input, sdata);

  // copy into local array
  ValueType ldata[K];
  for (unsigned int k = 0; k < K; k++)
    ldata[k] = sdata[k][threadIdx.x];

  // carry in
  if (threadIdx.x == 0 && carry_in)
  {
    // XXX WAR sm_10 issue
    ValueType tmp = carry;
    ldata[0] = binary_op(tmp, ldata[0]);
  }

  // scan local values
  for(unsigned int k = 1; k < K; k++)
  {
    const unsigned int offset = K * threadIdx.x + k;

    if (FullBlock || offset < n)
      ldata[k] = binary_op(ldata[k-1],ldata[k]);
  }

  sdata[K - 1][threadIdx.x] = ldata[K - 1];

  __syncthreads();

  // second level scan
  if (FullBlock)
    scan_block<CTA_SIZE>(sdata[K - 1], binary_op);
  else
    scan_block_n<CTA_SIZE>(sdata[K - 1], n / K, binary_op);
  
  // update local values
  if (threadIdx.x > 0)
  {
    ValueType left = sdata[K - 1][threadIdx.x - 1];

    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int offset = K * threadIdx.x + k;

      if (FullBlock || offset < n)
        ldata[k] = binary_op(left, ldata[k]);
    }
  }

  for (unsigned int k = 0; k < K; k++)
    sdata[k][threadIdx.x] = ldata[k];

  __syncthreads();

  // write data
  store_block<CTA_SIZE, K, Inclusive, FullBlock>(n, output, sdata, carry);
  
  // store carry out
  if (threadIdx.x == 0)
  {
    if (FullBlock)
      carry = sdata[K - 1][CTA_SIZE - 1];
    else
      carry = sdata[(n - 1) % K][(n - 1) / K]; // note: this must come after the local update
  }

  __syncthreads();
}


template <unsigned int CTA_SIZE,
          typename InputIterator,
          typename ValueType,
          typename BinaryFunction,
          typename Decomposition>
__launch_bounds__(CTA_SIZE,1)          
__global__
void upsweep_intervals(InputIterator input,
                       ValueType * block_results,
                       BinaryFunction binary_op,
                       Decomposition decomp)
{
    typedef typename Decomposition::index_type  IndexType;

#if __CUDA_ARCH__ >= 200
    const unsigned int SMEM = (48 * 1024) - 256;
#else
    const unsigned int SMEM = (16 * 1024) - 256;
#endif
    const unsigned int MAX_K = ((SMEM - 1 * sizeof(ValueType))/ (sizeof(ValueType) * (CTA_SIZE + 1)));
    const unsigned int K     = (MAX_K < 6) ? MAX_K : 6;

    __shared__ ValueType sdata[K][CTA_SIZE + 1];  // padded to avoid bank conflicts
    
    __shared__ ValueType carry; // storage for carry out
    
    __syncthreads(); // XXX needed because CUDA fires default constructors now
    
    thrust::detail::backend::index_range<IndexType> interval = decomp[blockIdx.x];

    IndexType base = interval.begin();

    input += base;

    const unsigned int unit_size = K * CTA_SIZE;

    bool carry_in = false;

    // process full units
    while (base + unit_size <= interval.end())
    {
        const unsigned int n = unit_size;
        upsweep_body<CTA_SIZE,K,true>(n, carry_in, input, binary_op, sdata, carry);
        base   += unit_size;
        input  += unit_size;
        carry_in = true;
    }

    // process partially full unit at end of input (if necessary)
    if (base < interval.end())
    {
        const unsigned int n = interval.end() - base;
        upsweep_body<CTA_SIZE,K,false>(n, carry_in, input, binary_op, sdata, carry);
    }

    // write interval sum
    if (threadIdx.x == 0)
        block_results[blockIdx.x] = carry;
}


template <unsigned int CTA_SIZE,
          bool Inclusive,
          typename InputIterator,
          typename OutputIterator,
          typename ValueType,
          typename BinaryFunction,
          typename Decomposition>
__launch_bounds__(CTA_SIZE,1)          
__global__
void downsweep_intervals(InputIterator input,
                         OutputIterator output,
                         ValueType * block_results,
                         BinaryFunction binary_op,
                         Decomposition decomp)
{
    typedef typename Decomposition::index_type IndexType;

#if __CUDA_ARCH__ >= 200
    const unsigned int SMEM = (48 * 1024) - 256;
#else
    const unsigned int SMEM = (16 * 1024) - 256;
#endif
    const unsigned int MAX_K = ((SMEM - 1 * sizeof(ValueType))/ (sizeof(ValueType) * (CTA_SIZE + 1)));
    const unsigned int K     = (MAX_K < 6) ? MAX_K : 6;

    __shared__ ValueType sdata[K][CTA_SIZE + 1];  // padded to avoid bank conflicts
    
    __shared__ ValueType carry; // storage for carry in and carry out

    __syncthreads(); // XXX needed because CUDA fires default constructors now

    thrust::detail::backend::index_range<IndexType> interval = decomp[blockIdx.x];

    IndexType base = interval.begin();

    input  += base;
    output += base;

    const unsigned int unit_size = K * CTA_SIZE;

    bool carry_in  = (Inclusive && blockIdx.x == 0) ? false : true;

    if (carry_in)
    {
        if (threadIdx.x == 0)
            carry = block_results[blockIdx.x];
        __syncthreads();
    }

    // process full units
    while (base + unit_size <= interval.end())
    {
        const unsigned int n = unit_size;
        scan_body<CTA_SIZE,K,Inclusive,true>(n, carry_in, input, output, binary_op, sdata, carry);
        base   += K * CTA_SIZE;
        input  += K * CTA_SIZE;
        output += K * CTA_SIZE;
        carry_in = true;
    }

    // process partially full unit at end of input (if necessary)
    if (base < interval.end())
    {
        const unsigned int n = interval.end() - base;
        scan_body<CTA_SIZE,K,Inclusive,false>(n, carry_in, input, output, binary_op, sdata, carry);
    }
}


template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction>
OutputIterator inclusive_scan(InputIterator first,
                              InputIterator last,
                              OutputIterator output,
                              BinaryFunction binary_op)
{
  // the pseudocode for deducing the type of the temporary used below:
  // 
  // if BinaryFunction is AdaptableBinaryFunction
  //   TemporaryType = AdaptableBinaryFunction::result_type
  // else if OutputIterator is a "pure" output iterator
  //   TemporaryType = InputIterator::value_type
  // else
  //   TemporaryType = OutputIterator::value_type
  //
  // XXX upon c++0x, TemporaryType needs to be:
  // result_of<BinaryFunction>::type

  typedef typename eval_if<
    has_result_type<BinaryFunction>::value,
    result_type<BinaryFunction>,
    eval_if<
      is_output_iterator<OutputIterator>::value,
      thrust::iterator_value<InputIterator>,
      thrust::iterator_value<OutputIterator>
    >
  >::type ValueType;

  typedef          unsigned int                                              IndexType;
  typedef          thrust::detail::backend::uniform_decomposition<IndexType> Decomposition;

  if (first == last)
      return output;

  Decomposition decomp = thrust::detail::backend::cuda::default_decomposition<IndexType>(last - first);

  thrust::detail::uninitialized_array<ValueType,thrust::detail::cuda_device_space_tag> block_results(decomp.size());
  
  // TODO tune this
  const static unsigned int CTA_SIZE = 32 * 7;

  // compute sum over each interval
  if (thrust::detail::is_commutative<BinaryFunction>::value)
  {
    // use reduce_intervals for commutative operators
    thrust::detail::backend::cuda::reduce_intervals(first, block_results.begin(), binary_op, decomp);
  }
  else
  {
    upsweep_intervals<CTA_SIZE> <<<decomp.size(), CTA_SIZE>>>
        (first,
         thrust::raw_pointer_cast(&block_results[0]),
         binary_op,
         decomp);
    synchronize_if_enabled("upsweep_intervals");
  }

  // second level inclusive scan of per-block results
  downsweep_intervals<CTA_SIZE,true> <<<         1, CTA_SIZE>>>
      (thrust::raw_pointer_cast(&block_results[0]),
       thrust::raw_pointer_cast(&block_results[0]),
       thrust::raw_pointer_cast(&block_results[0]), // not used
       binary_op,
       Decomposition(decomp.size(), 1, 1));
  synchronize_if_enabled("downsweep_intervals");
  
  // update intervals with result of second level scan
  downsweep_intervals<CTA_SIZE,true> <<<decomp.size(), CTA_SIZE>>>
      (first,
       output,
       thrust::raw_pointer_cast(&block_results[0]) - 1, // shift block results
       binary_op,
       decomp);
  synchronize_if_enabled("downsweep_intervals");
  
  return output + (last - first);
}


template <typename InputIterator,
          typename OutputIterator,
          typename T,
          typename BinaryFunction>
OutputIterator exclusive_scan(InputIterator first,
                              InputIterator last,
                              OutputIterator output,
                              const T init,
                              BinaryFunction binary_op)
{
  // the pseudocode for deducing the type of the temporary used below:
  // 
  // if BinaryFunction is AdaptableBinaryFunction
  //   TemporaryType = AdaptableBinaryFunction::result_type
  // else if OutputIterator is a "pure" output iterator
  //   TemporaryType = InputIterator::value_type
  // else
  //   TemporaryType = OutputIterator::value_type
  //
  // XXX upon c++0x, TemporaryType needs to be:
  // result_of<BinaryFunction>::type

  typedef typename eval_if<
    has_result_type<BinaryFunction>::value,
    result_type<BinaryFunction>,
    eval_if<
      is_output_iterator<OutputIterator>::value,
      thrust::iterator_value<InputIterator>,
      thrust::iterator_value<OutputIterator>
    >
  >::type ValueType;

  typedef          unsigned int                                              IndexType;
  typedef          thrust::detail::backend::uniform_decomposition<IndexType> Decomposition;

  if (first == last)
      return output;

  Decomposition decomp = thrust::detail::backend::cuda::default_decomposition<IndexType>(last - first);

  thrust::detail::uninitialized_array<ValueType,thrust::detail::cuda_device_space_tag> block_results(decomp.size() + 1);
  
  // TODO tune this
  const static unsigned int CTA_SIZE = 32 * 5;

  // compute sum over each interval
  if (thrust::detail::is_commutative<BinaryFunction>::value)
  {
    // use reduce_intervals for commutative operators
    thrust::detail::backend::cuda::reduce_intervals(first, block_results.begin() + 1, binary_op, decomp);
  }
  else
  {
    upsweep_intervals<CTA_SIZE> <<<decomp.size(), CTA_SIZE>>>
        (first,
         thrust::raw_pointer_cast(&block_results[0]) + 1,
         binary_op,
         decomp);
    synchronize_if_enabled("upsweep_intervals");
  }

  // place init before per-block results
  block_results[0] = init;
  
  // second level inclusive scan of per-block results
  downsweep_intervals<CTA_SIZE,true> <<<         1, CTA_SIZE>>>
      (thrust::raw_pointer_cast(&block_results[0]),
       thrust::raw_pointer_cast(&block_results[0]),
       thrust::raw_pointer_cast(&block_results[0]), // not used
       binary_op,
       Decomposition(decomp.size() + 1, 1, 1));
  synchronize_if_enabled("downsweep_intervals");
  
  // update intervals with result of second level scan
  downsweep_intervals<CTA_SIZE,false> <<<decomp.size(), CTA_SIZE>>>
      (first,
       output,
       thrust::raw_pointer_cast(&block_results[0]), 
       binary_op,
       decomp);
  synchronize_if_enabled("downsweep_intervals");
  
  return output + (last - first);
}

} // end namespace fast_scan
} // end namespace detail
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

