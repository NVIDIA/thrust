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


// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/uninitialized_array.h>
#include <thrust/detail/backend/dereference.h>
#include <thrust/detail/backend/cuda/synchronize.h>

// to configure launch parameters
#include <thrust/detail/backend/cuda/arch.h>
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
          __device__
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
          __device__
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
          typename OutputIterator,
          typename BinaryFunction,
          typename OutputType>
__device__ __forceinline__
void scan_body(const unsigned int n,
               const bool carry_in,
               InputIterator input,
               OutputIterator output,
               BinaryFunction binary_op,
               OutputType (&sdata)[K + 1][CTA_SIZE + 1])
{
  // read data
  for(unsigned int k = 0; k < K; k++)
  {
      const unsigned int offset = k*CTA_SIZE + threadIdx.x;

      if (FullBlock || offset < n)
      {
          InputIterator temp = input + offset;
          sdata[offset % K][offset / K] = thrust::detail::backend::dereference(temp);
      }
  }
  
  // carry in
  if (threadIdx.x == 0 && carry_in)
  {
      //sdata[0][0] = binary_op(sdata[K][CTA_SIZE - 1], sdata[0][0]);
      //// XXX WAR sm_10 issue
      OutputType tmp1 = sdata[K][CTA_SIZE - 1];
      OutputType tmp2 = sdata[0][0];
      sdata[0][0] = binary_op(tmp1, tmp2);
  }

  __syncthreads();

  // scan local values
  OutputType sum = sdata[0][threadIdx.x];

  for(unsigned int k = 1; k < K; k++)
  {
      const unsigned int offset = K * threadIdx.x + k;

      if (FullBlock || offset < n)
      {
          OutputType tmp = sdata[k][threadIdx.x];
          sum = binary_op(sum, tmp);
          sdata[k][threadIdx.x] = sum;
      }
  }

  // second level scan
  sdata[K][threadIdx.x] = sum;  __syncthreads();

  if (FullBlock)
    scan_block<CTA_SIZE>(sdata[K], binary_op);
  else
    scan_block_n<CTA_SIZE>(sdata[K], n / K, binary_op);
  
  // update local values
  if (threadIdx.x > 0)
  {
      sum = sdata[K][threadIdx.x - 1];

      for(unsigned int k = 0; k < K; k++)
      {
          const unsigned int offset = K * threadIdx.x + k;

          if (FullBlock || offset < n)
          {
              OutputType tmp = sdata[k][threadIdx.x];
              sdata[k][threadIdx.x] = binary_op(sum, tmp);
          }
      }
  }

  __syncthreads();

  // write data
  for(unsigned int k = 0; k < K; k++)
  {
      const unsigned int offset = k*CTA_SIZE + threadIdx.x;

      if (FullBlock || offset < n)
      {
          OutputIterator temp = output + offset;
          thrust::detail::backend::dereference(temp) = sdata[offset % K][offset / K];
      }
  }   
  
  __syncthreads();
}

template <unsigned int CTA_SIZE,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
__launch_bounds__(CTA_SIZE,1)          
__global__
void scan_intervals(InputIterator input,
                    OutputIterator output,
                    typename thrust::iterator_value<OutputIterator>::type * block_results,
                    BinaryFunction binary_op,
                    Decomposition decomp)
{
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;
    typedef typename Decomposition::index_type                    IndexType;

#if __CUDA_ARCH__ >= 200
    const unsigned int SMEM = (48 * 1024) - 256;
#else
    const unsigned int SMEM = (16 * 1024) - 256;
#endif
    const unsigned int MAX_K = (SMEM / (sizeof(OutputType) * (CTA_SIZE + 1))) - 1;
    const unsigned int K     = (MAX_K < 6) ? MAX_K : 6;

    __shared__ OutputType sdata[K + 1][CTA_SIZE + 1];  // padded to avoid bank conflicts
    
    __syncthreads(); // XXX needed because CUDA fires default constructors now
    
    thrust::detail::backend::index_range<IndexType> interval = decomp[blockIdx.x];

    IndexType base = interval.begin();

    input  += base;
    output += base;

    const unsigned int unit_size = K * CTA_SIZE;

    // process full units
    while (base + unit_size <= interval.end())
    {
        scan_body<CTA_SIZE,K,true>(unit_size, base != interval.begin(), input, output, binary_op, sdata);
        base   += K * CTA_SIZE;
        input  += K * CTA_SIZE;
        output += K * CTA_SIZE;
    }

    // process partially full unit at end of input (if necessary)
    if (base < interval.end())
        scan_body<CTA_SIZE,K,false>(interval.end() - base, base != interval.begin(), input, output, binary_op, sdata);

    __syncthreads();
    
    // write interval sum
    if (threadIdx.x == 0)
    {
        unsigned int offset = (interval.size() - 1) % (CTA_SIZE * K);
        block_results[blockIdx.x] = sdata[offset % K][offset / K];
    }
}


template <unsigned int CTA_SIZE,
          typename OutputIterator,
          typename OutputType,
          typename BinaryFunction,
          typename Decomposition>
__launch_bounds__(CTA_SIZE,1)          
__global__
void inclusive_update(OutputIterator output,
                      OutputType *   block_results,
                      BinaryFunction binary_op,
                      Decomposition decomp)
{
    typedef typename Decomposition::index_type                    IndexType;

    thrust::detail::backend::index_range<IndexType> interval = decomp[blockIdx.x];

    const unsigned int interval_begin = interval.begin();
    const unsigned int interval_end   = interval.end();

    if (blockIdx.x == 0)
        return;

    // value to add to this segment 
    OutputType sum = block_results[blockIdx.x - 1];
    
    // advance result iterator
    output += interval_begin + threadIdx.x;
    
    for(unsigned int base = interval_begin; base < interval_end; base += CTA_SIZE, output += CTA_SIZE)
    {
        const unsigned int i = base + threadIdx.x;

        if(i < interval_end)
        {
            OutputType tmp = thrust::detail::backend::dereference(output);
            thrust::detail::backend::dereference(output) = binary_op(sum, tmp);
        }

        __syncthreads();
    }
}

template <unsigned int CTA_SIZE,
          typename OutputIterator,
          typename OutputType,
          typename T,
          typename BinaryFunction,
          typename Decomposition>
__launch_bounds__(CTA_SIZE,1)          
__global__
void exclusive_update(OutputIterator output,
                      OutputType * block_results,
                      T init,
                      BinaryFunction binary_op,
                      Decomposition decomp)
{
    typedef typename Decomposition::index_type                    IndexType;

    __shared__ OutputType sdata[CTA_SIZE];  __syncthreads(); // XXX needed because CUDA fires default constructors now
    
    thrust::detail::backend::index_range<IndexType> interval = decomp[blockIdx.x];

    const unsigned int interval_begin = interval.begin();
    const unsigned int interval_end   = interval.end();

    // value to add to this segment 
    OutputType carry = init;

    if(blockIdx.x != 0)
    {
        OutputType tmp = block_results[blockIdx.x - 1];
        carry = binary_op(carry, tmp);
    }

    OutputType val   = carry;

    // advance result iterator
    output += interval_begin + threadIdx.x;

    for(unsigned int base = interval_begin; base < interval_end; base += CTA_SIZE, output += CTA_SIZE)
    {
        const unsigned int i = base + threadIdx.x;

        if(i < interval_end)
        {
            OutputType tmp = thrust::detail::backend::dereference(output);
            sdata[threadIdx.x] = binary_op(carry, tmp);
        }

        __syncthreads();

        if (threadIdx.x != 0)
            val = sdata[threadIdx.x - 1];

        if (i < interval_end)
            thrust::detail::backend::dereference(output) = val;

        if(threadIdx.x == 0)
            val = sdata[CTA_SIZE - 1];
        
        __syncthreads();
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
  typedef typename thrust::iterator_value<OutputIterator>::type              OutputType;
  typedef          unsigned int                                              IndexType;
  typedef          thrust::detail::backend::uniform_decomposition<IndexType> Decomposition;

  if (first == last)
      return output;

  Decomposition decomp = thrust::detail::backend::cuda::default_decomposition<IndexType>(last - first);

  thrust::detail::uninitialized_array<OutputType,thrust::detail::cuda_device_space_tag> block_results(decomp.size() + 1);
  
  // TODO tune this
  const static unsigned int CTA_SIZE = 256;

  // first level scan of interval (one interval per block)
  scan_intervals<CTA_SIZE> <<<decomp.size(), CTA_SIZE>>>
      (first,
       output,
       thrust::raw_pointer_cast(&block_results[0]),
       binary_op,
       decomp);
  synchronize_if_enabled("scan_intervals");
  
  // second level inclusive scan of per-block results
  scan_intervals<CTA_SIZE> <<<         1, CTA_SIZE>>>
      (thrust::raw_pointer_cast(&block_results[0]),
       thrust::raw_pointer_cast(&block_results[0]),
       thrust::raw_pointer_cast(&block_results[0]) + decomp.size(),
       binary_op,
       Decomposition(decomp.size(), 1, 1));
  synchronize_if_enabled("scan_intervals");
  
  // update intervals with result of second level scan
  inclusive_update<256> <<<decomp.size(), 256>>>
      (output,
       thrust::raw_pointer_cast(&block_results[0]),
       binary_op,
       decomp);
  synchronize_if_enabled("inclusive_update");
  
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
  typedef typename thrust::iterator_value<OutputIterator>::type              OutputType;
  typedef          unsigned int                                              IndexType;
  typedef          thrust::detail::backend::uniform_decomposition<IndexType> Decomposition;
  
  if (first == last)
      return output;

  Decomposition decomp = thrust::detail::backend::cuda::default_decomposition<IndexType>(last - first);

  thrust::detail::uninitialized_array<OutputType,thrust::detail::cuda_device_space_tag> block_results(decomp.size() + 1);
  
  // TODO tune this
  const static unsigned int CTA_SIZE = 256;

  // first level scan of interval (one interval per block)
  scan_intervals<CTA_SIZE> <<<decomp.size(), CTA_SIZE>>>
      (first,
       output,
       thrust::raw_pointer_cast(&block_results[0]),
       binary_op,
       decomp);
  synchronize_if_enabled("scan_intervals");
  
  // second level inclusive scan of per-block results
  scan_intervals<CTA_SIZE> <<<         1, CTA_SIZE>>>
      (thrust::raw_pointer_cast(&block_results[0]),
       thrust::raw_pointer_cast(&block_results[0]),
       thrust::raw_pointer_cast(&block_results[0]) + decomp.size(),
       binary_op,
       Decomposition(decomp.size(), 1, 1));
  synchronize_if_enabled("scan_intervals");
  
  // update intervals with result of second level scan
  exclusive_update<256> <<<decomp.size(), 256>>>
      (output,
       thrust::raw_pointer_cast(&block_results[0]),
       init,
       binary_op,
       decomp);
  synchronize_if_enabled("exclusive_update");

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

