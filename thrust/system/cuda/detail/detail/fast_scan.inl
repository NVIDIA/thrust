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


#include <thrust/detail/config.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/detail/temporary_array.h>

#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>

#include <thrust/system/cuda/detail/reduce_intervals.h>
#include <thrust/system/cuda/detail/synchronize.h>
#include <thrust/system/cuda/detail/default_decomposition.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <thrust/detail/raw_pointer_cast.h>

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN


namespace thrust
{
namespace detail
{

// forward declaration of temporary_array
template<typename,typename> class temporary_array;

} // end detail

namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{
namespace fast_scan
{
namespace fast_scan_detail
{


// TODO tune this
template <typename ValueType>
struct inclusive_scan_block_size
{
  private:
  static const unsigned int max_memory         = 16384 - 256 - 2 * sizeof(ValueType);
  static const unsigned int max_block_size     = max_memory / sizeof(ValueType);
  static const unsigned int default_block_size = 7 * 32;
  static const unsigned int block_size         = (max_block_size < default_block_size) ? max_block_size : default_block_size;

  public:
  static const unsigned int pass1 = block_size;
  static const unsigned int pass2 = block_size;
  static const unsigned int pass3 = block_size;
};

// TODO tune this
template <typename ValueType>
struct exclusive_scan_block_size
{
  private:
  static const unsigned int max_memory         = 16384 - 256 - 2 * sizeof(ValueType);
  static const unsigned int max_block_size     = max_memory / sizeof(ValueType);
  static const unsigned int default_block_size = 5 * 32;
  static const unsigned int block_size         = (max_block_size < default_block_size) ? max_block_size : default_block_size;

  public:
  static const unsigned int pass1 = block_size;
  static const unsigned int pass2 = block_size;
  static const unsigned int pass3 = block_size;
};


template <unsigned int CTA_SIZE,
          typename Context,
          typename SharedArray,
          typename BinaryFunction>
__device__ __thrust_forceinline__
void scan_block(Context context, SharedArray array, BinaryFunction binary_op)
{
    typedef typename thrust::iterator_value<SharedArray>::type T;

    T val = array[context.thread_index()];

    if (CTA_SIZE >    1) { if(context.thread_index() >=    1) { T tmp = array[context.thread_index() -    1]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >    2) { if(context.thread_index() >=    2) { T tmp = array[context.thread_index() -    2]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >    4) { if(context.thread_index() >=    4) { T tmp = array[context.thread_index() -    4]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >    8) { if(context.thread_index() >=    8) { T tmp = array[context.thread_index() -    8]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >   16) { if(context.thread_index() >=   16) { T tmp = array[context.thread_index() -   16]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >   32) { if(context.thread_index() >=   32) { T tmp = array[context.thread_index() -   32]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >   64) { if(context.thread_index() >=   64) { T tmp = array[context.thread_index() -   64]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >  128) { if(context.thread_index() >=  128) { T tmp = array[context.thread_index() -  128]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >  256) { if(context.thread_index() >=  256) { T tmp = array[context.thread_index() -  256]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }  
    if (CTA_SIZE >  512) { if(context.thread_index() >=  512) { T tmp = array[context.thread_index() -  512]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }  
    if (CTA_SIZE > 1024) { if(context.thread_index() >= 1024) { T tmp = array[context.thread_index() - 1024]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }  
}

template <unsigned int CTA_SIZE,
          typename Context,
          typename SharedArray,
          typename BinaryFunction>
__device__ __thrust_forceinline__
void scan_block_n(Context context, SharedArray array, const unsigned int n, BinaryFunction binary_op)
{
    typedef typename thrust::iterator_value<SharedArray>::type T;

    T val = array[context.thread_index()];

    if (CTA_SIZE >    1) { if(context.thread_index() < n && context.thread_index() >=    1) { T tmp = array[context.thread_index() -    1]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >    2) { if(context.thread_index() < n && context.thread_index() >=    2) { T tmp = array[context.thread_index() -    2]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >    4) { if(context.thread_index() < n && context.thread_index() >=    4) { T tmp = array[context.thread_index() -    4]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >    8) { if(context.thread_index() < n && context.thread_index() >=    8) { T tmp = array[context.thread_index() -    8]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >   16) { if(context.thread_index() < n && context.thread_index() >=   16) { T tmp = array[context.thread_index() -   16]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >   32) { if(context.thread_index() < n && context.thread_index() >=   32) { T tmp = array[context.thread_index() -   32]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >   64) { if(context.thread_index() < n && context.thread_index() >=   64) { T tmp = array[context.thread_index() -   64]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >  128) { if(context.thread_index() < n && context.thread_index() >=  128) { T tmp = array[context.thread_index() -  128]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >  256) { if(context.thread_index() < n && context.thread_index() >=  256) { T tmp = array[context.thread_index() -  256]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE >  512) { if(context.thread_index() < n && context.thread_index() >=  512) { T tmp = array[context.thread_index() -  512]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
    if (CTA_SIZE > 1024) { if(context.thread_index() < n && context.thread_index() >= 1024) { T tmp = array[context.thread_index() - 1024]; val = binary_op(tmp, val); } context.barrier(); array[context.thread_index()] = val; context.barrier(); }
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool FullBlock,
          typename Context,
          typename InputIterator,
          typename ValueType>
__device__ __thrust_forceinline__
void load_block(Context context,
                const unsigned int n,
                InputIterator input,
                ValueType (&sdata)[K][CTA_SIZE + 1])
{
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = k*CTA_SIZE + context.thread_index();

    if (FullBlock || offset < n)
    {
      InputIterator temp = input + offset;
      sdata[offset % K][offset / K] = *temp;
    }
  }

  context.barrier();
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool Inclusive,
          bool FullBlock,
          typename Context,
          typename OutputIterator,
          typename ValueType>
__device__ __thrust_forceinline__
void store_block(Context context,
                 const unsigned int n,
                 OutputIterator output,
                 ValueType (&sdata)[K][CTA_SIZE + 1],
                 ValueType& carry)
{
  if (Inclusive)
  {
    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int offset = k*CTA_SIZE + context.thread_index();

      if (FullBlock || offset < n)
      {
        OutputIterator temp = output + offset;
        *temp = sdata[offset % K][offset / K];
      }
    }   
  }
  else
  {
    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int offset = k*CTA_SIZE + context.thread_index();

      if (FullBlock || offset < n)
      {
        OutputIterator temp = output + offset;
        *temp = (offset == 0) ? carry : sdata[(offset - 1) % K][(offset - 1) / K];
      }
    }   
  }
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool FullBlock,
          typename Context,
          typename InputIterator,
          typename BinaryFunction,
          typename ValueType>
__device__ __thrust_forceinline__
void upsweep_body(Context context,
                  const unsigned int n,
                  const bool carry_in,
                  InputIterator input,
                  BinaryFunction binary_op,
                  ValueType (&sdata)[K][CTA_SIZE + 1],
                  ValueType& carry)
{
  // read data
  load_block<CTA_SIZE,K,FullBlock>(context, n, input, sdata);
 
  // copy into local array
  ValueType ldata[K];
  for (unsigned int k = 0; k < K; k++)
    ldata[k] = sdata[k][context.thread_index()];

  // carry in
  if (context.thread_index() == 0 && carry_in)
  {
    // XXX WAR sm_10 issue
    ValueType tmp = carry;
    ldata[0] = binary_op(tmp, ldata[0]);
  }

  // scan local values
  for(unsigned int k = 1; k < K; k++)
  {
    const unsigned int offset = K * context.thread_index() + k;

    if (FullBlock || offset < n)
      ldata[k] = binary_op(ldata[k-1],ldata[k]);
  }

  sdata[K - 1][context.thread_index()] = ldata[K - 1];

  context.barrier();

  // second level scan
  if (FullBlock && sizeof(ValueType) > 1) // TODO investigate why this WAR is necessary
    scan_block<CTA_SIZE>(context, sdata[K - 1], binary_op); 
  else
    scan_block_n<CTA_SIZE>(context, sdata[K - 1], n / K, binary_op);

  // store carry out
  if (FullBlock)
  {
     if (context.thread_index() == CTA_SIZE - 1)
        carry = sdata[K - 1][context.thread_index()];
  }
  else
  {
    if (context.thread_index() == (n - 1) / K)
    {
      ValueType sum;

      for (unsigned int k = 0; k < K; k++)
          if ((n - 1) % K == k)
              sum = ldata[k];

      if (context.thread_index() > 0)
      {
        // WAR sm_10 issue
        ValueType tmp = sdata[K - 1][context.thread_index() - 1];
        sum = binary_op(tmp, sum);
      }

      carry = sum;
    }
  }

  context.barrier();
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool Inclusive,
          bool FullBlock,
          typename Context,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename ValueType>
__device__ __thrust_forceinline__
void scan_body(Context context,
               const unsigned int n,
               const bool carry_in,
               InputIterator input,
               OutputIterator output,
               BinaryFunction binary_op,
               ValueType (&sdata)[K][CTA_SIZE + 1],
               ValueType& carry)
{
  // read data
  load_block<CTA_SIZE,K,FullBlock>(context, n, input, sdata);

  // copy into local array
  ValueType ldata[K];
  for (unsigned int k = 0; k < K; k++)
    ldata[k] = sdata[k][context.thread_index()];

  // carry in
  if (context.thread_index() == 0 && carry_in)
  {
    // XXX WAR sm_10 issue
    ValueType tmp = carry;
    ldata[0] = binary_op(tmp, ldata[0]);
  }

  // scan local values
  for(unsigned int k = 1; k < K; k++)
  {
    const unsigned int offset = K * context.thread_index() + k;

    if (FullBlock || offset < n)
      ldata[k] = binary_op(ldata[k-1],ldata[k]);
  }

  sdata[K - 1][context.thread_index()] = ldata[K - 1];

  context.barrier();

  // second level scan
  if (FullBlock)
    scan_block<CTA_SIZE>(context, sdata[K - 1], binary_op);
  else
    scan_block_n<CTA_SIZE>(context, sdata[K - 1], n / K, binary_op);
  
  // update local values
  if (context.thread_index() > 0)
  {
    ValueType left = sdata[K - 1][context.thread_index() - 1];

    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int offset = K * context.thread_index() + k;

      if (FullBlock || offset < n)
        ldata[k] = binary_op(left, ldata[k]);
    }
  }

  for (unsigned int k = 0; k < K; k++)
    sdata[k][context.thread_index()] = ldata[k];

  context.barrier();

  // write data
  store_block<CTA_SIZE, K, Inclusive, FullBlock>(context, n, output, sdata, carry);
  
  // store carry out
  if (context.thread_index() == 0)
  {
    if (FullBlock)
      carry = sdata[K - 1][CTA_SIZE - 1];
    else
      carry = sdata[(n - 1) % K][(n - 1) / K]; // note: this must come after the local update
  }

  context.barrier();
}

template <typename InputIterator,
          typename ValueType,
          typename BinaryFunction,
          typename Decomposition,
          typename Context>
struct upsweep_intervals_closure
{
  InputIterator  input;
  ValueType *    block_results; // TODO change this to ValueIterator
  BinaryFunction binary_op;
  Decomposition  decomp;
  Context        context;
  
  typedef Context context_type;

  upsweep_intervals_closure(InputIterator input,
                            ValueType * block_results,
                            BinaryFunction binary_op,
                            Decomposition decomp,
                            Context context = Context())
    : input(input), block_results(block_results), binary_op(binary_op), decomp(decomp), context(context) {}

  __device__ __thrust_forceinline__
  void operator()(void)
  {
    typedef typename Decomposition::index_type  IndexType;

    const unsigned int CTA_SIZE = context_type::ThreadsPerBlock::value;

#if __CUDA_ARCH__ >= 200
    const unsigned int SMEM = (48 * 1024);
#else
    const unsigned int SMEM = (16 * 1024) - 256;
#endif
    const unsigned int MAX_K = ((SMEM - 1 * sizeof(ValueType)) / (sizeof(ValueType) * (CTA_SIZE + 1)));
    const unsigned int K     = (MAX_K < 6) ? MAX_K : 6;

    __shared__ uninitialized<ValueType[K][CTA_SIZE + 1]> sdata; // padded to avoid bank conflicts
    
    __shared__ uninitialized<ValueType> carry; // storage for carry out
    if(context.thread_index() == 0) carry.construct();
    
    context.barrier();
    
    thrust::system::detail::internal::index_range<IndexType> interval = decomp[context.block_index()];

    IndexType base = interval.begin();

    input += base;

    const unsigned int unit_size = K * CTA_SIZE;

    bool carry_in = false;

    // process full units
    while (base + unit_size <= interval.end())
    {
      const unsigned int n = unit_size;
      upsweep_body<CTA_SIZE,K,true>(context, n, carry_in, input, binary_op, sdata.get(), carry.get());
      base   += unit_size;
      input  += unit_size;
      carry_in = true;
    }

    // process partially full unit at end of input (if necessary)
    if (base < interval.end())
    {
      const unsigned int n = interval.end() - base;
      upsweep_body<CTA_SIZE,K,false>(context, n, carry_in, input, binary_op, sdata.get(), carry.get());
    }

    // write interval sum
    if (context.thread_index() == 0)
      block_results[context.block_index()] = carry;
  }
};


template <bool Inclusive,
          typename InputIterator,
          typename OutputIterator,
          typename ValueType,
          typename BinaryFunction,
          typename Decomposition,
          typename Context>
struct downsweep_intervals_closure
{
  InputIterator  input;
  OutputIterator output;
  ValueType *    block_results;
  BinaryFunction binary_op;
  Decomposition  decomp;
  Context        context;

  typedef Context context_type;

  downsweep_intervals_closure(InputIterator input,
                              OutputIterator output,
                              ValueType * block_results,
                              BinaryFunction binary_op,
                              Decomposition decomp,
                              Context context = Context())
    : input(input), output(output), block_results(block_results), binary_op(binary_op), decomp(decomp), context(context) {}

  __device__ __thrust_forceinline__
  void operator()(void)
  {
    typedef typename Decomposition::index_type IndexType;
    
    const unsigned int CTA_SIZE = context_type::ThreadsPerBlock::value;

#if __CUDA_ARCH__ >= 200
    const unsigned int SMEM = (48 * 1024);
#else
    const unsigned int SMEM = (16 * 1024) - 256;
#endif
    const unsigned int MAX_K = ((SMEM - 1 * sizeof(ValueType))/ (sizeof(ValueType) * (CTA_SIZE + 1)));
    const unsigned int K     = (MAX_K < 6) ? MAX_K : 6;

    __shared__ uninitialized<ValueType[K][CTA_SIZE + 1]> sdata;  // padded to avoid bank conflicts
    
    __shared__ uninitialized<ValueType> carry; // storage for carry in and carry out
    if(context.thread_index() == 0) carry.construct();

    context.barrier();

    thrust::system::detail::internal::index_range<IndexType> interval = decomp[context.block_index()];

    IndexType base = interval.begin();

    input  += base;
    output += base;

    const unsigned int unit_size = K * CTA_SIZE;

    bool carry_in  = (Inclusive && context.block_index() == 0) ? false : true;

    if (carry_in)
    {
      if (context.thread_index() == 0)
        carry = block_results[context.block_index()];
      context.barrier();
    }

    // process full units
    while (base + unit_size <= interval.end())
    {
      const unsigned int n = unit_size;
      scan_body<CTA_SIZE,K,Inclusive,true>(context, n, carry_in, input, output, binary_op, sdata.get(), carry.get());
      base   += K * CTA_SIZE;
      input  += K * CTA_SIZE;
      output += K * CTA_SIZE;
      carry_in = true;
    }

    // process partially full unit at end of input (if necessary)
    if (base < interval.end())
    {
      const unsigned int n = interval.end() - base;
      scan_body<CTA_SIZE,K,Inclusive,false>(context, n, carry_in, input, output, binary_op, sdata.get(), carry.get());
    }
  }
};


} // end namespace fast_scan_detail


template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction>
OutputIterator inclusive_scan(execution_policy<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator output,
                              BinaryFunction binary_op)
{
  using namespace fast_scan_detail;

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

  typedef typename thrust::detail::eval_if<
    thrust::detail::has_result_type<BinaryFunction>::value,
    thrust::detail::result_type<BinaryFunction>,
    thrust::detail::eval_if<
      thrust::detail::is_output_iterator<OutputIterator>::value,
      thrust::iterator_value<InputIterator>,
      thrust::iterator_value<OutputIterator>
    >
  >::type ValueType;

  typedef unsigned int                                                       IndexType;
  typedef thrust::system::detail::internal::uniform_decomposition<IndexType> Decomposition;
  typedef thrust::detail::temporary_array<ValueType,DerivedPolicy>           ValueArray;

  if (first == last)
      return output;

  Decomposition decomp = thrust::system::cuda::detail::default_decomposition<IndexType>(last - first);

  ValueArray block_results(exec, decomp.size());
  
  // compute sum over each interval
  if (thrust::detail::is_commutative<BinaryFunction>::value)
  {
    // use reduce_intervals for commutative operators
    thrust::system::cuda::detail::reduce_intervals(exec, first, block_results.begin(), binary_op, decomp);
  }
  else
  {
    const static unsigned int ThreadsPerBlock = inclusive_scan_block_size<ValueType>::pass1;
    typedef detail::statically_blocked_thread_array<ThreadsPerBlock> Context;

    typedef upsweep_intervals_closure<InputIterator,ValueType,BinaryFunction,Decomposition,Context> Closure;
    Closure closure(first,
                    thrust::raw_pointer_cast(&block_results[0]),
                    binary_op,
                    decomp);
    detail::launch_closure(closure, decomp.size(), ThreadsPerBlock);
  }

  // second level inclusive scan of per-block results
  {
    const static unsigned int ThreadsPerBlock = inclusive_scan_block_size<ValueType>::pass2;
    typedef detail::statically_blocked_thread_array<ThreadsPerBlock> Context;

    typedef downsweep_intervals_closure<true,ValueType*,ValueType*,ValueType,BinaryFunction,Decomposition,Context> Closure;
    Closure closure(thrust::raw_pointer_cast(&block_results[0]),
                    thrust::raw_pointer_cast(&block_results[0]),
                    thrust::raw_pointer_cast(&block_results[0]), // not used
                    binary_op,
                    Decomposition(decomp.size(), 1, 1));
    detail::launch_closure(closure, 1, ThreadsPerBlock);
  }
  
  // update intervals with result of second level scan
  {
    const static unsigned int ThreadsPerBlock = inclusive_scan_block_size<ValueType>::pass3;
    typedef detail::statically_blocked_thread_array<ThreadsPerBlock> Context;

    typedef downsweep_intervals_closure<true,InputIterator,OutputIterator,ValueType,BinaryFunction,Decomposition,Context> Closure;
    Closure closure(first,
                    output,
                    thrust::raw_pointer_cast(&block_results[0]) - 1, // shift block results
                    binary_op,
                    decomp);
    detail::launch_closure(closure, decomp.size(), ThreadsPerBlock);
  }
  
  return output + (last - first);
}


template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename T,
          typename BinaryFunction>
OutputIterator exclusive_scan(execution_policy<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator output,
                              const T init,
                              BinaryFunction binary_op)
{
  using namespace fast_scan_detail;

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

  typedef typename thrust::detail::eval_if<
    thrust::detail::has_result_type<BinaryFunction>::value,
    thrust::detail::result_type<BinaryFunction>,
    thrust::detail::eval_if<
      thrust::detail::is_output_iterator<OutputIterator>::value,
      thrust::iterator_value<InputIterator>,
      thrust::iterator_value<OutputIterator>
    >
  >::type ValueType;

  typedef unsigned int                                                       IndexType;
  typedef thrust::system::detail::internal::uniform_decomposition<IndexType> Decomposition;
  typedef thrust::detail::temporary_array<ValueType,DerivedPolicy>           ValueArray;

  if (first == last)
      return output;

  Decomposition decomp = thrust::system::cuda::detail::default_decomposition<IndexType>(last - first);

  ValueArray block_results(exec, decomp.size() + 1);
  
  // compute sum over each interval
  if (thrust::detail::is_commutative<BinaryFunction>::value)
  {
    // use reduce_intervals for commutative operators
    thrust::system::cuda::detail::reduce_intervals(exec, first, block_results.begin() + 1, binary_op, decomp);
  }
  else
  {
    const static unsigned int ThreadsPerBlock = exclusive_scan_block_size<ValueType>::pass1;
    typedef detail::statically_blocked_thread_array<ThreadsPerBlock> Context;

    typedef upsweep_intervals_closure<InputIterator,ValueType,BinaryFunction,Decomposition,Context> Closure;
    Closure closure(first,
                    thrust::raw_pointer_cast(&block_results[0]) + 1,
                    binary_op,
                    decomp);
    detail::launch_closure(closure, decomp.size(), ThreadsPerBlock);
  }

  // place init before per-block results
  block_results[0] = init;
  
  // second level inclusive scan of per-block results
  {
    const static unsigned int ThreadsPerBlock = exclusive_scan_block_size<ValueType>::pass2;
    typedef detail::statically_blocked_thread_array<ThreadsPerBlock> Context;

    typedef downsweep_intervals_closure<true,ValueType*,ValueType*,ValueType,BinaryFunction,Decomposition,Context> Closure;
    Closure closure(thrust::raw_pointer_cast(&block_results[0]),
                    thrust::raw_pointer_cast(&block_results[0]),
                    thrust::raw_pointer_cast(&block_results[0]), // not used
                    binary_op,
                    Decomposition(decomp.size() + 1, 1, 1));
    detail::launch_closure(closure, 1, ThreadsPerBlock);
  }
  
  // update intervals with result of second level scan
  {
    const static unsigned int ThreadsPerBlock = exclusive_scan_block_size<ValueType>::pass3;
    typedef detail::statically_blocked_thread_array<ThreadsPerBlock> Context;

    typedef downsweep_intervals_closure<false,InputIterator,OutputIterator,ValueType,BinaryFunction,Decomposition,Context> Closure;
    Closure closure(first,
                    output,
                    thrust::raw_pointer_cast(&block_results[0]), // shift block results
                    binary_op,
                    decomp);
    detail::launch_closure(closure, decomp.size(), ThreadsPerBlock);
  }
  
  return output + (last - first);
}


} // end namespace fast_scan
} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

