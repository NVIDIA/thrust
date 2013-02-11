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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/select_system.h>

#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>

#include <thrust/detail/minmax.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/temporary_array.h>

#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/detail/default_decomposition.h>
#include <thrust/system/cuda/detail/block/inclusive_scan.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/reduce_intervals.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace reduce_by_key_detail
{

template <typename FlagType, typename IndexType, typename KeyType, typename BinaryPredicate>
struct tail_flag_functor
{
  BinaryPredicate binary_pred; // NB: this must be the first member for performance reasons
  IndexType n;

  typedef FlagType result_type;
  
  tail_flag_functor(IndexType n, BinaryPredicate binary_pred)
    : n(n), binary_pred(binary_pred)
  {}
  
  // XXX why is this noticably faster?  (it may read past the end of input)
  //FlagType operator()(const thrust::tuple<IndexType,KeyType,KeyType>& t) const
  
  template <typename Tuple>
  __host__ __device__ __thrust_forceinline__
  FlagType operator()(const Tuple& t)
  {
    if (thrust::get<0>(t) == (n - 1) || !binary_pred(thrust::get<1>(t), thrust::get<2>(t)))
      return 1;
    else
      return 0;
  }
};


template <unsigned int CTA_SIZE,
          unsigned int K,
          bool FullBlock,
          typename Context,
          typename FlagIterator,
          typename FlagType>
__device__ __thrust_forceinline__
FlagType load_flags(Context context,
                    const unsigned int n,
                    FlagIterator iflags,
                    FlagType  (&sflag)[CTA_SIZE])
{
  FlagType flag_bits = 0;

  // load flags in unordered fashion
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = k*CTA_SIZE + context.thread_index();

    if (FullBlock || offset < n)
    {
      FlagIterator temp = iflags + offset;
      if (*temp)
        flag_bits |= FlagType(1) << k;
    }
  }

  sflag[context.thread_index()] = flag_bits;
  
  context.barrier();

  flag_bits = 0;

  // obtain flags for iflags[K * context.thread_index(), K * context.thread_index() + K)
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = K * context.thread_index() + k;

    if (FullBlock || offset < n)
    {
      flag_bits |= ((sflag[offset % CTA_SIZE] >> (offset / CTA_SIZE)) & FlagType(1)) << k;
    }
  }

  context.barrier();
  
  sflag[context.thread_index()] = flag_bits;
  
  context.barrier();

  return flag_bits;
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool FullBlock,
          typename Context,
          typename InputIterator2,
          typename ValueType>
__device__ __thrust_forceinline__
void load_values(Context context,
                 const unsigned int n,
                 InputIterator2 ivals,
                 ValueType (&sdata)[K][CTA_SIZE + 1])
{
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = k*CTA_SIZE + context.thread_index();

    if (FullBlock || offset < n)
    {
      InputIterator2 temp = ivals + offset;
      sdata[offset % K][offset / K] = *temp;
    }
  }

  context.barrier();
}


template <unsigned int CTA_SIZE,
          unsigned int K,
          bool FullBlock,
          typename Context,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction,
          typename FlagIterator,
          typename FlagType,
          typename IndexType,
          typename ValueType>
__device__ __thrust_forceinline__
void reduce_by_key_body(Context context,
                        const unsigned int n,
                        InputIterator1   ikeys,
                        InputIterator2   ivals,
                        OutputIterator1  okeys,
                        OutputIterator2  ovals,
                        BinaryPredicate  binary_pred,
                        BinaryFunction   binary_op,
                        FlagIterator     iflags,
                        FlagType  (&sflag)[CTA_SIZE],
                        ValueType (&sdata)[K][CTA_SIZE + 1],
                        bool&      carry_in,
                        IndexType& carry_index,
                        ValueType& carry_value)
{
  // load flags
  const FlagType flag_bits  = load_flags<CTA_SIZE,K,FullBlock>(context, n, iflags, sflag);
  const FlagType flag_count = __popc(flag_bits); // TODO hide this behind a template
  const FlagType left_flag  = (context.thread_index() == 0) ? 0 : sflag[context.thread_index() - 1];
  const FlagType head_flag  = (context.thread_index() == 0 || flag_bits & ((1 << (K - 1)) - 1) || left_flag & (1 << (K - 1))) ? 1 : 0;
  
  context.barrier();

  // scan flag counts
  sflag[context.thread_index()] = flag_count; context.barrier();

  block::inclusive_scan(context, sflag, thrust::plus<FlagType>());

  const FlagType output_position = (context.thread_index() == 0) ? 0 : sflag[context.thread_index() - 1];
  const FlagType num_outputs     = sflag[CTA_SIZE - 1];

  context.barrier();

  // shuffle keys and write keys out
  if (!thrust::detail::is_discard_iterator<OutputIterator1>::value)
  {
    // XXX this could be improved
    for (unsigned int i = 0; i < num_outputs; i += CTA_SIZE)
    {
      FlagType position = output_position;

      for(unsigned int k = 0; k < K; k++)
      {
        if (flag_bits & (FlagType(1) << k))
        {
          if (i <= position && position < i + CTA_SIZE)
            sflag[position - i] = K * context.thread_index() + k;
          position++;
        }
      }

      context.barrier();

      if (i + context.thread_index() < num_outputs)
      {
        InputIterator1  tmp1 = ikeys + sflag[context.thread_index()];
        OutputIterator1 tmp2 = okeys + (i + context.thread_index());
        *tmp2 = *tmp1; 
      }
      
      context.barrier();
    }
  }

  // load values
  load_values<CTA_SIZE,K,FullBlock> (context, n, ivals, sdata);

  ValueType ldata[K];
  for (unsigned int k = 0; k < K; k++)
      ldata[k] = sdata[k][context.thread_index()];

  // carry in (if necessary)
  if (context.thread_index() == 0 && carry_in)
  {
    // XXX WAR sm_10 issue
    ValueType tmp1 = carry_value;
    ldata[0] = binary_op(tmp1, ldata[0]);
  }

  context.barrier();

  // sum local values
  {
    for(unsigned int k = 1; k < K; k++)
    {
      const unsigned int offset = K * context.thread_index() + k;

      if (FullBlock || offset < n)
      {
        if (!(flag_bits & (FlagType(1) << (k - 1))))
          ldata[k] = binary_op(ldata[k - 1], ldata[k]);
      }
    }
  }

  // second level segmented scan
  {
    // use head flags for segmented scan
    sflag[context.thread_index()] = head_flag;  sdata[K - 1][context.thread_index()] = ldata[K - 1]; context.barrier();

    if (FullBlock)
      block::inclusive_scan_by_flag(context, sflag, sdata[K-1], binary_op);
    else
      block::inclusive_scan_by_flag_n(context, sflag, sdata[K-1], n, binary_op);
  }

  // update local values
  if (context.thread_index() > 0)
  {
    unsigned int update_bits  = (flag_bits << 1) | (left_flag >> (K - 1));
// TODO remove guard
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    unsigned int update_count = __ffs(update_bits) - 1u; // NB: this might wrap around to UINT_MAX
#else
    unsigned int update_count = 0;
#endif // THRUST_DEVICE_COMPILER_NVCC

    if (!FullBlock && (K + 1) * context.thread_index() > n)
      update_count = thrust::min(n - K * context.thread_index(), update_count);

    ValueType left = sdata[K - 1][context.thread_index() - 1];

    for(unsigned int k = 0; k < K; k++)
    {
      if (k < update_count)
        ldata[k] = binary_op(left, ldata[k]);
    }
  }
  
  context.barrier();

  // store carry out
  if (FullBlock)
  {
    if (context.thread_index() == CTA_SIZE - 1)
    {
      carry_value = ldata[K - 1];
      carry_in    = (flag_bits & (FlagType(1) << (K - 1))) ? false : true;
      carry_index = num_outputs;
    }
  }
  else
  {
    if (context.thread_index() == (n - 1) / K)
    {
      for (unsigned int k = 0; k < K; k++)
          if (k == (n - 1) % K)
              carry_value = ldata[k];
      carry_in    = (flag_bits & (FlagType(1) << ((n - 1) % K))) ? false : true;
      carry_index = num_outputs;
    }
  }

  // shuffle values
  {
    FlagType position = output_position;
  
    for(unsigned int k = 0; k < K; k++)
    {
      const unsigned int offset = K * context.thread_index() + k;
  
      if (FullBlock || offset < n)
      {
        if (flag_bits & (FlagType(1) << k))
        {
          sdata[position / CTA_SIZE][position % CTA_SIZE] = ldata[k];
          position++;
        }
      }
    }
  }

  context.barrier();


  // write values out
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = CTA_SIZE * k + context.thread_index();

    if (offset < num_outputs)
    {
      OutputIterator2 tmp = ovals + offset;
      *tmp = sdata[k][context.thread_index()];
    }
  }

  context.barrier();
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction,
          typename FlagIterator,
          typename IndexIterator,
          typename ValueIterator,
          typename BoolIterator,
          typename Decomposition,
          typename Context>
struct reduce_by_key_closure
{
  InputIterator1   ikeys;
  InputIterator2   ivals;
  OutputIterator1  okeys;
  OutputIterator2  ovals;
  BinaryPredicate  binary_pred;
  BinaryFunction   binary_op;
  FlagIterator     iflags;
  IndexIterator    interval_counts;
  ValueIterator    interval_values;
  BoolIterator     interval_carry;
  Decomposition    decomp;
  Context          context;

  typedef Context context_type;

  reduce_by_key_closure(InputIterator1   ikeys,
                        InputIterator2   ivals,
                        OutputIterator1  okeys,
                        OutputIterator2  ovals,
                        BinaryPredicate  binary_pred,
                        BinaryFunction   binary_op,
                        FlagIterator     iflags,
                        IndexIterator    interval_counts,
                        ValueIterator    interval_values,
                        BoolIterator     interval_carry,
                        Decomposition    decomp,
                        Context          context = Context())
    : ikeys(ikeys), ivals(ivals), okeys(okeys), ovals(ovals), binary_pred(binary_pred), binary_op(binary_op),
      iflags(iflags), interval_counts(interval_counts), interval_values(interval_values), interval_carry(interval_carry),
      decomp(decomp), context(context) {}

  __device__ __thrust_forceinline__
  void operator()(void)
  {
    typedef typename thrust::iterator_value<InputIterator1>::type KeyType;
    typedef typename thrust::iterator_value<ValueIterator>::type  ValueType;
    typedef typename Decomposition::index_type                    IndexType;
    typedef typename thrust::iterator_value<FlagIterator>::type   FlagType;

    const unsigned int CTA_SIZE = context_type::ThreadsPerBlock::value;

// TODO centralize this mapping (__CUDA_ARCH__ -> smem bytes)
#if __CUDA_ARCH__ >= 200
    const unsigned int SMEM = (48 * 1024);
#else
    const unsigned int SMEM = (16 * 1024) - 256;
#endif
    const unsigned int SMEM_FIXED = CTA_SIZE * sizeof(FlagType) + sizeof(ValueType) + sizeof(IndexType) + sizeof(bool);
    const unsigned int BOUND_1 = (SMEM - SMEM_FIXED) / ((CTA_SIZE + 1) * sizeof(ValueType));
    const unsigned int BOUND_2 = 8 * sizeof(FlagType);
    const unsigned int BOUND_3 = 6;
  
    // TODO replace this with a static_min<BOUND_1,BOUND_2,BOUND_3>::value
    const unsigned int K = (BOUND_1 < BOUND_2) ? (BOUND_1 < BOUND_3 ? BOUND_1 : BOUND_3) : (BOUND_2 < BOUND_3 ? BOUND_2 : BOUND_3);
  
    __shared__ detail::uninitialized<FlagType[CTA_SIZE]>         sflag;
    __shared__ detail::uninitialized<ValueType[K][CTA_SIZE + 1]> sdata;  // padded to avoid bank conflicts
  
    __shared__ detail::uninitialized<ValueType> carry_value; // storage for carry in and carry out
    __shared__ detail::uninitialized<IndexType> carry_index;
    __shared__ detail::uninitialized<bool>      carry_in; 

    typename Decomposition::range_type interval = decomp[context.block_index()];
    //thrust::system::detail::internal::index_range<IndexType> interval = decomp[context.block_index()];
  

    if (context.thread_index() == 0)
    {
      carry_in = false; // act as though the previous segment terminated just before us
  
      if (context.block_index() == 0)
      {
        carry_index = 0;
      }
      else
      {
        interval_counts += (context.block_index() - 1);
        carry_index = *interval_counts;
      }
    }
  
    context.barrier();
  
    IndexType base = interval.begin();
  
    // advance input and output iterators
    ikeys  += base;
    ivals  += base;
    iflags += base;
    okeys  += carry_index;
    ovals  += carry_index;
  
    const unsigned int unit_size = K * CTA_SIZE;
  
    // process full units
    while (base + unit_size <= interval.end())
    {
      const unsigned int n = unit_size;
      reduce_by_key_body<CTA_SIZE,K,true>(context, n, ikeys, ivals, okeys, ovals, binary_pred, binary_op, iflags, sflag.get(), sdata.get(), carry_in.get(), carry_index.get(), carry_value.get());
      base   += unit_size;
      ikeys  += unit_size;
      ivals  += unit_size;
      iflags += unit_size;
      okeys  += carry_index;
      ovals  += carry_index;
    }
  
    // process partially full unit at end of input (if necessary)
    if (base < interval.end())
    {
      const unsigned int n = interval.end() - base;
      reduce_by_key_body<CTA_SIZE,K,false>(context, n, ikeys, ivals, okeys, ovals, binary_pred, binary_op, iflags, sflag.get(), sdata.get(), carry_in.get(), carry_index.get(), carry_value.get());
    }
  
    if (context.thread_index() == 0)
    {
      interval_values += context.block_index();
      interval_carry  += context.block_index();
      *interval_values = carry_value;
      *interval_carry  = carry_in;
    }
  }
}; // end reduce_by_key_closure

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
struct DefaultPolicy
{
  // typedefs
  typedef unsigned int                                                       FlagType;
  typedef typename thrust::iterator_traits<InputIterator1>::difference_type  IndexType;
  typedef typename thrust::iterator_traits<InputIterator1>::value_type       KeyType;
  typedef thrust::system::detail::internal::uniform_decomposition<IndexType> Decomposition;
    
  // the pseudocode for deducing the type of the temporary used below:
  // 
  // if BinaryFunction is AdaptableBinaryFunction
  //   TemporaryType = AdaptableBinaryFunction::result_type
  // else if OutputIterator2 is a "pure" output iterator
  //   TemporaryType = InputIterator2::value_type
  // else
  //   TemporaryType = OutputIterator2::value_type
  //
  // XXX upon c++0x, TemporaryType needs to be:
  // result_of<BinaryFunction>::type

  typedef typename thrust::detail::eval_if<
    thrust::detail::has_result_type<BinaryFunction>::value,
    thrust::detail::result_type<BinaryFunction>,
    thrust::detail::eval_if<
      thrust::detail::is_output_iterator<OutputIterator2>::value,
      thrust::iterator_value<InputIterator2>,
      thrust::iterator_value<OutputIterator2>
    >
  >::type ValueType;
 
  // XXX WAR problem on sm_11
  // TODO tune this
  const static unsigned int ThreadsPerBlock = (thrust::detail::is_pod<ValueType>::value) ? 256 : 192;

  DefaultPolicy(InputIterator1 first1, InputIterator1 last1)
    : decomp(default_decomposition<IndexType>(last1 - first1))
  {}

  // member variables
  Decomposition decomp;
};

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction,
          typename Policy>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(execution_policy<DerivedPolicy> &exec,
                InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op,
                Policy policy)
{
    typedef typename Policy::FlagType       FlagType;
    typedef typename Policy::Decomposition  Decomposition;
    typedef typename Policy::IndexType      IndexType;
    typedef typename Policy::KeyType        KeyType;
    typedef typename Policy::ValueType      ValueType;

    // temporary arrays
    typedef thrust::detail::temporary_array<IndexType,DerivedPolicy> IndexArray;
    typedef thrust::detail::temporary_array<KeyType,DerivedPolicy>   KeyArray;
    typedef thrust::detail::temporary_array<ValueType,DerivedPolicy> ValueArray;
    typedef thrust::detail::temporary_array<bool,DerivedPolicy>      BoolArray;

    Decomposition decomp = policy.decomp;

    // input size
    IndexType n = keys_last - keys_first;

    if (n == 0)
      return thrust::make_pair(keys_output, values_output);

    IndexArray interval_counts(exec, decomp.size());
    ValueArray interval_values(exec, decomp.size());
    BoolArray  interval_carry(exec, decomp.size());

    // an ode to c++11 auto
    typedef thrust::counting_iterator<IndexType> CountingIterator;
    typedef thrust::transform_iterator<
      tail_flag_functor<FlagType,IndexType,KeyType,BinaryPredicate>,
      thrust::zip_iterator<
        thrust::tuple<CountingIterator,InputIterator1,InputIterator1>
      >
    > FlagIterator;

    FlagIterator iflag= thrust::make_transform_iterator
       (thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), keys_first, keys_first + 1)),
        tail_flag_functor<FlagType,IndexType,KeyType,BinaryPredicate>(n, binary_pred));

    // count number of tail flags per interval
    thrust::system::cuda::detail::reduce_intervals(exec, iflag, interval_counts.begin(), thrust::plus<IndexType>(), decomp);

    thrust::inclusive_scan(exec,
                           interval_counts.begin(), interval_counts.end(),
                           interval_counts.begin(),
                           thrust::plus<IndexType>());
 
    // determine output size
    const IndexType N = interval_counts[interval_counts.size() - 1];
   
    const static unsigned int ThreadsPerBlock = Policy::ThreadsPerBlock;
    typedef typename IndexArray::iterator IndexIterator;
    typedef typename ValueArray::iterator ValueIterator; 
    typedef typename BoolArray::iterator  BoolIterator;  
    typedef detail::statically_blocked_thread_array<ThreadsPerBlock> Context;
    typedef reduce_by_key_closure<InputIterator1,InputIterator2,OutputIterator1,OutputIterator2,BinaryPredicate,BinaryFunction,
                                  FlagIterator,IndexIterator,ValueIterator,BoolIterator,Decomposition,Context> Closure;
    Closure closure
      (keys_first,  values_first,
       keys_output, values_output,
       binary_pred, binary_op,
       iflag,
       interval_counts.begin(),
       interval_values.begin(),
       interval_carry.begin(),
       decomp);
    detail::launch_closure(closure, decomp.size(), ThreadsPerBlock);
   
    if (decomp.size() > 1)
    {
      ValueArray interval_values2(exec, decomp.size());
      IndexArray interval_counts2(exec, decomp.size());
      BoolArray  interval_carry2(exec, decomp.size());

      IndexType N2 = 
      thrust::reduce_by_key
        (exec,
         thrust::make_zip_iterator(thrust::make_tuple(interval_counts.begin(), interval_carry.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(interval_counts.end(),   interval_carry.end())),
         interval_values.begin(),
         thrust::make_zip_iterator(thrust::make_tuple(interval_counts2.begin(), interval_carry2.begin())),
         interval_values2.begin(),
         thrust::equal_to< thrust::tuple<IndexType,bool> >(),
         binary_op).first
        -
        thrust::make_zip_iterator(thrust::make_tuple(interval_counts2.begin(), interval_carry2.begin()));
    
      thrust::transform_if
        (exec,
         interval_values2.begin(), interval_values2.begin() + N2,
         thrust::make_permutation_iterator(values_output, interval_counts2.begin()),
         interval_carry2.begin(),
         thrust::make_permutation_iterator(values_output, interval_counts2.begin()),
         binary_op,
         thrust::identity<bool>());
    }
  
    return thrust::make_pair(keys_output + N, values_output + N); 
}

} // end namespace reduce_by_key_detail


template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(execution_policy<DerivedPolicy> &exec,
                InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  return reduce_by_key_detail::reduce_by_key
    (exec, 
     keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op,
     reduce_by_key_detail::DefaultPolicy<InputIterator1,InputIterator2,OutputIterator1,OutputIterator2,BinaryPredicate,BinaryFunction>(keys_first, keys_last));
} // end reduce_by_key()

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

