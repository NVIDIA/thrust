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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>

#include <thrust/detail/minmax.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/uninitialized_array.h>

#include <thrust/detail/backend/scan.h>
#include <thrust/detail/backend/cuda/synchronize.h>
#include <thrust/detail/backend/cuda/reduce_intervals.h>
#include <thrust/detail/backend/cuda/default_decomposition.h>
#include <thrust/detail/backend/cuda/block/inclusive_scan.h>
#include <thrust/detail/backend/cuda/detail/uninitialized.h>

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace detail
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
  __host__ __device__ __forceinline__
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
          typename FlagIterator,
          typename FlagType>
__device__ __forceinline__
FlagType load_flags(const unsigned int n,
                    FlagIterator iflags,
                    FlagType  (&sflag)[CTA_SIZE])
{
  FlagType flag_bits = 0;

  // load flags in unordered fashion
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = k*CTA_SIZE + threadIdx.x;

    if (FullBlock || offset < n)
    {
      FlagIterator temp = iflags + offset;
      if (thrust::detail::backend::dereference(temp))
        flag_bits |= FlagType(1) << k;
    }
  }

  sflag[threadIdx.x] = flag_bits;
  
  __syncthreads();

  flag_bits = 0;

  // obtain flags for iflags[K * threadIdx.x, K * threadIdx.x + K)
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = K * threadIdx.x + k;

    if (FullBlock || offset < n)
    {
      flag_bits |= ((sflag[offset % CTA_SIZE] >> (offset / CTA_SIZE)) & FlagType(1)) << k;
    }
  }

  __syncthreads();
  
  sflag[threadIdx.x] = flag_bits;
  
  __syncthreads();

  return flag_bits;
}

template <unsigned int CTA_SIZE,
          unsigned int K,
          bool FullBlock,
          typename InputIterator2,
          typename ValueType>
__device__ __forceinline__
void load_values(const unsigned int n,
                 InputIterator2 ivals,
                 ValueType (&sdata)[K][CTA_SIZE + 1])
{
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = k*CTA_SIZE + threadIdx.x;

    if (FullBlock || offset < n)
    {
      InputIterator2 temp = ivals + offset;
      sdata[offset % K][offset / K] = thrust::detail::backend::dereference(temp);
    }
  }

  __syncthreads();
}


template <unsigned int CTA_SIZE,
          unsigned int K,
          bool FullBlock,
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
__device__ __forceinline__
void reduce_by_key_body(const unsigned int n,
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
  const FlagType flag_bits  = load_flags<CTA_SIZE,K,FullBlock>(n, iflags, sflag);
  const FlagType flag_count = __popc(flag_bits); // TODO hide this behind a template
  const FlagType left_flag  = (threadIdx.x == 0) ? 0 : sflag[threadIdx.x - 1];
  const FlagType head_flag  = (threadIdx.x == 0 || flag_bits & ((1 << (K - 1)) - 1) || left_flag & (1 << (K - 1))) ? 1 : 0;
  
  __syncthreads();

  // scan flag counts
  sflag[threadIdx.x] = flag_count; __syncthreads();

  thrust::detail::backend::cuda::block::inplace_inclusive_scan<CTA_SIZE>(sflag, thrust::plus<FlagType>());

  const FlagType output_position = (threadIdx.x == 0) ? 0 : sflag[threadIdx.x - 1];
  const FlagType num_outputs     = sflag[CTA_SIZE - 1];

  __syncthreads();

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
            sflag[position - i] = K * threadIdx.x + k;
          position++;
        }
      }

      __syncthreads();

      if (i + threadIdx.x < num_outputs)
      {
        InputIterator1  tmp1 = ikeys + sflag[threadIdx.x];
        OutputIterator1 tmp2 = okeys + (i + threadIdx.x);
        dereference(tmp2) = dereference(tmp1); 
      }
      
      __syncthreads();
    }
  }

  // load values
  load_values<CTA_SIZE,K,FullBlock> (n, ivals, sdata);

  ValueType ldata[K];
  for (unsigned int k = 0; k < K; k++)
      ldata[k] = sdata[k][threadIdx.x];

  // carry in (if necessary)
  if (threadIdx.x == 0 && carry_in)
  {
    // XXX WAR sm_10 issue
    ValueType tmp1 = carry_value;
    ldata[0] = binary_op(tmp1, ldata[0]);
  }

  __syncthreads();

  // sum local values
  {
    for(unsigned int k = 1; k < K; k++)
    {
      const unsigned int offset = K * threadIdx.x + k;

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
    sflag[threadIdx.x] = head_flag;  sdata[K - 1][threadIdx.x] = ldata[K - 1]; __syncthreads();

    if (FullBlock)
      thrust::detail::backend::cuda::block::inplace_inclusive_segscan<CTA_SIZE>(sflag, sdata[K-1], binary_op);
    else
      thrust::detail::backend::cuda::block::inplace_inclusive_segscan_n(sflag, sdata[K-1], n, binary_op);
      
  }

  // update local values
  if (threadIdx.x > 0)
  {
    unsigned int update_bits  = (flag_bits << 1) | (left_flag >> (K - 1));
    unsigned int update_count = __ffs(update_bits) - 1u; // NB: this might wrap around to UINT_MAX

    if (!FullBlock && (K + 1) * threadIdx.x > n)
      update_count = thrust::min(n - K * threadIdx.x, update_count);

    ValueType left = sdata[K - 1][threadIdx.x - 1];

    for(unsigned int k = 0; k < K; k++)
    {
      if (k < update_count)
        ldata[k] = binary_op(left, ldata[k]);
    }
  }
  
  __syncthreads();

  // store carry out
  if (FullBlock)
  {
    if (threadIdx.x == CTA_SIZE - 1)
    {
      carry_value = ldata[K - 1];
      carry_in    = (flag_bits & (FlagType(1) << (K - 1))) ? false : true;
      carry_index = num_outputs;
    }
  }
  else
  {
    if (threadIdx.x == (n - 1) / K)
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
      const unsigned int offset = K * threadIdx.x + k;
  
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

  __syncthreads();


  // write values out
  for(unsigned int k = 0; k < K; k++)
  {
    const unsigned int offset = CTA_SIZE * k + threadIdx.x;

    if (offset < num_outputs)
    {
      OutputIterator2 tmp = ovals + offset;
      dereference(tmp) = sdata[k][threadIdx.x];
    }
  }

  __syncthreads();
}

template <unsigned int CTA_SIZE,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction,
          typename FlagIterator,
          typename IndexIterator,
          typename ValueIterator,
          typename BoolIterator,
          typename Decomposition>
__launch_bounds__(CTA_SIZE,1)          
__global__
void reduce_by_key_kernel(InputIterator1   ikeys,
                          InputIterator2   ivals,
                          OutputIterator1  okeys,
                          OutputIterator2  ovals,
                          BinaryPredicate  binary_pred,
                          BinaryFunction   binary_op,
                          FlagIterator     iflags,
                          IndexIterator    interval_counts,
                          ValueIterator    interval_values,
                          BoolIterator     interval_carry,
                          Decomposition    decomp)
{
  typedef typename thrust::iterator_value<InputIterator1>::type KeyType;
  typedef typename thrust::iterator_value<ValueIterator>::type  ValueType;
  typedef typename Decomposition::index_type                    IndexType;
  typedef typename thrust::iterator_value<FlagIterator>::type   FlagType;


#if __CUDA_ARCH__ >= 200
  const unsigned int SMEM = (48 * 1024);
#else
  const unsigned int SMEM = (16 * 1024);
#endif
  const unsigned int SMEM_FIXED = 256 + CTA_SIZE * sizeof(FlagType) + sizeof(ValueType) + sizeof(IndexType) + sizeof(bool);
  const unsigned int BOUND_1 = (SMEM - SMEM_FIXED)/ ((CTA_SIZE + 1) * sizeof(ValueType));
  const unsigned int BOUND_2 = 8 * sizeof(FlagType);
  const unsigned int BOUND_3 = 6;

  // TODO replace this with a static_min<BOUND_1,BOUND_2,BOUND_3>::value
  const unsigned int K = (BOUND_1 < BOUND_2) ? (BOUND_1 < BOUND_3 ? BOUND_1 : BOUND_3) : (BOUND_2 < BOUND_3 ? BOUND_2 : BOUND_3);

  __shared__ uninitialized<FlagType[CTA_SIZE]>         sflag; 
  __shared__ uninitialized<ValueType[K][CTA_SIZE + 1]> sdata;  // padded to avoid bank conflicts

  __shared__ detail::uninitialized<ValueType> carry_value; // storage for carry in and carry out
  __shared__ detail::uninitialized<IndexType> carry_index;
  __shared__ detail::uninitialized<bool>      carry_in; 

  thrust::detail::backend::index_range<IndexType> interval = decomp[blockIdx.x];

  if (threadIdx.x == 0)
  {
    carry_in = false; // act as though the previous segment terminated just before us

    if (blockIdx.x == 0)
    {
      carry_index = 0;
    }
    else
    {
      interval_counts += (blockIdx.x - 1);
      carry_index = dereference(interval_counts);
    }
  }

  __syncthreads();

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
    reduce_by_key_body<CTA_SIZE,K,true>(n, ikeys, ivals, okeys, ovals, binary_pred, binary_op, iflags, sflag.get(), sdata.get(), carry_in.get(), carry_index.get(), carry_value.get());
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
    reduce_by_key_body<CTA_SIZE,K,false>(n, ikeys, ivals, okeys, ovals, binary_pred, binary_op, iflags, sflag.get(), sdata.get(), carry_in.get(), carry_index.get(), carry_value.get());
  }

  if (threadIdx.x == 0)
  {
    interval_values += blockIdx.x;
    interval_carry  += blockIdx.x;
    dereference(interval_values) = carry_value;
    dereference(interval_carry)  = carry_in;
  }
}

} // end namespace detail

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
    typedef          unsigned int                                              FlagType;
    typedef typename thrust::iterator_traits<InputIterator1>::difference_type  IndexType;
    typedef typename thrust::iterator_traits<InputIterator1>::value_type       KeyType;
    
    typedef thrust::detail::backend::uniform_decomposition<IndexType>                            Decomposition;

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

    typedef typename eval_if<
      has_result_type<BinaryFunction>::value,
      result_type<BinaryFunction>,
      eval_if<
        is_output_iterator<OutputIterator2>::value,
        thrust::iterator_value<InputIterator2>,
        thrust::iterator_value<OutputIterator2>
      >
    >::type ValueType;
   
    // temporary arrays
    typedef thrust::detail::uninitialized_array<IndexType,thrust::detail::cuda_device_space_tag> IndexArray;
    typedef thrust::detail::uninitialized_array<KeyType,thrust::detail::cuda_device_space_tag>   KeyArray;
    typedef thrust::detail::uninitialized_array<ValueType,thrust::detail::cuda_device_space_tag> ValueArray;
    typedef thrust::detail::uninitialized_array<bool,thrust::detail::cuda_device_space_tag>      BoolArray;

    // input size
    IndexType n = keys_last - keys_first;

    if (n == 0)
      return thrust::make_pair(keys_output, values_output);
 
    Decomposition decomp = thrust::detail::backend::cuda::default_decomposition<IndexType>(n);

    IndexArray interval_counts(decomp.size());
    ValueArray interval_values(decomp.size());
    BoolArray  interval_carry(decomp.size());

    // an ode to c++11 auto
    typedef thrust::counting_iterator<IndexType> CountingIterator;
    typedef thrust::transform_iterator
      <detail::tail_flag_functor<FlagType,IndexType,KeyType,BinaryPredicate>,
       thrust::zip_iterator< thrust::tuple<CountingIterator,InputIterator1,InputIterator1> > > FlagIterator;

    FlagIterator iflag= thrust::make_transform_iterator
       (thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), keys_first, keys_first + 1)),
        detail::tail_flag_functor<FlagType,IndexType,KeyType,BinaryPredicate>(n, binary_pred));

    // count number of tail flags per interval
    thrust::detail::backend::cuda::reduce_intervals(iflag, interval_counts.begin(), thrust::plus<IndexType>(), decomp);

//    std::cout << std::endl << "-----------------------------------" << std::endl;
//
//    std::cout << "tail flag counts" << std::endl;
//    for (IndexType i = 0; i < decomp.size(); i++)
//    {
//      std::cout << "[" << decomp[i].begin() << "," << decomp[i].end() << ") = " << interval_counts[i] << std::endl;
//    }

    thrust::detail::backend::cuda::inclusive_scan(interval_counts.begin(), interval_counts.end(),
                                                  interval_counts.begin(),
                                                  thrust::plus<IndexType>());
 
    // determine output size
    const IndexType N = interval_counts[interval_counts.size() - 1];
    
    // TODO tune this
    const static unsigned int CTA_SIZE = 8 * 32;

    // launch kernel
    detail::reduce_by_key_kernel<CTA_SIZE><<<decomp.size(), CTA_SIZE>>>
      (keys_first,  values_first,
       keys_output, values_output,
       binary_pred, binary_op,
       iflag,
       interval_counts.begin(),
       interval_values.begin(),
       interval_carry.begin(),
       decomp);
    synchronize_if_enabled("reduce_by_key_kernel");
   
//    std::cout << "interval (value,carry) " << std::endl;
//    for (IndexType i = 0; i < decomp.size(); i++)
//    {
//      std::cout << "[" << decomp[i].begin() << "," << decomp[i].end() << ") = " << interval_values[i] << " " << interval_carry[i] << std::endl;
//    }
    

    if (decomp.size() > 1)
    {
      ValueArray interval_values2(decomp.size());
      IndexArray interval_counts2(decomp.size());
      BoolArray  interval_carry2(decomp.size());

      IndexType N2 = 
      thrust::detail::backend::cuda::reduce_by_key
        (thrust::make_zip_iterator(thrust::make_tuple(interval_counts.begin(), interval_carry.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(interval_counts.end(),   interval_carry.end())),
         interval_values.begin(),
         thrust::make_zip_iterator(thrust::make_tuple(interval_counts2.begin(), interval_carry2.begin())),
         interval_values2.begin(),
         thrust::equal_to< thrust::tuple<IndexType,bool> >(),
         binary_op).first
        -
        thrust::make_zip_iterator(thrust::make_tuple(interval_counts2.begin(), interval_carry2.begin()));
    
//      std::cout << "second level " << std::endl;
//      for (IndexType i = 0; i < N2; i++)
//      {
//        std::cout << interval_values2[i] << " " << interval_counts2[i] << " " << interval_carry2[i] << std::endl;
//      }

      thrust::transform_if
        (interval_values2.begin(), interval_values2.begin() + N2,
         thrust::make_permutation_iterator(values_output, interval_counts2.begin()),
         interval_carry2.begin(),
         thrust::make_permutation_iterator(values_output, interval_counts2.begin()),
         binary_op,
         thrust::identity<bool>());
    }
  
    return thrust::make_pair(keys_output + N, values_output + N); 
}

} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

