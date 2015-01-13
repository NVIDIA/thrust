/*
 *  Copyright 2008-2013 NVIDIA Corporation
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


/*! \file reduce.inl
 *  \brief Inline file for reduce.h
 */

#include <thrust/detail/config.h>
#include <thrust/reduce.h>
#include <thrust/detail/seq.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/bulk.h>
#include <thrust/system/cuda/detail/decomposition.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/execute_on_stream.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace reduce_detail
{


struct reduce_partitions
{
  template<typename ConcurrentGroup, typename Iterator1, typename Iterator2, typename T, typename BinaryOperation>
  __device__
  void operator()(ConcurrentGroup &this_group, Iterator1 first, Iterator1 last, Iterator2 result, T init, BinaryOperation binary_op)
  {
    T sum = bulk_::reduce(this_group, first, last, init, binary_op);

    if(this_group.this_exec.index() == 0)
    {
      *result = sum;
    }
  }

  template<typename ConcurrentGroup, typename Iterator1, typename Iterator2, typename BinaryOperation>
  __device__
  void operator()(ConcurrentGroup &this_group, Iterator1 first, Iterator1 last, Iterator2 result, BinaryOperation binary_op)
  {
    // noticeably faster to pass the last element as the init
    typename thrust::iterator_value<Iterator2>::type init = thrust::raw_reference_cast(last[-1]);
    (*this)(this_group, first, last - 1, result, init, binary_op);
  }


  template<typename ConcurrentGroup, typename Iterator1, typename Decomposition, typename Iterator2, typename T, typename BinaryFunction>
  __device__
  void operator()(ConcurrentGroup &this_group, Iterator1 first, Decomposition decomp, Iterator2 result, T init, BinaryFunction binary_op)
  {
    typename Decomposition::range range = decomp[this_group.index()];

    Iterator1 last = first + range.second;
    first += range.first;

    if(this_group.index() != 0)
    {
      // noticeably faster to pass the last element as the init 
      init = thrust::raw_reference_cast(last[-1]);
      --last;
    } // end if

    (*this)(this_group, first, last, result + this_group.index(), init, binary_op);
  }
};


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
OutputType tuned_reduce(execution_policy<DerivedPolicy> &exec,
                        InputIterator first,
                        InputIterator last,
                        OutputType init,
                        BinaryFunction binary_op)
{
  typedef typename thrust::iterator_difference<InputIterator>::type size_type;

  const size_type n = last - first;

  if(n <= 0) return init;

  cudaStream_t s = stream(thrust::detail::derived_cast(exec));

  const size_type groupsize = 128;
  const size_type grainsize = 7;
  const size_type tile_size = groupsize * grainsize;
  const size_type num_tiles = (n + tile_size - 1) / tile_size;
  const size_type subscription = 10;

  bulk_::concurrent_group<
    bulk_::agent<grainsize>,
    groupsize
  > g;

  const size_type num_groups = thrust::min<size_type>(subscription * g.hardware_concurrency(), num_tiles);

  aligned_decomposition<size_type> decomp(n, num_groups, tile_size);

  thrust::detail::temporary_array<OutputType,DerivedPolicy> partial_sums(exec, decomp.size());

  // reduce into partial sums
  bulk_::async(bulk_::par(s, g, decomp.size()), reduce_detail::reduce_partitions(), bulk_::root.this_exec, first, decomp, partial_sums.begin(), init, binary_op).wait();

  if(partial_sums.size() > 1)
  {
    // reduce the partial sums
    bulk_::async(bulk_::par(s, g, 1), reduce_detail::reduce_partitions(), bulk_::root.this_exec, partial_sums.begin(), partial_sums.end(), partial_sums.begin(), binary_op);
  } // end while

  return partial_sums[0];
} // end tuned_reduce()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
OutputType general_reduce(execution_policy<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          OutputType init,
                          BinaryFunction binary_op)
{
  typedef typename thrust::iterator_difference<InputIterator>::type size_type;

  const size_type n = last - first;

  if(n <= 0) return init;

  cudaStream_t s = stream(thrust::detail::derived_cast(exec));

  typedef thrust::detail::temporary_array<OutputType,thrust::cuda::tag> temporary_array;

  // automatically choose a number of groups and a group size
  size_type num_groups = 0;
  size_type group_size = 0;

  thrust::tie(num_groups, group_size) = bulk_::choose_sizes(bulk_::grid(), reduce_partitions(), bulk_::root.this_exec, first, uniform_decomposition<size_type>(), typename temporary_array::iterator(), init, binary_op);

  num_groups = thrust::min<size_type>(num_groups, thrust::detail::util::divide_ri(n, group_size));

  uniform_decomposition<size_type> decomp(n, num_groups);

  thrust::cuda::tag t;
  temporary_array partial_sums(t, decomp.size());

  // reduce into partial sums
  bulk_::async(bulk_::grid(decomp.size(), group_size, bulk_::use_default, s), reduce_partitions(), bulk_::root.this_exec, first, decomp, partial_sums.begin(), init, binary_op);

  if(partial_sums.size() > 1)
  {
    // need to rechoose the group_size because the type of the kernel launch below differs from the first one
    thrust::tie(num_groups, group_size) = bulk_::choose_sizes(bulk_::grid(1), reduce_partitions(), bulk_::root.this_exec, partial_sums.begin(), partial_sums.end(), partial_sums.begin(), binary_op);

    // reduce the partial sums
    bulk_::async(bulk_::grid(num_groups, group_size, bulk_::use_default, s), reduce_partitions(), bulk_::root.this_exec, partial_sums.begin(), partial_sums.end(), partial_sums.begin(), binary_op);
  } // end while

  return partial_sums[0];
} // end general_reduce()


// use a tuned implementation for arithmetic types
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
typename thrust::detail::enable_if<
  thrust::detail::is_arithmetic<OutputType>::value,
  OutputType
>::type
  reduce(execution_policy<DerivedPolicy> &exec,
         InputIterator first,
         InputIterator last,
         OutputType init,
         BinaryFunction binary_op)
{
  return reduce_detail::tuned_reduce(exec, first, last, init, binary_op);
} // end reduce()


// use a general implementation for non-arithmetic types
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
typename thrust::detail::disable_if<
  thrust::detail::is_arithmetic<OutputType>::value,
  OutputType
>::type
  reduce(execution_policy<DerivedPolicy> &exec,
         InputIterator first,
         InputIterator last,
         OutputType init,
         BinaryFunction binary_op)
{
  return reduce_detail::general_reduce(exec, first, last, init, binary_op);
} // end reduce()



} // end reduce_detail


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
OutputType reduce(execution_policy<DerivedPolicy> &exec,
                  InputIterator first,
                  InputIterator last,
                  OutputType init,
                  BinaryFunction binary_op)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  struct workaround
  {
    __host__ __device__
    static OutputType parallel_path(execution_policy<DerivedPolicy> &exec,
                                    InputIterator first,
                                    InputIterator last,
                                    OutputType init,
                                    BinaryFunction binary_op)
    {
      return thrust::system::cuda::detail::reduce_detail::reduce(exec, first, last, init, binary_op);
    }

    __host__ __device__
    static OutputType sequential_path(execution_policy<DerivedPolicy> &,
                                      InputIterator first,
                                      InputIterator last,
                                      OutputType init,
                                      BinaryFunction binary_op)
    {
      return thrust::reduce(thrust::seq, first, last, init, binary_op);
    }
  };

#if __BULK_HAS_CUDART__
  return workaround::parallel_path(exec, first, last, init, binary_op);
#else
  return workaround::sequential_path(exec, first, last, init, binary_op);
#endif
} // end reduce()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

