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


/*! \file reduce.inl
 *  \brief Inline file for reduce.h
 */

#include <thrust/detail/config.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/bulk.h>
#include <thrust/system/cuda/detail/decomposition.h>
#include <thrust/system/cuda/detail/execution_policy.h>

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
    T sum = bulk::reduce(this_group, first, last, init, binary_op);

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


} // end reduce_detail


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(execution_policy<DerivedPolicy> &exec,
                    InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
  typedef typename thrust::iterator_difference<InputIterator>::type size_type;

  const size_type n = last - first;

  if(n <= 0) return init;

  const size_type groupsize = 128;
  const size_type grainsize = 9;
  const size_type tile_size = groupsize * grainsize;
  const size_type num_tiles = (n + tile_size - 1) / tile_size;
  const size_type subscription = 10;

  bulk::concurrent_group<
    bulk::agent<grainsize>,
    groupsize
  > g;

  const size_type num_groups = thrust::min<size_type>(subscription * g.hardware_concurrency(), num_tiles);

  aligned_decomposition<size_type> decomp(n, num_groups, tile_size);

  thrust::cuda::tag t;
  thrust::detail::temporary_array<OutputType,thrust::cuda::tag> partial_sums(t, decomp.size());

  // reduce into partial sums
  bulk::async(bulk::par(g, decomp.size()), reduce_detail::reduce_partitions(), bulk::root.this_exec, first, decomp, partial_sums.begin(), init, binary_op).wait();

  if(partial_sums.size() > 1)
  {
    // reduce the partial sums
    bulk::async(g, reduce_detail::reduce_partitions(), bulk::root, partial_sums.begin(), partial_sums.end(), partial_sums.begin(), binary_op);
  } // end while

  return partial_sums[0];
} // end reduce()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

