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

#include <thrust/detail/device/cuda/detail/split_for_set_operation.h>
#include <thrust/detail/device/cuda/detail/rank_iterator.h>
#include <thrust/detail/device/cuda/detail/get_set_operation_splitter_ranks.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
{

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename Compare,
         typename Size1,
         typename Size2>
  void split_for_set_operation
    ::operator()(RandomAccessIterator1 first1,
                 RandomAccessIterator1 last1,
                 RandomAccessIterator2 first2,
                 RandomAccessIterator2 last2,
                 RandomAccessIterator3 splitter_ranks1,
                 RandomAccessIterator4 splitter_ranks2,
                 Compare comp,
                 Size1 partition_size,
                 Size2 num_splitters_from_each_range)
{
  using namespace thrust::detail;

  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  typedef typename detail::rank_iterator<RandomAccessIterator1,Compare>::type RankIterator1;
  typedef typename detail::rank_iterator<RandomAccessIterator2,Compare>::type RankIterator2;

  // enumerate each key within its sub-segment of equivalent keys
  RankIterator1 key_ranks1 = detail::make_rank_iterator(first1, comp);
  RankIterator2 key_ranks2 = detail::make_rank_iterator(first2, comp);

  // zip up the keys with their ranks to disambiguate repeated elements during rank-finding
  typedef thrust::tuple<RandomAccessIterator1,RankIterator1> iterator_tuple1;
  typedef thrust::tuple<RandomAccessIterator2,RankIterator2> iterator_tuple2;
  typedef thrust::zip_iterator<iterator_tuple1> iterator_and_rank1;
  typedef thrust::zip_iterator<iterator_tuple2> iterator_and_rank2;

  iterator_and_rank1 first_and_rank1 =
    thrust::make_zip_iterator(thrust::make_tuple(first1, key_ranks1));
  iterator_and_rank1 last_and_rank1 = first_and_rank1 + num_elements1;

  iterator_and_rank2 first_and_rank2 =
    thrust::make_zip_iterator(thrust::make_tuple(first2, key_ranks2));
  iterator_and_rank2 last_and_rank2 = first_and_rank2 + num_elements2;

  // take into account the tuples when comparing
  typedef compare_first_less_second<Compare> splitter_compare;

  using namespace thrust::detail::device::cuda::detail;
  return get_set_operation_splitter_ranks(first_and_rank1, last_and_rank1,
                                          first_and_rank2, last_and_rank2,
                                          splitter_ranks1,
                                          splitter_ranks2,
                                          splitter_compare(comp),
                                          partition_size,
                                          num_splitters_from_each_range);
} // end split_for_set_operation::operator()

} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

