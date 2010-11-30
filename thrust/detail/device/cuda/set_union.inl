/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/cuda/block/set_union.h>
#include <thrust/detail/device/cuda/detail/get_set_operation_splitter_ranks.h>
#include <thrust/detail/device/cuda/detail/set_operation.h>
#include <thrust/iterator/zip_iterator.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

namespace set_union_detail
{

struct block_convergent_set_union_functor
{
  __host__ __device__ __forceinline__
  static unsigned int get_temporary_array_size(unsigned int block_size)
  {
    return block_size * sizeof(int);
  }

  // operator() simply calls the block-wise function
  template<typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename StrictWeakOrdering>
  __device__ __forceinline__
    RandomAccessIterator3 operator()(RandomAccessIterator1 first1,
                                     RandomAccessIterator1 last1,
                                     RandomAccessIterator2 first2,
                                     RandomAccessIterator2 last2,
                                     void *temporary,
                                     RandomAccessIterator3 result,
                                     StrictWeakOrdering comp)
  {
    return block::set_union(first1,last1,first2,last2,reinterpret_cast<int*>(temporary),result,comp);
  } // end operator()()
}; // end block_convergent_set_union_functor



// this functor keeps an iterator pointing to a sorted range and a Compare
// operator() takes an index as an argument, looks up x = first[index]
// and returns x's rank in the segment of elements equivalent to x
template<typename RandomAccessIterator, typename Compare>
  struct nth_occurrence_functor
    : thrust::unary_function<
        typename thrust::iterator_difference<RandomAccessIterator>::type,
        typename thrust::iterator_difference<RandomAccessIterator>::type
      >
{
  nth_occurrence_functor(RandomAccessIterator f, Compare c)
    : first(f), comp(c) {}

  template<typename Index>
  __host__ __device__ __forceinline__
  typename thrust::iterator_difference<RandomAccessIterator>::type operator()(Index index)
  {
    RandomAccessIterator x = first;
    x += index;

    return x - thrust::detail::device::generic::scalar::lower_bound(first,x,dereference(x),comp);
  }

  RandomAccessIterator first;
  Compare comp;
}; // end nth_occurrence_functor


template<typename RandomAccessIterator, typename Compare>
  class rank_iterator
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference;
  typedef thrust::counting_iterator<difference> counter;

  public:
    typedef thrust::transform_iterator<
      nth_occurrence_functor<RandomAccessIterator,Compare>,
      counter
    > type;
}; // end rank_iterator


template<typename RandomAccessIterator, typename Compare>
  typename rank_iterator<RandomAccessIterator,Compare>::type
    make_rank_iterator(RandomAccessIterator iter, Compare comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference;
  typedef thrust::counting_iterator<difference> CountingIterator;

  nth_occurrence_functor<RandomAccessIterator,Compare> f(iter,comp);

  return thrust::make_transform_iterator(CountingIterator(0), f);
} // end make_rank_iterator()


// XXX this is redundant with split_for_set_intersection
struct split_for_set_union
{
  template<typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4,
           typename Compare,
           typename Size1,
           typename Size2>
    void operator()(RandomAccessIterator1 first1,
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
  
    typedef typename rank_iterator<RandomAccessIterator1,Compare>::type RankIterator1;
    typedef typename rank_iterator<RandomAccessIterator2,Compare>::type RankIterator2;
  
    // enumerate each key within its sub-segment of equivalent keys
    RankIterator1 key_ranks1 = make_rank_iterator(first1, comp);
    RankIterator2 key_ranks2 = make_rank_iterator(first2, comp);
  
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
  } // end operator()
}; // end split_for_set_union

} // end namespace set_union_detail


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename Compare>
RandomAccessIterator3 set_union(RandomAccessIterator1 first1,
                                RandomAccessIterator1 last1,
                                RandomAccessIterator2 first2,
                                RandomAccessIterator2 last2,
                                RandomAccessIterator3 result,
                                Compare comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  // check for trivial problem
  if(num_elements1 == 0 && num_elements2 == 0)
    return result;

  return detail::set_operation(first1, last1,
                               first2, last2,
                               result,
                               comp,
                               thrust::make_pair(thrust::max<size_t>(num_elements1, num_elements2), num_elements1 + num_elements2),
                               set_union_detail::split_for_set_union(),
                               set_union_detail::block_convergent_set_union_functor());
} // end set_union

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

