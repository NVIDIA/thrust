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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>
#include <thrust/detail/device/cuda/block/copy.h>
#include <thrust/detail/device/cuda/block/merge.h>
#include <thrust/detail/device/cuda/scalar/binary_search.h>
#include <thrust/detail/device/cuda/synchronize.h>
#include <thrust/detail/device/generic/scalar/select.h>
#include <thrust/sort.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

namespace scalar
{

template<typename ForwardIterator,
         typename StrictWeakCompare>
__device__ __forceinline__
  bool is_sorted(ForwardIterator first,
                 ForwardIterator last,
                 StrictWeakCompare comp)
{
  ForwardIterator next = first;
  ++next;
  for(; next < last; ++first, ++next)
  {
    if(comp(*next, *first))
    {
      return false;
    }
  }

  return true;
}

}

namespace merge_detail
{

template<typename T1, typename T2>
T1 ceil_div(T1 up, T2 down)
{
  T1 div = up / down;
  T1 rem = up % down;
  return (rem != 0) ? div + 1 : div;
}

template<unsigned int N>
  struct align_size_to_int
{
  static const unsigned int value = (N / sizeof(int)) + ((N % sizeof(int)) ? 1 : 0);
};

// the iterator arguments are declared const because they are not to be
// modified during "strip mining" in the merge implementation
template<unsigned int block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename StrictWeakOrdering,
         typename Size>
__launch_bounds__(block_size, 1)         
__global__ void merge_kernel(const RandomAccessIterator1 first1, 
                             const RandomAccessIterator1 last1,
                             const RandomAccessIterator2 first2,
                             const RandomAccessIterator2 last2,
                             RandomAccessIterator3 splitter_ranks1,
                             RandomAccessIterator4 splitter_ranks2,
                             const RandomAccessIterator5 result,
                             StrictWeakOrdering comp,
                             Size num_merged_partitions)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;
  typedef typename thrust::iterator_value<RandomAccessIterator5>::type value_type5;

  // allocate shared storage
  const unsigned int array_size1 = align_size_to_int<block_size * sizeof(value_type1)>::value;
  const unsigned int array_size2 = align_size_to_int<block_size * sizeof(value_type2)>::value;
  const unsigned int array_size3 = align_size_to_int<2 * block_size * sizeof(value_type5)>::value;
  __shared__ int _shared1[array_size1];
  __shared__ int _shared2[array_size2];
  __shared__ int _result[array_size3];

  value_type1 *s_input1 = reinterpret_cast<value_type1*>(_shared1);
  value_type2 *s_input2 = reinterpret_cast<value_type2*>(_shared2);
  value_type5 *s_result = reinterpret_cast<value_type5*>(_result);

  // advance splitter iterators
  splitter_ranks1 += blockIdx.x;
  splitter_ranks2 += blockIdx.x;

  for(Size partition_idx = blockIdx.x;
      partition_idx < num_merged_partitions;
      partition_idx   += gridDim.x,
      splitter_ranks1 += gridDim.x,
      splitter_ranks2 += gridDim.x)
  {
    RandomAccessIterator1 input_begin1 = first1;
    RandomAccessIterator1 input_end1   = last1;
    RandomAccessIterator2 input_begin2 = first2;
    RandomAccessIterator2 input_end2   = last2;

    RandomAccessIterator5 output_begin = result;

    // find the end of the input if this is not the last block
    // the end of merged partition i is at splitter_ranks1[i] + splitter_ranks2[i]
    if(partition_idx != num_merged_partitions - 1)
    {
      RandomAccessIterator3 rank1 = splitter_ranks1;
      RandomAccessIterator4 rank2 = splitter_ranks2;

      input_end1 = first1 + dereference(rank1);
      input_end2 = first2 + dereference(rank2);
    }

    // find the beginning of the input and output if this is not the first partition
    // merged partition i begins at splitter_ranks1[i-1] + splitter_ranks2[i-1]
    if(partition_idx != 0)
    {
      RandomAccessIterator3 rank1 = splitter_ranks1;
      --rank1;
      RandomAccessIterator4 rank2 = splitter_ranks2;
      --rank2;

      // advance the input to point to the beginning
      input_begin1 += dereference(rank1);
      input_begin2 += dereference(rank2);

      // advance the result to point to the beginning of the output
      output_begin += dereference(rank1);
      output_begin += dereference(rank2);
    }

    if(input_begin1 < input_end1 && input_begin2 < input_end2)
    {
      typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;

      typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

      // load the first segment
      difference1 s_input1_size = thrust::min<difference1>(block_size, input_end1 - input_begin1);

      block::copy(input_begin1, input_begin1 + s_input1_size, s_input1);
      input_begin1 += s_input1_size;

      // load the second segment
      difference2 s_input2_size = thrust::min<difference2>(block_size, input_end2 - input_begin2);

      block::copy(input_begin2, input_begin2 + s_input2_size, s_input2);
      input_begin2 += s_input2_size;

      __syncthreads();

      block::merge(s_input1, s_input1 + s_input1_size,
                   s_input2, s_input2 + s_input2_size,
                   s_result,
                   comp);

      __syncthreads();

      // store to gmem
      output_begin = block::copy(s_result, s_result + s_input1_size + s_input2_size, output_begin);
    }

    // simply copy any remaining input
    block::copy(input_begin2, input_end2, block::copy(input_begin1, input_end1, output_begin));
  } // end for partition
} // end merge_kernel


template<typename T>
struct mult_by
  : thrust::unary_function<T,T>
{
  T _value;
  
  mult_by(const T& v):_value(v){}
  
  __host__ __device__
  T operator()(const T& v) const
  {
    return _value * v;
  }
};

// this predicate tests two two-element tuples
// we first use a Compare for the first element
// if the first elements are equivalent, we use
// < for the second elements
template<typename Compare>
  struct compare_first_less_second
{
  compare_first_less_second(Compare c)
    : comp(c) {}

  template<typename Tuple>
  __host__ __device__
  bool operator()(Tuple lhs, Tuple rhs)
  {
    return comp(lhs.get<0>(), rhs.get<0>()) || (!comp(rhs.get<0>(), lhs.get<0>()) && lhs.get<1>() < rhs.get<1>());
  }

  Compare comp;
}; // end compare_first_less_second

template<typename Iterator1, typename Iterator2, typename Compare>
  struct select_functor
{
  Iterator1 first1, last1;
  Iterator2 first2, last2;
  Compare comp;

  select_functor(Iterator1 f1, Iterator1 l1,
                 Iterator2 f2, Iterator2 l2,
                 Compare c)
    : first1(f1), last1(l1), first2(f2), last2(l2), comp(c)
  {}
  
  // satisfy AdaptableUnaryFunction
  typedef typename thrust::iterator_value<Iterator1>::type      result_type;
  typedef typename thrust::iterator_difference<Iterator1>::type argument_type;

  __host__ __device__
  result_type operator()(argument_type k)
  {
    typedef typename thrust::iterator_value<Iterator1>::type value_type;
    return thrust::detail::device::generic::scalar::select(first1, last1, first2, last2, k, comp);
  }
}; // end select_functor

template<typename Iterator1, typename Iterator2, typename Compare>
  class merge_iterator
{
  typedef thrust::counting_iterator<typename thrust::iterator_difference<Iterator1>::type> counting_iterator;
  typedef select_functor<Iterator1,Iterator2,Compare> function;

  public:
    typedef thrust::transform_iterator<function, counting_iterator> type;
}; // end merge_iterator

template<typename Iterator1, typename Iterator2, typename Compare>
  typename merge_iterator<Iterator1,Iterator2,Compare>::type
    make_merge_iterator(Iterator1 first1, Iterator1 last1,
                        Iterator2 first2, Iterator2 last2,
                        Compare comp)
{
  typedef typename thrust::iterator_difference<Iterator1>::type difference;
  difference zero = 0;

  select_functor<Iterator1,Iterator2,Compare> f(first1,last1,first2,last2,comp);
  return thrust::make_transform_iterator(thrust::make_counting_iterator<difference>(zero),
                                         f);
} // end make_merge_iterator()

template<typename Integer>
  class leapfrog_iterator
{
  typedef thrust::counting_iterator<Integer> counter;

  public:
    typedef thrust::transform_iterator<mult_by<Integer>, counter> type;
}; // end leapfrog_iterator

template<typename Integer>
  typename leapfrog_iterator<Integer>::type
    make_leapfrog_iterator(Integer init, Integer leap_size)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator<Integer>(init),
                                         mult_by<Integer>(leap_size));
} // end make_leapfrog_iterator()


template<typename RandomAccessIterator>
  class splitter_iterator
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference;
  typedef typename leapfrog_iterator<difference>::type leapfrog_iterator;

  public:
    typedef thrust::permutation_iterator<RandomAccessIterator, leapfrog_iterator> type;
}; // end splitter_iterator

template<typename RandomAccessIterator, typename Size>
  typename splitter_iterator<RandomAccessIterator>::type
    make_splitter_iterator(RandomAccessIterator iter, Size split_size)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference;
  return thrust::make_permutation_iterator(iter, make_leapfrog_iterator<difference>(0, split_size));
} // end make_splitter_iterator()


template<typename Compare>
  struct strong_compare
{
  strong_compare(Compare c)
    : comp(c) {}

  // T1 and T2 are tuples
  template<typename T1, typename T2>
  __host__ __device__
  bool operator()(T1 lhs, T2 rhs)
  {
    if(comp(lhs.get<0>(), rhs.get<0>()))
    {
      return true;
    }

    return lhs.get<1>() < rhs.get<1>();
  }

  Compare comp;
};

} // end namespace merge_detail


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename Compare>
RandomAccessIterator3 merge(RandomAccessIterator1 first1,
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
  else if(num_elements2 == 0)
    return thrust::copy(first1, last1, result);
  else if(num_elements1 == 0)
    return thrust::copy(first2, last2, result);

  using namespace merge_detail;
  using namespace thrust::detail;

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  
  // XXX vary block_size dynamically
  const size_t block_size = 512;
  const size_t partition_size = block_size;
  
  const difference1 num_partitions = ceil_div(num_elements1, partition_size);
  const difference1 num_splitters_from_each_range  = num_partitions - 1;

  // zip up the ranges with a counter to disambiguate repeated elements during rank-finding
  typedef thrust::tuple<RandomAccessIterator1,thrust::counting_iterator<difference1> > iterator_tuple1;
  typedef thrust::tuple<RandomAccessIterator2,thrust::counting_iterator<difference2> > iterator_tuple2;
  typedef thrust::zip_iterator<iterator_tuple1> iterator_and_counter1;
  typedef thrust::zip_iterator<iterator_tuple2> iterator_and_counter2;

  iterator_and_counter1 first_and_counter1 =
    thrust::make_zip_iterator(thrust::make_tuple(first1, thrust::make_counting_iterator<difference1>(0)));
  iterator_and_counter1 last_and_counter1 = first_and_counter1 + num_elements1;

  // make the second range begin counting at num_elements1 so they sort after elements from the first range when ambiguous
  iterator_and_counter2 first_and_counter2 =
    thrust::make_zip_iterator(thrust::make_tuple(first2, thrust::make_counting_iterator<difference2>(num_elements1)));
  iterator_and_counter2 last_and_counter2 = first_and_counter2 + num_elements2;

  // create the range [first1[partition_size], first1[2*partition_size], first1[3*partition_size], ...]
  typedef typename splitter_iterator<iterator_and_counter1>::type splitter_iterator1;

  // we +1 to begin at first1[partition_size] instead of first1[0]
  splitter_iterator1 splitters1_begin = make_splitter_iterator(first_and_counter1, partition_size) + 1;
  splitter_iterator1 splitters1_end = splitters1_begin + num_splitters_from_each_range;

  // create the range [first2[partition_size], first2[2*partition_size], first2[3*partition_size], ...]
  typedef typename splitter_iterator<iterator_and_counter1>::type splitter_iterator2;

  // we +1 to begin at first2[partition_size] instead of first1[0]
  splitter_iterator2 splitters2_begin = make_splitter_iterator(first_and_counter2, partition_size) + 1;
  splitter_iterator2 splitters2_end = splitters2_begin + num_splitters_from_each_range;

  typedef compare_first_less_second<Compare> splitter_compare;

  typedef typename merge_detail::merge_iterator<splitter_iterator1,splitter_iterator2,splitter_compare>::type merge_iterator;

  // "merge" the splitters
  merge_iterator splitters_begin = merge_detail::make_merge_iterator(splitters1_begin, splitters1_end,
                                                                     splitters2_begin, splitters2_end,
                                                                     splitter_compare(comp));
  merge_iterator splitters_end   = splitters_begin + 2 * num_splitters_from_each_range;

  size_t num_merged_partitions = 2 * num_splitters_from_each_range + 1;

  raw_buffer<difference1, cuda_device_space_tag> splitter_ranks1(splitters_end - splitters_begin);
  raw_buffer<difference2, cuda_device_space_tag> splitter_ranks2(splitters_end - splitters_begin);

  // find the rank of each splitter in the other range
  // XXX it's possible to fuse rank-finding with the merge_kernel below
  //     this eliminates the temporary buffers splitter_ranks1 & splitter_ranks2
  //     but this spills to lmem and causes a 10x speeddown
  thrust::lower_bound(first_and_counter2, last_and_counter2,
                      splitters_begin, splitters_end, 
                      splitter_ranks2.begin(), strong_compare<Compare>(comp));

  thrust::upper_bound(first_and_counter1, last_and_counter1,
                      splitters_begin, splitters_end,
                      splitter_ranks1.begin(), strong_compare<Compare>(comp));

  // maximize the number of blocks we can launch
  size_t num_blocks = thrust::min(num_merged_partitions, 64000u);

  merge_detail::merge_kernel<block_size><<<num_blocks, (unsigned int) block_size >>>( 
  	first1, last1,
        first2, last2,
  	splitter_ranks1.begin(),
  	splitter_ranks2.begin(),
  	result, 
  	comp,
        num_merged_partitions);
  synchronize_if_enabled("merge_kernel");

  return result + num_elements1 + num_elements2;
} // end merge

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

