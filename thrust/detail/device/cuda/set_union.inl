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
#include <thrust/pair.h>
#include <thrust/detail/device/cuda/block/set_union.h>
#include <thrust/detail/device/cuda/detail/split_for_set_operation.h>
#include <thrust/detail/device/cuda/detail/set_operation.h>

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
                               detail::split_for_set_operation(),
                               set_union_detail::block_convergent_set_union_functor());
} // end set_union

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

