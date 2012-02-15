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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/pair.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/backend/copy.h>
#include <thrust/detail/backend/cuda/block/set_symmetric_difference.h>
#include <thrust/detail/backend/cuda/detail/split_for_set_operation.h>
#include <thrust/detail/backend/cuda/detail/set_operation.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{

namespace set_symmetric_difference_detail
{

struct block_convergent_set_symmetric_difference_functor
{
  __host__ __device__ __forceinline__
  static size_t get_min_size_of_result_in_number_of_elements(size_t size_of_range1,
                                                             size_t size_of_range2)
  {
    // set_symmetric_difference could result in zero output
    return 0u;
  }

  __host__ __device__ __forceinline__
  static size_t get_max_size_of_result_in_number_of_elements(size_t size_of_range1,
                                                             size_t size_of_range2)
  {
    // set_intersection could output all of range1 and range2
    return size_of_range1 + size_of_range2;
  }

  __host__ __device__ __forceinline__
  static unsigned int get_temporary_array_size_in_number_of_bytes(unsigned int block_size)
  {
    // set_symmetric_difference needs temporary arrays for both inputs range1 and range2
    return 2 * block_size * sizeof(int);
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
    return block::set_symmetric_difference(first1,last1,first2,last2,reinterpret_cast<int*>(temporary),result,comp);
  } // end operator()()
}; // end block_convergent_set_symmetric_difference_functor


} // end namespace set_symmetric_difference_detail


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename Compare>
RandomAccessIterator3 set_symmetric_difference(RandomAccessIterator1 first1,
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
  if(num_elements1 == 0)
    return thrust::detail::backend::copy(first2, last2, result);
  else if (num_elements2 == 0)
    return thrust::detail::backend::copy(first1, last1, result);

  return detail::set_operation(first1, last1,
                               first2, last2,
                               result,
                               comp,
                               detail::split_for_set_operation(),
                               set_symmetric_difference_detail::block_convergent_set_symmetric_difference_functor());
} // end set_symmetric_difference

} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

