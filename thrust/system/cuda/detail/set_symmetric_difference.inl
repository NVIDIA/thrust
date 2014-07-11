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

#include <thrust/detail/config.h>
#include <thrust/set_operations.h>
#include <thrust/detail/cstdint.h>
#include <thrust/system/cuda/detail/detail/set_operation.h>
#include <thrust/system/cuda/detail/bulk.h>
#include <thrust/detail/seq.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace set_symmetric_difference_detail
{


struct serial_bounded_set_symmetric_difference
{
  // max_input_size <= 32
  template<typename Size, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  inline __device__
    thrust::detail::uint32_t operator()(Size max_input_size,
                                        InputIterator1 first1, InputIterator1 last1,
                                        InputIterator2 first2, InputIterator2 last2,
                                        OutputIterator result,
                                        Compare comp)
  {
    thrust::detail::uint32_t active_mask = 0;
    thrust::detail::uint32_t active_bit = 1;
  
    while(first1 != last1 && first2 != last2)
    {
      if(comp(*first1,*first2))
      {
        *result = *first1;
        active_mask |= active_bit;
        ++first1;
      } // end if
      else if(comp(*first2,*first1))
      {
        *result = *first2;
        active_mask |= active_bit;
        ++first2;
      } // end else if
      else
      {
        ++first1;
        ++first2;
      } // end else
  
      ++result;
      active_bit <<= 1;
    } // end while

    while(first1 != last1)
    {
      *result = *first1;
      ++first1;
      ++result;
      active_mask |= active_bit;
      active_bit <<= 1;
    }

    while(first2 != last2)
    {
      *result = *first2;
      ++first2;
      ++result;
      active_mask |= active_bit;
      active_bit <<= 1;
    }
  
    return active_mask;
  }


  template<typename Size, typename InputIterator1, typename InputIterator2, typename Compare>
  inline __device__
    Size count(Size max_input_size,
               InputIterator1 first1, InputIterator1 last1,
               InputIterator2 first2, InputIterator2 last2,
               Compare comp)
  {
    Size result = 0;
  
    while(first1 != last1 && first2 != last2)
    {
      if(comp(*first1,*first2))
      {
        ++first1;
        ++result;
      } // end if
      else if(comp(*first2,*first1))
      {
        ++first2;
        ++result;
      } // end else if
      else
      {
        ++first1;
        ++first2;
      } // end else
    } // end while
  
    return result + thrust::max(last1 - first1,last2 - first2);
  }
}; // end serial_bounded_set_symmetric_difference


} // end namespace set_symmetric_difference_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename Compare>
__host__ __device__
RandomAccessIterator3 set_symmetric_difference(execution_policy<DerivedPolicy> &exec,
                                               RandomAccessIterator1 first1,
                                               RandomAccessIterator1 last1,
                                               RandomAccessIterator2 first2,
                                               RandomAccessIterator2 last2,
                                               RandomAccessIterator3 result,
                                               Compare comp)
{
  struct workaround
  {
    __host__ __device__
    static RandomAccessIterator3 parallel_path(execution_policy<DerivedPolicy> &exec,
                                               RandomAccessIterator1 first1,
                                               RandomAccessIterator1 last1,
                                               RandomAccessIterator2 first2,
                                               RandomAccessIterator2 last2,
                                               RandomAccessIterator3 result,
                                               Compare comp)
    {
      return thrust::system::cuda::detail::detail::set_operation(exec, first1, last1, first2, last2, result, comp, set_symmetric_difference_detail::serial_bounded_set_symmetric_difference());
    }

    __host__ __device__
    static RandomAccessIterator3 sequential_path(execution_policy<DerivedPolicy> &,
                                                 RandomAccessIterator1 first1,
                                                 RandomAccessIterator1 last1,
                                                 RandomAccessIterator2 first2,
                                                 RandomAccessIterator2 last2,
                                                 RandomAccessIterator3 result,
                                                 Compare comp)
    {
      return thrust::set_symmetric_difference(thrust::seq, first1, last1, first2, last2, result, comp);
    }
  };

#if __BULK_HAS_CUDART__
  return workaround::parallel_path(exec, first1, last1, first2, last2, result, comp);
#else
  return workaround::sequential_path(exec, first1, last1, first2, last2, result, comp);
#endif
} // end set_symmetric_difference


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

