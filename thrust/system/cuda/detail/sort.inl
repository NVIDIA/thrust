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


/*! \file sort.inl
 *  \brief Inline file for sort.h
 */

#include <thrust/system/cuda/detail/detail/stable_merge_sort.h>
#include <thrust/system/cuda/detail/detail/stable_primitive_sort.h>

#include <thrust/reverse.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/detail/trivial_sequence.h>
#include <thrust/detail/copy.h>
#include <thrust/detail/seq.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/bulk.h>


/*
 *  This file implements the following dispatch procedure for cuda::stable_sort()
 *  and cuda::stable_sort_by_key(). The first level inspects the KeyType
 *  and StrictWeakOrdering to determine whether a sort assuming primitive-typed
 *  data may be applied.
 *
 *  If a sort assuming primitive-typed data can be applied (i.e., a radix sort),
 *  the input ranges are first trivialized (turned into simple contiguous ranges
 *  if they are not already). To implement descending orderings, an ascending
 *  sort will be reversed.
 *
 *  If a sort assuming primitive-typed data cannot be applied, a comparison-based
 *  sort is used. Depending on the size of the key and value types, one level of
 *  indirection may be applied to their input ranges. This transformation
 *  may be applied to either range to convert an ill-suited problem (i.e. sorting with
 *  large keys or large value) into a problem more amenable to the underlying
 *  merge sort algorithm.
 */


namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace stable_sort_detail
{


template<typename KeyType, typename StrictWeakCompare>
  struct can_use_primitive_sort
    : thrust::detail::and_<
        thrust::detail::is_arithmetic<KeyType>,
        thrust::detail::or_<
          thrust::detail::is_same<StrictWeakCompare,thrust::less<KeyType> >,
          thrust::detail::is_same<StrictWeakCompare,thrust::greater<KeyType> >
        >
      >
{};


template<typename RandomAccessIterator, typename StrictWeakCompare>
  struct enable_if_primitive_sort
    : thrust::detail::enable_if<
        can_use_primitive_sort<
          typename iterator_value<RandomAccessIterator>::type,
          StrictWeakCompare
        >::value
      >
{};


template<typename RandomAccessIterator, typename StrictWeakCompare>
  struct enable_if_comparison_sort
    : thrust::detail::disable_if<
        can_use_primitive_sort<
          typename iterator_value<RandomAccessIterator>::type,
          StrictWeakCompare
        >::value
      >
{};


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
typename enable_if_primitive_sort<RandomAccessIterator,StrictWeakOrdering>::type
  stable_sort(execution_policy<DerivedPolicy> &exec,
              RandomAccessIterator first,
              RandomAccessIterator last,
              StrictWeakOrdering comp)
{
  // ensure sequence has trivial iterators
  thrust::detail::trivial_sequence<RandomAccessIterator,DerivedPolicy> keys(exec, first, last);

  thrust::system::cuda::detail::detail::stable_primitive_sort(exec, keys.begin(), keys.end(), comp);
  
  // copy results back, if necessary
  if(!thrust::detail::is_trivial_iterator<RandomAccessIterator>::value)
  {
    thrust::copy(exec, keys.begin(), keys.end(), first);
  }
}


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
typename enable_if_comparison_sort<RandomAccessIterator,StrictWeakOrdering>::type
  stable_sort(execution_policy<DerivedPolicy> &exec,
              RandomAccessIterator first,
              RandomAccessIterator last,
              StrictWeakOrdering comp)
{
  thrust::system::cuda::detail::detail::stable_merge_sort(exec, first, last, comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
typename enable_if_primitive_sort<RandomAccessIterator1,StrictWeakOrdering>::type
  stable_sort_by_key(execution_policy<DerivedPolicy> &exec,
                     RandomAccessIterator1 keys_first,
                     RandomAccessIterator1 keys_last,
                     RandomAccessIterator2 values_first,
                     StrictWeakOrdering comp)
{
  // ensure sequences have trivial iterators
  thrust::detail::trivial_sequence<RandomAccessIterator1,DerivedPolicy> keys(exec, keys_first, keys_last);
  thrust::detail::trivial_sequence<RandomAccessIterator2,DerivedPolicy> values(exec, values_first, values_first + (keys_last - keys_first));
  
  thrust::system::cuda::detail::detail::stable_primitive_sort_by_key(exec, keys.begin(), keys.end(), values.begin(), comp);
  
  // copy results back, if necessary
  if(!thrust::detail::is_trivial_iterator<RandomAccessIterator1>::value)
  {
    thrust::copy(exec, keys.begin(), keys.end(), keys_first);
  }

  if(!thrust::detail::is_trivial_iterator<RandomAccessIterator2>::value)
  {
    thrust::copy(exec, values.begin(), values.end(), values_first);
  }
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
typename enable_if_comparison_sort<RandomAccessIterator1,StrictWeakOrdering>::type
  stable_sort_by_key(execution_policy<DerivedPolicy> &exec,
                     RandomAccessIterator1 keys_first,
                     RandomAccessIterator1 keys_last,
                     RandomAccessIterator2 values_first,
                     StrictWeakOrdering comp)
{
  thrust::system::cuda::detail::detail::stable_merge_sort_by_key(exec, keys_first, keys_last, values_first, comp);
}


} // end namespace stable_sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort(execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<RandomAccessIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  struct workaround
  {
    __host__ __device__
    static void parallel_path(execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator first,
                              RandomAccessIterator last,
                              StrictWeakOrdering comp)
    {
      stable_sort_detail::stable_sort(exec, first, last, comp);
    }

    __host__ __device__
    static void sequential_path(RandomAccessIterator first,
                                RandomAccessIterator last,
                                StrictWeakOrdering comp)
    {
      thrust::sort(thrust::seq, first, last, comp);
    }
  };

#if __BULK_HAS_CUDART__
  workaround::parallel_path(exec, first, last, comp);
#else
  workaround::sequential_path(first, last, comp);
#endif
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort_by_key(execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator1 keys_first,
                        RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first,
                        StrictWeakOrdering comp)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<RandomAccessIterator1, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  struct workaround
  {
    __host__ __device__
    static void parallel_path(execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 keys_first,
                              RandomAccessIterator1 keys_last,
                              RandomAccessIterator2 values_first,
                              StrictWeakOrdering comp)
    {
      stable_sort_detail::stable_sort_by_key(exec, keys_first, keys_last, values_first, comp);
    }

    __host__ __device__
    static void sequential_path(RandomAccessIterator1 keys_first,
                                RandomAccessIterator1 keys_last,
                                RandomAccessIterator2 values_first,
                                StrictWeakOrdering comp)
    {
      thrust::stable_sort_by_key(thrust::seq, keys_first, keys_last, values_first, comp);
    }
  };
  
#if __BULK_HAS_CUDART__
  workaround::parallel_path(exec, keys_first, keys_last, values_first, comp);
#else
  workaround::sequential_path(keys_first, keys_last, values_first, comp);
#endif
}


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

