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


/*! \file sort.inl
 *  \brief Inline file for sort.h
 */

#include <thrust/system/cuda/detail/detail/stable_merge_sort.h>
#include <thrust/system/cuda/detail/detail/stable_primitive_sort.h>

#include <thrust/reverse.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/cuda/detail/tag.h>
#include <thrust/system/cuda/detail/temporary_indirect_permutation.h>
#include <thrust/detail/trivial_sequence.h>


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


template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
  typename enable_if_primitive_sort<RandomAccessIterator,StrictWeakOrdering>::type
    stable_sort(dispatchable<System> &system,
                RandomAccessIterator first,
                RandomAccessIterator last,
                StrictWeakOrdering comp)
{
  // ensure sequence has trivial iterators
  thrust::detail::trivial_sequence<RandomAccessIterator,System> keys(system, first, last);
  
  // CUDA path for thrust::stable_sort with primitive keys
  // (e.g. int, float, short, etc.) and a less<T> or greater<T> comparison
  // method is implemented with a primitive sort
  thrust::system::cuda::detail::detail::stable_primitive_sort(system, keys.begin(), keys.end());
  
  // copy results back, if necessary
  if(!thrust::detail::is_trivial_iterator<RandomAccessIterator>::value)
  {
    thrust::copy(system, keys.begin(), keys.end(), first);
  }
  
  // if comp is greater<T> then reverse the keys
  typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
  const static bool reverse = thrust::detail::is_same<StrictWeakOrdering, typename thrust::greater<KeyType> >::value;
  
  if(reverse)
  {
    thrust::reverse(first, last);
  }
}

template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
  typename enable_if_comparison_sort<RandomAccessIterator,StrictWeakOrdering>::type
    stable_sort(dispatchable<System> &system,
                RandomAccessIterator first,
                RandomAccessIterator last,
                StrictWeakOrdering comp)
{
  // decide whether to sort keys indirectly
  typedef typename thrust::iterator_value<RandomAccessIterator>::type KeyType;
  typedef thrust::detail::integral_constant<bool, (sizeof(KeyType) > 8)> use_key_indirection;
  
  conditional_temporary_indirect_ordering<use_key_indirection, System, RandomAccessIterator, StrictWeakOrdering> potentially_indirect_keys(derived_cast(system), first, last, comp);
  
  thrust::system::cuda::detail::detail::stable_merge_sort(system,
                                                          potentially_indirect_keys.begin(),
                                                          potentially_indirect_keys.end(),
                                                          potentially_indirect_keys.comp());
}

template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  typename enable_if_primitive_sort<RandomAccessIterator1,StrictWeakOrdering>::type
    stable_sort_by_key(dispatchable<System> &system,
                       RandomAccessIterator1 keys_first,
                       RandomAccessIterator1 keys_last,
                       RandomAccessIterator2 values_first,
                       StrictWeakOrdering comp)
{
  // path for thrust::stable_sort_by_key with primitive keys
  // (e.g. int, float, short, etc.) and a less<T> or greater<T> comparison
  // method is implemented with stable_primitive_sort_by_key
  
  // if comp is greater<T> then reverse the keys and values
  typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
  const static bool reverse = thrust::detail::is_same<StrictWeakOrdering, typename thrust::greater<KeyType> >::value;
  
  // note, we also have to reverse the (unordered) input to preserve stability
  if (reverse)
  {
    thrust::reverse(system, keys_first,  keys_last);
    thrust::reverse(system, values_first, values_first + (keys_last - keys_first));
  }
  
  // ensure sequences have trivial iterators
  thrust::detail::trivial_sequence<RandomAccessIterator1,System> keys(system, keys_first, keys_last);
  thrust::detail::trivial_sequence<RandomAccessIterator2,System> values(system, values_first, values_first + (keys_last - keys_first));
  
  thrust::system::cuda::detail::detail::stable_primitive_sort_by_key(system, keys.begin(), keys.end(), values.begin());
  
  // copy results back, if necessary
  if(!thrust::detail::is_trivial_iterator<RandomAccessIterator1>::value)
      thrust::copy(system, keys.begin(), keys.end(), keys_first);
  if(!thrust::detail::is_trivial_iterator<RandomAccessIterator2>::value)
      thrust::copy(system, values.begin(), values.end(), values_first);
  
  if (reverse)
  {
    thrust::reverse(system, keys_first,  keys_last);
    thrust::reverse(system, values_first, values_first + (keys_last - keys_first));
  }
}


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  typename enable_if_comparison_sort<RandomAccessIterator1,StrictWeakOrdering>::type
    stable_sort_by_key(dispatchable<System> &system,
                       RandomAccessIterator1 keys_first,
                       RandomAccessIterator1 keys_last,
                       RandomAccessIterator2 values_first,
                       StrictWeakOrdering comp)
{
  // decide whether to apply indirection to either range
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type ValueType;
  
  typedef thrust::detail::integral_constant<bool, (sizeof(KeyType) > 8)> use_key_indirection;
  typedef thrust::detail::integral_constant<bool, (sizeof(ValueType) > 4)> use_value_indirection;
  
  conditional_temporary_indirect_ordering<
    use_key_indirection,
    System,
    RandomAccessIterator1,
    StrictWeakOrdering
  > potentially_indirect_keys(derived_cast(system), keys_first, keys_last, comp);
  
  conditional_temporary_indirect_permutation<
    use_value_indirection,
    System,
    RandomAccessIterator2
  > potentially_indirect_values(derived_cast(system), values_first, values_first + (keys_last - keys_first));
  
  thrust::system::cuda::detail::detail::stable_merge_sort_by_key(system,
                                                                 potentially_indirect_keys.begin(),
                                                                 potentially_indirect_keys.end(),
                                                                 potentially_indirect_values.begin(),
                                                                 potentially_indirect_keys.comp());
}


} // end namespace stable_sort_detail


template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(dispatchable<System> &system,
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
  
  stable_sort_detail::stable_sort(system, first, last, comp);
}


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(dispatchable<System> &system,
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
  
  stable_sort_detail::stable_sort_by_key(system, keys_first, keys_last, values_first, comp);
}


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

