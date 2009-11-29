/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>

#include <thrust/sorting/detail/device/cuda/stable_merge_sort.h>
#include <thrust/sorting/detail/device/cuda/stable_radix_sort.h>

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

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp,
                   thrust::detail::true_type)
{
    // CUDA path for thrust::stable_sort with primitive keys
    // (e.g. int, float, short, etc.) and the default less<T> comparison
    // method is implemented with stable_radix_sort_by_key
    thrust::sorting::detail::device::cuda::stable_radix_sort(first, last);
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp,
                   thrust::detail::false_type)
{
    // device path for thrust::stable_sort with general keys 
    // and comparison methods is implemented with stable_merge_sort
    thrust::sorting::detail::device::cuda::stable_merge_sort(first, last, comp);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp,
                          thrust::detail::true_type)
{
    // device path for thrust::stable_sort_by_key with primitive keys
    // (e.g. int, float, short, etc.) and the default less<T> comparison
    // method is implemented with stable_radix_sort_by_key
    thrust::sorting::detail::device::cuda::stable_radix_sort_by_key(keys_first, keys_last, values_first);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp,
                          thrust::detail::false_type)
{
    // device path for thrust::stable_sort with general keys 
    // and comparison methods is implemented with stable_merge_sort
    thrust::sorting::detail::device::cuda::stable_merge_sort_by_key(keys_first, keys_last, values_first, comp);
}

} // end namespace detail


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
    // dispatch on whether we can use radix_sort
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
    static const bool use_radix_sort = thrust::detail::is_pod<KeyType>::value &&
                                       thrust::detail::is_same<StrictWeakOrdering, typename thrust::less<KeyType> >::value;

    detail::stable_sort(first, last, comp,
            thrust::detail::integral_constant<bool, use_radix_sort>());
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
    // dispatch on whether we can use radix_sort
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
    static const bool use_radix_sort = thrust::detail::is_pod<KeyType>::value &&
                                       thrust::detail::is_same<StrictWeakOrdering, typename thrust::less<KeyType> >::value;
    
    detail::stable_sort_by_key(keys_first, keys_last, values_first, comp,
            thrust::detail::integral_constant<bool, use_radix_sort>());
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

