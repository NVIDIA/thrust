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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/internal/scalar/stable_primitive_sort.h>
#include <thrust/system/detail/internal/scalar/stable_radix_sort.h>
#include <thrust/functional.h>
#include <thrust/system/detail/internal/scalar/partition.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace internal
{
namespace scalar
{
namespace stable_primitive_sort_detail
{


template<typename Iterator>
  struct enable_if_bool_sort
    : thrust::detail::enable_if<
        thrust::detail::is_same<
          bool,
          typename thrust::iterator_value<Iterator>::type
        >::value
      >
{};


template<typename Iterator>
  struct disable_if_bool_sort
    : thrust::detail::disable_if<
        thrust::detail::is_same<
          bool,
          typename thrust::iterator_value<Iterator>::type
        >::value
      >
{};



template<typename RandomAccessIterator>
  typename enable_if_bool_sort<RandomAccessIterator>::type
    stable_primitive_sort(RandomAccessIterator first, RandomAccessIterator last)
{
  // use stable_partition if we're sorting bool
  // stable_partition puts true values first, so we need to logical_not
  scalar::stable_partition(first, last, thrust::logical_not<bool>());
}


template<typename RandomAccessIterator>
  typename disable_if_bool_sort<RandomAccessIterator>::type
    stable_primitive_sort(RandomAccessIterator first, RandomAccessIterator last)
{
  // call stable_radix_sort
  scalar::stable_radix_sort(first,last);
}


struct logical_not_first
{
  template<typename Tuple>
  __host__ __device__
  bool operator()(Tuple t)
  {
    return !thrust::get<0>(t);
  }
};


template<typename RandomAccessIterator1, typename RandomAccessIterator2>
  typename enable_if_bool_sort<RandomAccessIterator1>::type
    stable_primitive_sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
                                 RandomAccessIterator2 values_first)
{
  // use stable_partition if we're sorting bool
  // stable_partition puts true values first, so we need to logical_not
  scalar::stable_partition(thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)),
                           thrust::make_zip_iterator(thrust::make_tuple(keys_last, values_first)),
                           logical_not_first());
}


template<typename RandomAccessIterator1, typename RandomAccessIterator2>
  typename disable_if_bool_sort<RandomAccessIterator1>::type
    stable_primitive_sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
                                 RandomAccessIterator2 values_first)
{
  // call stable_radix_sort_by_key
  scalar::stable_radix_sort_by_key(keys_first, keys_last, values_first);
}


}

template<typename RandomAccessIterator>
void stable_primitive_sort(RandomAccessIterator first,
                           RandomAccessIterator last)
{
  scalar::stable_primitive_sort_detail::stable_primitive_sort(first,last);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void stable_primitive_sort_by_key(RandomAccessIterator1 keys_first,
                                  RandomAccessIterator1 keys_last,
                                  RandomAccessIterator2 values_first)
{
  scalar::stable_primitive_sort_detail::stable_primitive_sort_by_key(keys_first, keys_last, values_first);
}

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

#include <thrust/system/detail/internal/scalar/stable_primitive_sort.inl>

