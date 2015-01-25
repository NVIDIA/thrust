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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/detail/stable_primitive_sort.h>
#include <thrust/system/cuda/detail/detail/stable_radix_sort.h>
#include <thrust/functional.h>
#include <thrust/partition.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
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


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
typename enable_if_bool_sort<RandomAccessIterator>::type
  stable_primitive_sort(execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator first,
                        RandomAccessIterator last,
                        thrust::less<typename thrust::iterator_value<RandomAccessIterator>::type>)
{
  // use stable_partition if we're sorting bool
  // stable_partition puts true values first, so we need to logical_not
  thrust::stable_partition(exec, first, last, thrust::logical_not<bool>());
}


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
typename enable_if_bool_sort<RandomAccessIterator>::type
  stable_primitive_sort(execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator first,
                        RandomAccessIterator last,
                        thrust::greater<typename thrust::iterator_value<RandomAccessIterator>::type>)
{
  // use stable_partition if we're sorting bool
  // stable_partition puts true values first, so we don't need to logical_not
  thrust::stable_partition(exec, first, last, thrust::identity<bool>());
}


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename Compare>
__host__ __device__
typename disable_if_bool_sort<RandomAccessIterator>::type
  stable_primitive_sort(execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator first,
                        RandomAccessIterator last,
                        Compare comp)
{
  // call stable_radix_sort
  thrust::system::cuda::detail::detail::stable_radix_sort(exec,first,last,comp);
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


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
typename enable_if_bool_sort<RandomAccessIterator1>::type
  stable_primitive_sort_by_key(execution_policy<DerivedPolicy> &exec,
                               RandomAccessIterator1 keys_first,
                               RandomAccessIterator1 keys_last,
                               RandomAccessIterator2 values_first,
                               thrust::less<typename thrust::iterator_value<RandomAccessIterator1>::type>)
{
  // use stable_partition if we're sorting bool
  // stable_partition puts true values first, so we need to logical_not
  thrust::stable_partition(exec,
                           thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)),
                           thrust::make_zip_iterator(thrust::make_tuple(keys_last, values_first)),
                           logical_not_first());
}


struct first_tuple_element
{
  template<typename Tuple>
  __host__ __device__
  bool operator()(Tuple t)
  {
    return thrust::get<0>(t);
  }
};


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
typename enable_if_bool_sort<RandomAccessIterator1>::type
  stable_primitive_sort_by_key(execution_policy<DerivedPolicy> &exec,
                               RandomAccessIterator1 keys_first,
                               RandomAccessIterator1 keys_last,
                               RandomAccessIterator2 values_first,
                               thrust::greater<typename thrust::iterator_value<RandomAccessIterator1>::type>)
{
  // use stable_partition if we're sorting bool
  // stable_partition puts true values first, so we need to just return the first tuple element
  // i.e., we don't need to use logical_not_first
  thrust::stable_partition(exec,
                           thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)),
                           thrust::make_zip_iterator(thrust::make_tuple(keys_last, values_first)),
                           first_tuple_element());
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Compare>
__host__ __device__
typename disable_if_bool_sort<RandomAccessIterator1>::type
  stable_primitive_sort_by_key(execution_policy<DerivedPolicy> &exec,
                               RandomAccessIterator1 keys_first,
                               RandomAccessIterator1 keys_last,
                               RandomAccessIterator2 values_first,
                               Compare comp)
{
  // call stable_radix_sort_by_key
  thrust::system::cuda::detail::detail::stable_radix_sort_by_key(exec, keys_first, keys_last, values_first, comp);
}
  

} // end stable_primitive_sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
void stable_primitive_sort(execution_policy<DerivedPolicy> &exec,
                           RandomAccessIterator first,
                           RandomAccessIterator last,
                           thrust::less<typename thrust::iterator_value<RandomAccessIterator>::type> comp)
{
  thrust::system::cuda::detail::detail::stable_primitive_sort_detail::stable_primitive_sort(exec,first,last, comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
void stable_primitive_sort(execution_policy<DerivedPolicy> &exec,
                           RandomAccessIterator first,
                           RandomAccessIterator last,
                           thrust::greater<typename thrust::iterator_value<RandomAccessIterator>::type> comp)
{
  thrust::system::cuda::detail::detail::stable_primitive_sort_detail::stable_primitive_sort(exec,first,last, comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void stable_primitive_sort_by_key(execution_policy<DerivedPolicy> &exec,
                                  RandomAccessIterator1 keys_first,
                                  RandomAccessIterator1 keys_last,
                                  RandomAccessIterator2 values_first,
                                  thrust::less<typename thrust::iterator_value<RandomAccessIterator1>::type> comp)
{
  thrust::system::cuda::detail::detail::stable_primitive_sort_detail::stable_primitive_sort_by_key(exec, keys_first, keys_last, values_first, comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void stable_primitive_sort_by_key(execution_policy<DerivedPolicy> &exec,
                                  RandomAccessIterator1 keys_first,
                                  RandomAccessIterator1 keys_last,
                                  RandomAccessIterator2 values_first,
                                  thrust::greater<typename thrust::iterator_value<RandomAccessIterator1>::type> comp)
{
  thrust::system::cuda::detail::detail::stable_primitive_sort_detail::stable_primitive_sort_by_key(exec, keys_first, keys_last, values_first, comp);
}


} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

