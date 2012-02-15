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
#include <thrust/pair.h>
#include <thrust/detail/backend/dereference.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

namespace backend
{

namespace generic
{

namespace scalar
{

// XXX generalize these upon implementation of scalar::distance & scalar::advance

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator lower_bound(RandomAccessIterator first, RandomAccessIterator last,
                                 const T &val,
                                 BinaryPredicate comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference_type;

  // XXX should read len = distance(first,last)
  difference_type len = last - first;

  while(len > 0)
  {
    difference_type half = len >> 1;
    RandomAccessIterator middle = first;

    // XXX should read advance(middle,half)
    middle += half;

    if(comp(dereference(middle), val))
    {
      first = middle;
      ++first;
      len = len - half - 1;
    }
    else
    {
      len = half;
    }
  }

  return first;
}

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator upper_bound(RandomAccessIterator first, RandomAccessIterator last,
                                 const T &val,
                                 BinaryPredicate comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference_type;

  // XXX should read len = distance(first,last)
  difference_type len = last - first;

  while(len > 0)
  {
    difference_type half = len >> 1;
    RandomAccessIterator middle = first;

    // XXX should read advance(middle,half)
    middle += half;

    if(comp(val, dereference(middle)))
    {
      len = half;
    }
    else
    {
      first = middle;
      ++first;
      len = len - half - 1;
    }
  }

  return first;
}

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
  pair<RandomAccessIterator,RandomAccessIterator>
    equal_range(RandomAccessIterator first, RandomAccessIterator last,
                const T &val,
                BinaryPredicate comp)
{
  RandomAccessIterator lb = thrust::detail::backend::generic::scalar::lower_bound(first, last, val, comp);
  return thrust::make_pair(lb, thrust::detail::backend::generic::scalar::upper_bound(lb, last, val, comp));
}


template<typename RandomAccessIterator, typename T, typename Compare>
__host__ __device__
bool binary_search(RandomAccessIterator first, RandomAccessIterator last, const T &value, Compare comp)
{
  RandomAccessIterator iter = thrust::detail::backend::generic::scalar::lower_bound(first,last,value,comp);
  return iter != last && !comp(value, *iter);
}

} // end scalar

} // end generic

} // end backend

} // end detail

} // end thrust

#include <thrust/detail/backend/generic/scalar/binary_search.inl>

