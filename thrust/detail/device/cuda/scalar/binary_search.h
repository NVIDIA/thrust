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

#pragma once

#include <thrust/detail/config.h>

// do not attempt to compile this file with anything other than nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/pair.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/iterator/iterator_traits.h>

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

// XXX generalize these upon implementation of scalar::distance & scalar::advance

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__device__
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
__device__
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
__device__
  pair<RandomAccessIterator,RandomAccessIterator>
    equal_range(RandomAccessIterator first, RandomAccessIterator last,
                const T &val,
                BinaryPredicate comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference_type;

  // XXX this should read difference_type len = distance(first,last);
  difference_type len = last - first;

  difference_type half;
  RandomAccessIterator middle, left, right;

  while(len > 0)
  {
    half = len >> 1;
    middle = first;

    // XXX this should read advance(middle,half);
    middle += half;

    if(comp(dereference(middle), val))
    {
      first = middle;
      ++first;
      len = len - half - 1;
    }
    else if(comp(val, dereference(middle)))
    {
      len = half;
    }
    else
    {
      left = scalar::lower_bound(first, middle, val, comp);
      // XXX this should read advance(first, len);
      first += len;
      right = scalar::upper_bound(++middle, first, val, comp);
      return thrust::make_pair(left, right);
    }
  }

  return thrust::make_pair(first,first);
}

} // end scalar

} // end cuda

} // end device

} // end detail

} // end thrust

#endif // THRUST_DEVICE_COMPILER_NVCC

