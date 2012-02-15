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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/dereference.h>
#include <thrust/detail/minmax.h>

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

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Size,
         typename Compare>
  typename thrust::iterator_value<RandomAccessIterator1>::type
  __host__ __device__
    select(RandomAccessIterator1 first1,
           RandomAccessIterator1 last1,
           RandomAccessIterator2 first2,
           RandomAccessIterator2 last2,
           Size k,
           Compare comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference;

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  while (first1 != last1 && first2 != last2)
  {
    // shrink the ranges (if possible)
    if (last1 - first1 > k + 1)
      last1 = first1 + (k + 1);
    if (last2 - first2 > k + 1)
      last2 = first2 + (k + 1);
    if (last1 - first1 < k)
    {
      first2 += k - (last1 - first1);
      k = last1 - first1;
    }
    if (last2 - first2 < k)
    {
      first1 += k - (last2 - first2);
      k = last2 - first2;
    }

    if (first1 == last1) break;
    if (first2 == last2) break;

    Size i = k / 2;

    RandomAccessIterator1 mid1 = first1 + i;
    RandomAccessIterator2 mid2 = first2 + i;
    
    value_type1 a = dereference(mid1);
    value_type1 b = dereference(mid2);

    if (k == 0)
      return thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION (a, b, comp);

    // ensure forward progress
    if (k % 2)
    {
      ++i;
      ++mid1;
      ++mid2;
    }

    if (comp(b,a))
    {
      last1  = mid1;
      first2 = mid2;
    }
    else
    {
      first1 = mid1;
      last2  = mid2;
    }
      
    k -= i;
  }

  // handle trivial problems
  if(first1 == last1)
  {
    first2 += k;  return dereference(first2);
  } 
  else
  {
    first1 += k;  return dereference(first1);
  }
}


} // end scalar
} // end generic
} // end backend
} // end detail
} // end thrust

