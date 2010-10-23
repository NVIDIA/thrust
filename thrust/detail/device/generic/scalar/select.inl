/*
 *  Copyright 2008-2010 NVIDIA Corporation
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
#include <thrust/detail/device/generic/scalar/select.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/extrema.h>

namespace thrust
{
namespace detail
{
namespace device
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
  // check for trivial problem
  if(first1 == last1)
  {
    first2 += k;
    return dereference(first2);
  }

  if(first2 == last2)
  {
    first1 += k;
    return dereference(first1);
  }

  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference;

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  difference size1 = last1 - first1;
  difference size2 = last2 - first2;

  while(k > 1 && size1 > 1 && size2 > 1 && k + 1 < size1 + size2)
  {
    // subdivide the problem
    Size i = k / 2;

    // did we ask for too much from A?
    if(k/2 >= size1)
    {
      i = size1 - 1;
    } // end if
    else if(k - 1 >= size2)
    {
      i = k - (size2 - 1);
    } // end else if

    Size j = k - i;

    RandomAccessIterator1 a = first1;
    a += i;

    RandomAccessIterator2 b = first2;
    b += j;

    if(comp(dereference(b),dereference(a)))
    {
      first2 = b;
      k -= j;
      size2 -= j;
    } // end if
    else
    {
      first1 = a;
      k -= i;
      size1 -= i;
    } // end else
  } // end while

  // XXX consider just doing a serial merge here
  //     to simplify this code as in case 3 below

  if(k == 0)
  {
    // case 0
    // return min of a,b (a wins if equivalent)
    return thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION (dereference(first1), dereference(first2), comp);
  } // end if
  else if(size1 == 1)
  {
    // case 1
    // b wins if equivalent
    //return std::max(*(first2 + k - 1), a);
    first2 += (k - 1);
  } // end else if
  else if(size2 == 1)
  {
    // case 2
    // b wins if equivalent
    //return std::max(b, *(first1 + k - 1));
    first1 += k - 1;
  } // end else if
  else if(k == 1)
  {
    // case 3
    // return the 2nd element of the merger of
    // [A[0], A[1]] and [B[0], B[1]]
    if(comp(dereference(first2), dereference(first1)))
    {
      ++first2;
    } // end if
    else
    {
      ++first1;
    } // end else

    return thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION (dereference(first1), dereference(first2), comp);
  } // end else if
  else if(k + 1 == size1 + size2)
  {
    // case 4
    // return max(B[-1], A[-1])
    first1 = last1;
    --first1;

    first2 = last2;
    --first2;
  } // end else if

  return thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION (dereference(first2),dereference(first1),comp);
} // end select()


} // end scalar
} // end generic
} // end device
} // end detail
} // end thrust

