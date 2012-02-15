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

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace scalar
{


namespace detail
{

template<typename EuclideanRingElement>
__device__
  EuclideanRingElement gcd(EuclideanRingElement m,
                           EuclideanRingElement n)
{
  while(n != 0)
  {
    EuclideanRingElement t = m % n;
    m = n;
    n = t;
  }

  return m;
} // end gcd()

} // end detail


template<typename T>
__device__
  void rotate(T *first, T *middle, T *last)
{
  if(first == middle || last == middle) return;

  typedef typename thrust::iterator_difference<T*>::type difference;

  const difference n = last - first;
  const difference k = middle - first;
  const difference l = n - k;

//  if(k == l)
//  {
//    swap_ranges(first, middle, middle);
//    return;
//  }

  const difference d = detail::gcd(n,k);

  for(difference i = 0; i < d; ++i)
  {
    T temp = *first;
    T *p = first;

    if(k < l)
    {
      for(difference j = 0; j < l/d; ++j)
      {
        if(p > first + l)
        {
          *p = *(p - l);
          p -= l;
        } // end if

        *p = *(p + k);
        p += k;
      } // end for j
    } // end if
    else
    {
      for(difference j = 0; j < k/d - 1; ++j)
      {
        if(p < last - k)
        {
          *p = *(p + k);
          p += k;
        } // end if

        *p = *(p - l);
        p -= l;
      } // end for j
    } // end else

    *p = temp;
    ++first;
  } // end for i
} // end rotate()

} // end scalar
} // end cuda
} // end backend
} // end detail
} // end thrust

