/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

#include <thrust/detail/backend/dereference.h>
#include <thrust/detail/copy.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cpp
{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator merge(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result,
                       StrictWeakOrdering comp)
{
  while(first1 != last1 && first2 != last2)
  {
    if(comp(thrust::detail::backend::dereference(first2), 
            thrust::detail::backend::dereference(first1)))
    {
      thrust::detail::backend::dereference(result) =
        thrust::detail::backend::dereference(first2);

      ++first2;
    } // end if
    else
    {
      thrust::detail::backend::dereference(result) =
        thrust::detail::backend::dereference(first1);

      ++first1;
    } // end else

    ++result;
  } // end while

  return thrust::copy(first2,last2, thrust::copy(first1, last1, result));
} // end merge()

} // end cpp
} // end backend
} // end detail
} // end thrust

