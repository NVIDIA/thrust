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

#include <thrust/iterator/iterator_traits.h>

#include <thrust/system/detail/internal/scalar/copy.h>
#include <thrust/detail/backend/dereference.h>

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
  using namespace thrust::detail;

  while(first1 != last1 && first2 != last2)
  {
    if(comp(backend::dereference(first2), backend::dereference(first1)))
    {
      backend::dereference(result) = backend::dereference(first2);
      ++first2;
    } // end if
    else
    {
      backend::dereference(result) = backend::dereference(first1);
      ++first1;
    } // end else

    ++result;
  } // end while

  return thrust::system::detail::internal::scalar::copy(first2, last2, thrust::system::detail::internal::scalar::copy(first1, last1, result));
} // end merge()


template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 first2,
                 InputIterator2 last2,
                 InputIterator3 first3,
                 InputIterator4 first4,
                 OutputIterator1 output1,
                 OutputIterator2 output2,
                 StrictWeakOrdering comp)
{
  using namespace thrust::detail;

  while(first1 != last1 && first2 != last2)
  {
    if(!comp(backend::dereference(first2), backend::dereference(first1)))
    {
      // *first1 <= *first2
      backend::dereference(output1) = backend::dereference(first1);
      backend::dereference(output2) = backend::dereference(first3);
      ++first1;
      ++first3;
    }
    else
    {
      // *first1 > first2
      backend::dereference(output1) = backend::dereference(first2);
      backend::dereference(output2) = backend::dereference(first4);
      ++first2;
      ++first4;
    }

    ++output1;
    ++output2;
  }

  while(first1 != last1)
  {
    backend::dereference(output1) = backend::dereference(first1);
    backend::dereference(output2) = backend::dereference(first3);
    ++first1;
    ++first3;
    ++output1;
    ++output2;
  }

  while(first2 != last2)
  {
    backend::dereference(output1) = backend::dereference(first2);
    backend::dereference(output2) = backend::dereference(first4);
    ++first2;
    ++first4;
    ++output1;
    ++output2;
  }

  return thrust::make_pair(output1, output2);
}

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

