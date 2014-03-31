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
#include <thrust/iterator/iterator_traits.h>

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


// sequential merge for when we have a static bound on the size of the result
template<unsigned int result_size_bound, typename Iterator1, typename Iterator2, typename Iterator3, typename Compare>
__device__
void sequential_bounded_merge(Iterator1 first1, Iterator1 last1,
                              Iterator2 first2, Iterator2 last2,
                              Iterator3 result,
                              Compare comp)
{ 
  // XXX nvcc generates the wrong code for the path below for sm_1x
  //     so use this (slower) but equivalent implementation which does not prefetch
#if __CUDA_ARCH__ < 200
  for(unsigned int i = 0; i < result_size_bound; ++i, ++result)
  {
    bool p = (first2 >= last2) || ((first1 < last1) && !comp(*first2, *first1));
    
    *result = p ? *first1 : *first2;
    
    if(p)
    {
      ++first1;
    }
    else
    {
      ++first2;
    }
  }
#else
  typename thrust::iterator_value<Iterator1>::type aKey = *first1;
  typename thrust::iterator_value<Iterator2>::type bKey = *first2;
  
  for(unsigned int i = 0; i < result_size_bound; ++i, ++result)
  {
    bool p = (first2 >= last2) || ((first1 < last1) && !comp(bKey, aKey));
    
    *result = p ? aKey : bKey;
    
    if(p)
    {
      ++first1;
      aKey = *first1;
    }
    else
    {
      ++first2;
      bKey = *first2;
    }
  }
#endif
}


template<typename Size, typename Iterator1, typename Iterator2, typename Compare>
__device__
Size merge_path(Size pos, Iterator1 first1, Size n1, Iterator2 first2, Size n2, Compare comp)
{
  Size begin = (pos >= n2) ? (pos - n2) : Size(0);
  Size end = thrust::min<Size>(pos, n1);
  
  while(begin < end)
  {
    Size mid = (begin + end) >> 1;

    if(comp(first2[pos - 1 - mid], first1[mid]))
    {
      end = mid;
    }
    else
    {
      begin = mid + 1;
    }
  }
  return begin;
}


} // end detail
} // end detail
} // end cuda
} // end system
} // end thrust

