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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/execution_policy.hpp>


BULK_NAMESPACE_PREFIX
namespace bulk
{


template<typename ExecutionGroup,
         typename RandomAccessIterator,
         typename Size,
         typename Function>
__device__
RandomAccessIterator for_each_n(ExecutionGroup &g, RandomAccessIterator first, Size n, Function f)
{
  for(Size i = g.this_thread.index();
      i < n;
      i += g.size())
  {
    f(first[i]);
  } // end for i

  g.wait();

  return first + n;
} // end for_each()


template<std::size_t bound,
         std::size_t grainsize,
         typename RandomAccessIterator,
         typename Size,
         typename Function>
__device__
RandomAccessIterator for_each_n(bounded<bound, bulk::agent<grainsize> > &b,
                                RandomAccessIterator first,
                                Size n,
                                Function f)
{
  typedef typename bounded<bound, bulk::agent<grainsize> >::size_type size_type;

  for(size_type i = 0; i < bound; ++i)
  {
    if(i < n)
    {
      f(first[i]);
    } // end if
  } // end for i

  return first + n;
} // end for_each_n()
                                

} // end bulk
BULK_NAMESPACE_SUFFIX

