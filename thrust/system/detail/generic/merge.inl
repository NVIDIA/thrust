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
#include <thrust/detail/static_assert.h>
#include <thrust/system/detail/generic/merge.h>
#include <thrust/merge.h>
#include <thrust/functional.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator merge(tag,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result,
                       StrictWeakOrdering comp)
{
  // unimplemented
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator1, false>::value) );
} // end merge()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator merge(tag,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result)
{
  typedef typename thrust::iterator_value<InputIterator1>::type value_type;
  return thrust::merge(first1,last1,first2,last2,result,thrust::less<value_type>());
} // end merge()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

