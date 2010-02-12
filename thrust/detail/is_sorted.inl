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


/*! \file is_sorted.inl
 *  \brief Inline file for is_sorted.h
 */

#include <thrust/is_sorted.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/inner_product.h>

namespace thrust
{

template <typename ForwardIterator>
bool is_sorted(ForwardIterator first, ForwardIterator last)
{
  // use less<T> for comp
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type T;
  return thrust::is_sorted(first, last, thrust::less<T>());
} // end is_sorted() 

template <typename ForwardIterator, typename StrictWeakOrdering>
bool is_sorted(ForwardIterator first, ForwardIterator last, StrictWeakOrdering comp)
{
  if(first == last)
    return true;

  // XXX cant create this one, is this a device_vector<bool> problem?
  // return !inner_product(first + 1, last, first, false, logical_or<bool>(), comp);

  return !thrust::inner_product(first + 1, last, first, (int) 0, thrust::logical_or<int>(), comp);
} // end is_sorted()

} // end namespace thrust


