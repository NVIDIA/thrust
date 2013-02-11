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


/*! \file binary_search.h
 *  \brief C++ implementation of binary search algorithms.
 */

#pragma once

#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/detail/internal/scalar/binary_search.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template <typename ForwardIterator,
          typename T,
          typename StrictWeakOrdering>
ForwardIterator lower_bound(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T& val,
                            StrictWeakOrdering comp)
{
  return thrust::system::detail::internal::scalar::lower_bound(first, last, val, comp);
}


template <typename ForwardIterator,
          typename T,
          typename StrictWeakOrdering>
ForwardIterator upper_bound(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T& val, 
                            StrictWeakOrdering comp)
{
  return thrust::system::detail::internal::scalar::upper_bound(first, last, val, comp);
}

template <typename ForwardIterator,
          typename T,
          typename StrictWeakOrdering>
bool binary_search(tag,
                   ForwardIterator first,
                   ForwardIterator last,
                   const T& val, 
                   StrictWeakOrdering comp)
{
  return thrust::system::detail::internal::scalar::binary_search(first, last, val, comp);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

