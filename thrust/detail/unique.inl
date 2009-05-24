/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file unique.inl
 *  \brief Inline file for unique.h.
 */

#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/unique.h>

namespace thrust
{
    
template <typename ForwardIterator>
ForwardIterator unique(ForwardIterator first, ForwardIterator last)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  return thrust::unique(first, last, thrust::equal_to<InputType>());
} // end unique()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first, ForwardIterator last,
                       BinaryPredicate binary_pred)
{
  return detail::dispatch::unique(first, last, binary_pred,
    typename thrust::iterator_traits<ForwardIterator>::iterator_category());
} // end unique()

} // end namespace thrust

