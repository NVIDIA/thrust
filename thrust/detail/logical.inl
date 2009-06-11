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


/*! \file logical.inl
 *  \brief Inline file for logical.h.
 */

#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

template <class InputIterator, class Predicate>
bool all_of(InputIterator begin, InputIterator end, Predicate pred)
{
    return thrust::transform_reduce(begin, end, pred, true, thrust::logical_and<bool>());
}

template <class InputIterator, class Predicate>
bool any_of(InputIterator begin, InputIterator end, Predicate pred)
{
    return thrust::transform_reduce(begin, end, pred, false, thrust::logical_or<bool>());
}

template <class InputIterator, class Predicate>
bool none_of(InputIterator begin, InputIterator end, Predicate pred)
{
    return !any_of(begin, end, pred);
}

} // namespace end thrust

