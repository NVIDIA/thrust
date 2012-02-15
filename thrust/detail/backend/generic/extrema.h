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


/*! \file extrema.h
 *  \brief Generic device implementations of extrema functions.
 */

#pragma once

#include <thrust/pair.h>
#include <thrust/detail/backend/generic/extrema.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace generic
{

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp);

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp);

template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp);

} // end namespace generic
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#include "extrema.inl"

