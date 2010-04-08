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


/*! \file mismatch.h
 *  \brief Search for differences between sequences [device].
 */

#pragma once

#include <thrust/pair.h>

#include <thrust/detail/device/generic/mismatch.h>

namespace thrust
{
namespace detail
{
namespace device
{

template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
thrust::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2,
                                                      BinaryPredicate pred)
{
    return thrust::detail::device::generic::mismatch(first1, last1, first2, pred);
}

} // end namespace device
} // end namespace detail
} // end namespace thrust

