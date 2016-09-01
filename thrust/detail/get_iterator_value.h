#pragma once
/*
 *  Copyright 2008-2016 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/execution_policy.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/system/detail/generic/memory.h> // for get_value()

namespace thrust {
namespace detail {

// dereference an iterator with a provided execution policy
// This should handle safe derefencing
// of raw (device) pointer, smart pointer and iterators
template<typename DerivedPolicy, typename Iterator>
__host__ __device__
typename thrust::iterator_traits<Iterator>::value_type
get_iterator_value(thrust::execution_policy<DerivedPolicy> &exec, Iterator it)
{
  typename thrust::iterator_traits<Iterator>::value_type value;
  thrust::detail::two_system_copy_n(exec, thrust::cpp::tag(), 
                                    it, 1, &value);
  return value; 
} // get_iterator_value(exec,Iterator);

} // namespace detail
} // namespace thrust
