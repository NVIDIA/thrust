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

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/execution_policy.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{

template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
  void trivial_copy_n(execution_policy<DerivedPolicy> &exec,
                      RandomAccessIterator1 first,
                      Size n,
                      RandomAccessIterator2 result);

template<typename System1,
         typename System2,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
  void trivial_copy_n(cross_system<System1,System2> &exec,
                      RandomAccessIterator1 first,
                      Size n,
                      RandomAccessIterator2 result);

} // end detail
} // end cuda
} // end system
} // end thrust

#include <thrust/system/cuda/detail/trivial_copy.inl>

