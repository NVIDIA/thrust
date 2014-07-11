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
namespace detail
{


template<unsigned int work_per_thread,
         typename DerivedPolicy,
         typename Context,
         typename RandomAccessIterator1,
         typename Pointer,
         typename RandomAccessIterator2,
         typename Compare>
__host__ __device__
void stable_sort_each_copy(execution_policy<DerivedPolicy> &exec,
                           Context context,
                           unsigned int block_size,
                           RandomAccessIterator1 first, RandomAccessIterator1 last,
                           Pointer vitual_smem,
                           RandomAccessIterator2 result,
                           Compare comp);


} // end detail
} // end detail
} // end cuda
} // end system
} // end thrust

#include <thrust/system/cuda/detail/detail/stable_sort_each.inl>

