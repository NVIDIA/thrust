/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
{


struct split_for_set_operation
{
  template<typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4,
           typename Compare,
           typename Size1,
           typename Size2>
    void operator()(RandomAccessIterator1 first1,
                    RandomAccessIterator1 last1,
                    RandomAccessIterator2 first2,
                    RandomAccessIterator2 last2,
                    RandomAccessIterator3 splitter_ranks1,
                    RandomAccessIterator4 splitter_ranks2,
                    Compare comp,
                    Size1 partition_size,
                    Size2 num_splitters_from_each_range);
}; // end split_for_set_operation


} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#include <thrust/detail/device/cuda/detail/split_for_set_operation.inl>

