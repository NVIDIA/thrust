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

#pragma once

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace detail
{

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename Compare,
         typename SplittingFunction,
         typename BlockConvergentSetOperation>
  RandomAccessIterator3 set_operation(RandomAccessIterator1 first1,
                                      RandomAccessIterator1 last1,
                                      RandomAccessIterator2 first2,
                                      RandomAccessIterator2 last2,
                                      RandomAccessIterator3 result,
                                      Compare comp,
                                      SplittingFunction split,
                                      BlockConvergentSetOperation set_op);


} // end detail
} // end cuda
} // end backend
} // end detail
} // end thrust

#include <thrust/detail/backend/cuda/detail/set_operation.inl>

