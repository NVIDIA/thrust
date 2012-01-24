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
#include <thrust/system/detail/internal/reduce_intervals.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/cpp/detail/reduce_intervals.h>
#include <thrust/system/omp/detail/reduce_intervals.h>
#include <thrust/system/cuda/detail/reduce_intervals.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace internal
{

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void reduce_intervals(InputIterator input,
                      OutputIterator output,
                      BinaryFunction binary_op,
                      Decomposition decomp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  reduce_intervals(select_system(space1(),space2()), input, output, binary_op, decomp);
} // end reduce_intervals()

} // end internal
} // end detail
} // end system
} // end thrust

