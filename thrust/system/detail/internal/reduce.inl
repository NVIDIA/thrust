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
#include <thrust/system/detail/internal/reduce.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/system/detail/internal/default_decomposition.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/detail/internal/reduce_intervals.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace internal
{


template<typename Tag,
         typename RandomAccessIterator,
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(Tag,
                    RandomAccessIterator first,
                    RandomAccessIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference_type;

  const difference_type n = thrust::distance(first,last);

  // determine first and second level decomposition
  thrust::system::detail::internal::uniform_decomposition<difference_type> decomp1 = thrust::system::detail::internal::default_decomposition<Tag>(n);
  thrust::system::detail::internal::uniform_decomposition<difference_type> decomp2(decomp1.size() + 1, 1, 1);

  // allocate storage for the initializer and partial sums
  thrust::detail::temporary_array<OutputType,Tag> partial_sums(decomp1.size() + 1);
  
  // set first element of temp array to init
  partial_sums[0] = init;
  
  // accumulate partial sums (first level reduction)
  thrust::system::detail::internal::reduce_intervals(first, partial_sums.begin() + 1, binary_op, decomp1);

  // reduce partial sums (second level reduction)
  thrust::system::detail::internal::reduce_intervals(partial_sums.begin(), partial_sums.begin(), binary_op, decomp2);

  return partial_sums[0];
} // end reduce()


} // end internal
} // end detail
} // end system
} // end thrust

