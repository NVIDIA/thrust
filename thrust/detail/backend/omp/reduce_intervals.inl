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

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/backend/dereference.h>

using thrust::detail::backend::dereference;

namespace thrust
{
namespace detail
{
namespace backend
{
namespace omp
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
  typedef typename thrust::iterator_value<OutputIterator>::type OutputType;
  typedef thrust::detail::intptr_t index_type;

  index_type n = static_cast<index_type>(decomp.size());

#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
# pragma omp parallel for
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
  for(index_type i = 0; i < n; i++)
  {
    InputIterator begin = input + decomp[i].begin();
    InputIterator end   = input + decomp[i].end();

    if (begin != end)
    {
      OutputType sum = dereference(begin);

      ++begin;

      while (begin != end)
      {
        sum = binary_op(sum, dereference(begin));
        ++begin;
      }

      OutputIterator tmp = output + i;
      dereference(tmp) = sum;
    }
  }
}

} // end namespace omp
} // end namespace backend
} // end namespace detail
} // end namespace thrust

