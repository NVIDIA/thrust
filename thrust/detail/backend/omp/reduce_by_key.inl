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

#include <thrust/detail/config.h>
#include <thrust/detail/backend/omp/reduce_by_key.h>
#include <thrust/detail/backend/generic/reduce_by_key.h>
#include <thrust/distance.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace omp
{

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
    reduce_by_key(tag,
                  InputIterator1 keys_first, 
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output,
                  BinaryPredicate binary_pred,
                  BinaryFunction binary_op)
{
  // omp prefers generic::reduce_by_key to cpp::reduce_by_key
  return thrust::detail::backend::generic::reduce_by_key(tag(), keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
} // end reduce_by_key()


} // end omp
} // end backend
} // end detail
} // end thrust

