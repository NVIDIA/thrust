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


#include <thrust/iterator/iterator_traits.h>

#include <thrust/remove.h>

#include <thrust/detail/internal_functional.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  typedef typename thrust::iterator_space<ForwardIterator>::type Space;

  // copy input to temp buffer
  thrust::detail::raw_buffer<InputType,Space> temp(first, last);

  return thrust::detail::device::generic::stable_partition_copy(temp.begin(), temp.end(), first, pred);
}

template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename Predicate>
  ForwardIterator2 stable_partition_copy(ForwardIterator1 first,
                                         ForwardIterator1 last,
                                         ForwardIterator2 result,
                                         Predicate pred)
{
  thrust::detail::unary_negate<Predicate> not_pred(pred);

  // remove_copy_if the false partition to result
  ForwardIterator2 end_of_true_partition = thrust::remove_copy_if(first, last, result, not_pred);

  // remove_copy_if the true partition to the end of the true partition
  thrust::remove_copy_if(first, last, end_of_true_partition, pred);

  return end_of_true_partition;
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
    return thrust::detail::device::generic::stable_partition(first, last, pred);
}

template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename Predicate>
  ForwardIterator2 partition_copy(ForwardIterator1 first,
                                  ForwardIterator1 last,
                                  ForwardIterator2 result,
                                  Predicate pred)
{
    return thrust::detail::device::generic::stable_partition_copy(first, last, result, pred);
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

