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


#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>

#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/advance.h>

#include <thrust/detail/internal_functional.h>
#include <thrust/detail/uninitialized_array.h>

namespace thrust
{
namespace detail
{
namespace backend
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
  thrust::detail::uninitialized_array<InputType,Space> temp(first, last);

  // count the size of the true partition
  typename thrust::iterator_difference<ForwardIterator>::type num_true = thrust::count_if(first,last,pred);

  // point to the beginning of the false partition
  ForwardIterator out_false = first;
  thrust::advance(out_false, num_true);

  return thrust::stable_partition_copy(temp.begin(), temp.end(), first, out_false, pred).first;
}

template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  thrust::detail::unary_negate<Predicate> not_pred(pred);

  // remove_copy_if the true partition to out_true
  OutputIterator1 end_of_true_partition = thrust::remove_copy_if(first, last, out_true, not_pred);

  // remove_copy_if the false partition to out_false
  OutputIterator2 end_of_false_partition = thrust::remove_copy_if(first, last, out_false, pred);

  return thrust::make_pair(end_of_true_partition, end_of_false_partition);
}

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  return thrust::stable_partition(first, last, pred);
}

template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  return thrust::stable_partition_copy(first,last,out_true,out_false,pred);
}

} // end namespace generic
} // end namespace backend
} // end namespace detail
} // end namespace thrust

