/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file partition.h
 *  \brief Defines the interface to the
 *         dispatch layer of the partition
 *         family of functions.
 */

#pragma once

#include <algorithm>
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

///////////////
// Host Path //
///////////////
template<typename ForwardIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator stable_partition_copy(ForwardIterator begin,
                                       ForwardIterator end,
                                       OutputIterator result,
                                       Predicate pred,
                                       thrust::forward_host_iterator_tag,
                                       thrust::forward_host_iterator_tag)
{
  // copy [begin,end) to result
  std::copy(begin, end, result);

  return std::stable_partition(result, result + thrust::distance(begin, end), pred);
}


/////////////////
// Device Path //
/////////////////
template<typename ForwardIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator stable_partition_copy(ForwardIterator begin,
                                       ForwardIterator end,
                                       OutputIterator result,
                                       Predicate pred,
                                       thrust::random_access_device_iterator_tag,
                                       thrust::random_access_device_iterator_tag)
{
  thrust::unary_negate<Predicate> not_pred(pred);

  // remove_copy_if the false partition to result
  OutputIterator end_of_true_partition = thrust::remove_copy_if(begin, end, result, not_pred);

  // remove_copy_if the true partition to the end of the true partition
  thrust::remove_copy_if(begin, end, end_of_true_partition, pred);

  return end_of_true_partition;
}


///////////////
// Host Path //
///////////////
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator begin,
                                   ForwardIterator end,
                                   Predicate pred,
                                   thrust::forward_host_iterator_tag)
{
  return std::stable_partition(begin, end, pred);
}


/////////////////
// Device Path //
/////////////////
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator begin,
                                   ForwardIterator end,
                                   Predicate pred,
                                   thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // partition to temp space
  thrust::device_ptr<InputType> temp = thrust::device_malloc<InputType>(end - begin);
  thrust::copy(begin, end, temp);
  thrust::device_ptr<InputType> temp_middle = thrust::experimental::stable_partition_copy(begin, end, temp, pred);
    
  // copy back to original sequence
  thrust::copy(temp, temp + (end - begin), begin);

  // free temp space
  thrust::device_free(temp);

  return begin + (temp_middle - temp);
}


} // end namespace dispatch

} // end detail

} // end thrust

