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
#include <komrade/partition.h>
#include <komrade/functional.h>
#include <komrade/iterator/iterator_categories.h>
#include <komrade/remove.h>
#include <komrade/copy.h>
#include <komrade/distance.h>
#include <komrade/remove.h>
#include <komrade/device_ptr.h>
#include <komrade/device_malloc.h>
#include <komrade/device_free.h>

namespace komrade
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
                                       komrade::forward_host_iterator_tag,
                                       komrade::forward_host_iterator_tag)
{
  // copy [begin,end) to result
  std::copy(begin, end, result);

  return std::stable_partition(result, result + komrade::distance(begin, end), pred);
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
                                       komrade::random_access_device_iterator_tag,
                                       komrade::random_access_device_iterator_tag)
{
  komrade::unary_negate<Predicate> not_pred(pred);

  // remove_copy_if the false partition to result
  OutputIterator end_of_true_partition = komrade::remove_copy_if(begin, end, result, not_pred);

  // remove_copy_if the true partition to the end of the true partition
  komrade::remove_copy_if(begin, end, end_of_true_partition, pred);

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
                                   komrade::forward_host_iterator_tag)
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
                                   komrade::random_access_device_iterator_tag)
{
  typedef typename komrade::iterator_traits<ForwardIterator>::value_type InputType;

  // partition to temp space
  komrade::device_ptr<InputType> temp = komrade::device_malloc<InputType>(end - begin);
  komrade::copy(begin, end, temp);
  komrade::device_ptr<InputType> temp_middle = komrade::stable_partition_copy(begin, end, temp, pred);
    
  // copy back to original sequence
  komrade::copy(temp, temp + (end - begin), begin);

  // free temp space
  komrade::device_free(temp);

  return begin + (temp_middle - temp);
}


} // end namespace dispatch

} // end detail

} // end komrade

