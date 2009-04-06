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


/*! \file remove.h
 *  \brief Dispatch layer to the remove functions.
 */

#pragma once

#include <algorithm>
#include <komrade/functional.h>
#include <komrade/copy.h>
#include <komrade/transform.h>
#include <komrade/scan.h>
#include <komrade/scatter.h>
#include <komrade/device_ptr.h>
#include <komrade/device_malloc.h>
#include <komrade/device_free.h>
#include <komrade/iterator/iterator_categories.h>
#include <komrade/iterator/iterator_traits.h>

namespace komrade
{

namespace detail
{

namespace dispatch
{

////////////////
// Host Paths //
////////////////
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator begin,
                            ForwardIterator end,
                            Predicate pred,
                            komrade::forward_host_iterator_tag)
{
  return std::remove_if(begin, end, pred);
}

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                Predicate pred,
                                komrade::input_host_iterator_tag,
                                komrade::input_host_iterator_tag)
{
  return std::remove_copy_if(begin, end, result, pred);
}


//////////////////
// Device Paths //
//////////////////
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator begin,
                            ForwardIterator end,
                            Predicate pred,
                            komrade::random_access_device_iterator_tag)
{
  // XXX do we need to call destructors for elements which get removed?

  typedef typename komrade::iterator_traits<ForwardIterator>::value_type InputType;

  // create temporary storage for an intermediate result
  komrade::device_ptr<InputType> temp = komrade::device_malloc<InputType>(end - begin);

  // remove into temp
  komrade::device_ptr<InputType> new_end = komrade::remove_copy_if(begin, end, temp, pred);

  // copy temp to the original range
  komrade::copy(temp, new_end, begin);
  
  // free temp space
  komrade::device_free(temp);

  return begin + (new_end - temp);
} 


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                Predicate pred,
                                komrade::random_access_device_iterator_tag,
                                komrade::random_access_device_iterator_tag)
{
  typedef typename komrade::iterator_traits<InputIterator>::difference_type difference_type;

  difference_type n = end - begin;

  difference_type size_of_new_sequence = 0;
  if(n > 0)
  {
    // negate the predicate -- this tells us which elements to keep
    komrade::unary_negate<Predicate> not_pred(pred);

    // evaluate not_pred on [begin,end), store result to temp vector
    komrade::device_ptr<difference_type> result_of_not_pred = komrade::device_malloc<difference_type>(n);

    komrade::transform(begin,
                       end,
                       result_of_not_pred,
                       not_pred);

    // scan the pred result to a temp vector
    komrade::device_ptr<difference_type> not_pred_scatter_indices = komrade::device_malloc<difference_type>(n);
    komrade::exclusive_scan(result_of_not_pred,
                            result_of_not_pred + n,
                            not_pred_scatter_indices);

    // scatter the true partition
    komrade::scatter_if(begin,
                        end,
                        not_pred_scatter_indices,
                        result_of_not_pred,
                        result);

    // find the end of the new sequence
    size_of_new_sequence = not_pred_scatter_indices[n - 1]
                         + (not_pred(*(end-1)) ? 1 : 0);

    komrade::device_free(result_of_not_pred);
    komrade::device_free(not_pred_scatter_indices);
  } // end if

  return result + size_of_new_sequence;
} // end remove_copy_if()

} // end namespace dispatch

} // end namespace detail

} // end namespace komrade

