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
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
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
                            thrust::forward_host_iterator_tag)
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
                                thrust::input_host_iterator_tag,
                                thrust::input_host_iterator_tag)
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
                            thrust::random_access_device_iterator_tag)
{
  // XXX do we need to call destructors for elements which get removed?

  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // create temporary storage for an intermediate result
  thrust::device_ptr<InputType> temp = thrust::device_malloc<InputType>(end - begin);

  // remove into temp
  thrust::device_ptr<InputType> new_end = thrust::remove_copy_if(begin, end, temp, pred);

  // copy temp to the original range
  thrust::copy(temp, new_end, begin);
  
  // free temp space
  thrust::device_free(temp);

  return begin + (new_end - temp);
} 


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                Predicate pred,
                                thrust::random_access_device_iterator_tag,
                                thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<InputIterator>::difference_type difference_type;

  difference_type n = end - begin;

  difference_type size_of_new_sequence = 0;
  if(n > 0)
  {
    // negate the predicate -- this tells us which elements to keep
    thrust::unary_negate<Predicate> not_pred(pred);

    // evaluate not_pred on [begin,end), store result to temp vector
    thrust::device_ptr<difference_type> result_of_not_pred = thrust::device_malloc<difference_type>(n);

    thrust::transform(begin,
                       end,
                       result_of_not_pred,
                       not_pred);

    // scan the pred result to a temp vector
    thrust::device_ptr<difference_type> not_pred_scatter_indices = thrust::device_malloc<difference_type>(n);
    thrust::exclusive_scan(result_of_not_pred,
                            result_of_not_pred + n,
                            not_pred_scatter_indices);

    // scatter the true partition
    thrust::scatter_if(begin,
                        end,
                        not_pred_scatter_indices,
                        result_of_not_pred,
                        result);

    // find the end of the new sequence
    size_of_new_sequence = not_pred_scatter_indices[n - 1]
                         + (not_pred(*(end-1)) ? 1 : 0);

    thrust::device_free(result_of_not_pred);
    thrust::device_free(not_pred_scatter_indices);
  } // end if

  return result + size_of_new_sequence;
} // end remove_copy_if()

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

