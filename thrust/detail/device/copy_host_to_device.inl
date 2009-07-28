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


/*! \file copy_device_to_device.h
 *  \brief Device implementations for copying on the device.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <stdlib.h>           // for malloc & free
#include <thrust/distance.h>
#include <thrust/device_ptr.h>

#include <thrust/detail/device/trivial_copy.h>

namespace thrust
{

namespace detail
{

namespace device
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::input_host_iterator_tag, 
                      thrust::random_access_device_iterator_tag)
{
    // host container to device container
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    typename thrust::iterator_traits<InputIterator>::difference_type n = thrust::distance(begin, end);

    // allocate temporary storage
    InputType *temp = reinterpret_cast<InputType*>(malloc(sizeof(InputType) * n));
    InputType *temp_end = thrust::copy(begin, end, temp);

    result = thrust::copy(temp, temp_end, result);

    free(temp);
    return result;
}

// host pointer to device pointer with matching types
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_host_iterator_tag,
                      thrust::random_access_device_iterator_tag,
                      typename thrust::iterator_traits<OutputIterator>::value_type *,            // InputIterator::pointer
                      thrust::device_ptr<typename thrust::iterator_traits<OutputIterator>::value_type>)  // OutputIterator::pointer
{
  // specialization for host to device when the types pointed to match (and operator= is not overloaded)

  // how many elements to copy?
  typename thrust::iterator_traits<OutputIterator>::difference_type n = end - begin;

  // what is the output type?
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  // XXX TODO check is_normal_iterator()
  trivial_copy_host_to_device((&*result).get(), &*begin,  n * sizeof(OutputType));

  return result + n;
}

// random access host iterator to random access device iterator with mixed types
template<typename InputIterator,
         typename OutputIterator,
         typename InputIteratorPointer,
         typename OutputIteratorPointer>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_host_iterator_tag,
                      thrust::random_access_device_iterator_tag,
                      InputIteratorPointer,
                      OutputIteratorPointer)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  typename thrust::iterator_traits<InputIterator>::difference_type n = thrust::distance(begin,end);

  // allocate temporary storage
  OutputType *temp = reinterpret_cast<OutputType*>(malloc(sizeof(OutputType) * n));
  OutputType *temp_end = thrust::copy(begin, end, temp);

  result = thrust::copy(temp, temp_end, result);

  free(temp);
  return result;
}

// random access host iterator to random access device iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_host_iterator_tag input_tag,
                      thrust::random_access_device_iterator_tag output_tag)
{
    // dispatch on the type of each iterator's pointers
    // XXX also need to dispatch on if output type fulfills has_trivial_assign_operator
    return copy(begin, end, result, input_tag, output_tag,
            typename thrust::iterator_traits<InputIterator>::pointer(),
            typename thrust::iterator_traits<OutputIterator>::pointer());
}


/////////////////
// Entry Point //
/////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_host_to_device(InputIterator begin, 
                                     InputIterator end, 
                                     OutputIterator result)
{
    return copy(begin, end, result, 
            typename thrust::iterator_traits<InputIterator>::iterator_category(),
            typename thrust::iterator_traits<OutputIterator>::iterator_category());
}

} // end namespace device

} // end namespace detail

} // end namespace thrust

