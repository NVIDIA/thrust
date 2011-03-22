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


/*! \file copy.h
 *  \brief Dispatch layer for copy.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>

// host
#include <thrust/detail/host/copy.h>

// device
#include <thrust/detail/device/copy.h>

namespace thrust
{

namespace detail
{

namespace device
{

// XXX WAR circular #inclusion with these forward declarations
template<typename InputIterator, typename OutputIterator> OutputIterator copy(InputIterator, InputIterator, OutputIterator);

template<typename InputIterator, typename Size, typename OutputIterator> OutputIterator copy_n(InputIterator, Size, OutputIterator);

template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
  OutputIterator
    copy_if(InputIterator1,
            InputIterator1,
            InputIterator2,
            OutputIterator,
            Predicate);


} // end device

namespace dispatch
{


// XXX idea: detect whether both spaces
//     are convertible to host, if so
//     dispatch host to host path
//     if not, dispatch device::copy()

//////////
// copy //
//////////

///////////////////////
// Host to Host Path //
///////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::host_space_tag)
{
    return thrust::detail::host::copy(begin, end, result);
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator begin,
                        Size n,
                        OutputIterator result,
                        thrust::host_space_tag)
{
    return thrust::detail::host::copy_n(begin, n, result);
}

///////////////////////////
// Device to Device Path //
//////////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::device_space_tag)
{
    return thrust::detail::device::copy(begin, end, result);
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator begin,
                        Size n,
                        OutputIterator result,
                        thrust::device_space_tag)
{
    return thrust::detail::device::copy_n(begin, n, result);
}

///////////////
// Any Paths //
///////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::any_space_tag)
{
    return thrust::detail::device::copy(begin, end, result);
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator begin,
                        Size n,
                        OutputIterator result,
                        thrust::any_space_tag)
{
    return thrust::detail::device::copy_n(begin, n, result);
}

//////////////////////
// Cross-Space Path //
//////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::false_type cross_space_copy)
{
  return thrust::detail::device::copy(first, last, result);
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::false_type cross_space_copy)
{
  return thrust::detail::device::copy_n(first, n, result);
}

//////////////////////
// Intra-Space Path //
//////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::true_type cross_space_copy)
{
  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  // find the minimum space of the two
  typedef typename thrust::detail::minimum_space<space1,space2>::type minimum_space;

  return thrust::detail::dispatch::copy(first, last, result, minimum_space());
}


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::true_type cross_space_copy)
{
  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  // find the minimum space of the two
  typedef typename thrust::detail::minimum_space<space1,space2>::type minimum_space;

  return thrust::detail::dispatch::copy_n(first, n, result, minimum_space());
}


// entry point
template<typename InputIterator,
         typename OutputIterator,
         typename Space1,
         typename Space2>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      Space1,
                      Space2)
{
  return thrust::detail::dispatch::copy(first, last, result,
    typename thrust::detail::is_one_convertible_to_the_other<Space1,Space2>::type());
}


template<typename InputIterator,
         typename Size,
         typename OutputIterator,
         typename Space1,
         typename Space2>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        Space1,
                        Space2)
{
  return thrust::detail::dispatch::copy_n(first, n, result,
    typename thrust::detail::is_one_convertible_to_the_other<Space1,Space2>::type());
}



////////////////////////
// Host to Host Paths //
////////////////////////

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
  OutputIterator copy_if(InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred,
                         thrust::host_space_tag,
                         thrust::host_space_tag,
                         thrust::host_space_tag)
{
  return thrust::detail::host::copy_if(first, last, stencil, result, pred);
} // end copy_if()

////////////////////////////
// Device to Device Paths //
////////////////////////////

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred,
                         thrust::device_space_tag,
                         thrust::device_space_tag,
                         thrust::device_space_tag)
{
  return thrust::detail::device::copy_if(first, last, stencil, result, pred);
} // end copy_if()

} // end dispatch

} // end detail

} // end thrust

