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


/*! \file copy.h
 *  \brief Dispatch layer for copy.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

// host
#include <algorithm>
#include <thrust/detail/host/copy.h>

// device
#include <thrust/detail/device/copy.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

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
                      thrust::host_space_tag,
                      thrust::host_space_tag)
{
    return std::copy(begin, end, result);
}


/////////////////////////
// Host to Device Path //
/////////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::host_space_tag,
                      thrust::device_space_tag)
{
    return thrust::detail::device::copy_host_to_device(begin, end, result);
}


/////////////////////////
// Device to Host Path //
/////////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::device_space_tag,
                      thrust::host_space_tag)
{
    return thrust::detail::device::copy_device_to_host(begin, end, result);
}

///////////////////////////
// Device to Device Path //
///////////////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::device_space_tag,
                      thrust::device_space_tag)
{
    return thrust::detail::device::copy_device_to_device(begin, end, result);
}

///////////////
// Any Paths //
///////////////

template<typename InputIterator,
         typename OutputIterator,
         typename Space>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::any_space_tag,
                      Space)
{
    return thrust::detail::dispatch::copy(begin, end, result, Space(), Space());
}

template<typename InputIterator,
         typename OutputIterator,
         typename Space>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      Space,
                      thrust::any_space_tag)
{
    return thrust::detail::dispatch::copy(begin, end, result, Space(), Space());
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::any_space_tag,
                      thrust::any_space_tag)
{
    return thrust::detail::dispatch::copy(begin, end, result, thrust::device_space_tag(), thrust::device_space_tag());
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

