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

/*! \file general_copy.h
 *  \brief Sequential copy algorithms for general iterators.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace sequential
{
namespace general_copy_detail
{


template<typename InputIterator, typename OutputIterator>
struct reference_is_assignable
  : thrust::detail::is_assignable<
      typename thrust::iterator_reference<OutputIterator>::type,
      typename thrust::iterator_reference<InputIterator>::type
    >
{};


// introduce an iterator assign helper to deal with assignments from
// a wrapped reference

__thrust_hd_warning_disable__
template<typename OutputIterator, typename InputIterator>
inline __host__ __device__
typename thrust::detail::enable_if<
  reference_is_assignable<InputIterator,OutputIterator>::value
>::type
iter_assign(OutputIterator dst, InputIterator src)
{
  *dst = *src;
}


__thrust_hd_warning_disable__
template<typename OutputIterator, typename InputIterator>
inline __host__ __device__
typename thrust::detail::disable_if<
  reference_is_assignable<InputIterator,OutputIterator>::value
>::type
iter_assign(OutputIterator dst, InputIterator src)
{
  typedef typename thrust::iterator_value<InputIterator>::type value_type;

  // insert a temporary and hope for the best
  *dst = static_cast<value_type>(*src);
}


} // end general_copy_detail


__thrust_hd_warning_disable__
template<typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator general_copy(InputIterator first,
                              InputIterator last,
                              OutputIterator result)
{
  for(; first != last; ++first, ++result)
  {
    general_copy_detail::iter_assign(result, first);
  }

  return result;
} // end general_copy()


__thrust_hd_warning_disable__
template<typename InputIterator,
         typename Size,
         typename OutputIterator>
__host__ __device__
  OutputIterator general_copy_n(InputIterator first,
                                Size n,
                                OutputIterator result)
{
  for(; n > Size(0); ++first, ++result, --n)
  {
    general_copy_detail::iter_assign(result, first);
  }

  return result;
} // end general_copy_n()


} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace thrust

