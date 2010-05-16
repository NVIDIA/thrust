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

//  Copyright Thorsten Ottosen 2003-2004. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// For more information, see http://www.boost.org/libs/range/

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <cstddef> // for std::size_t
#include <utility> // for std::pair


namespace thrust
{

namespace experimental
{

namespace detail
{


// default
template<typename Range>
  struct range_size
{
  typedef typename Range::size_type type;
};


// std::pair
template<typename Iterator>
  struct range_size< std::pair<Iterator,Iterator> >
{
  typedef std::size_t type;
};


// thrust::pair
template<typename Iterator>
  struct range_size< thrust::pair<Iterator,Iterator> >
{
  typedef std::size_t type;
};


// thrust::tuple
template<typename Iterator>
  struct range_size< thrust::tuple<Iterator,Iterator> >
{
  typedef std::size_t type;
};


// array
template<typename T, std::size_t sz>
  struct range_size<T[sz]>
{
  typedef std::size_t type;
};


} // end detail


template<typename Range>
  struct range_size :
    detail::range_size<Range>
{};


// i assume this specialization is to handle const arrays
template<typename Range>
  struct range_size<const Range> :
    detail::range_size<Range>
{};


} // end experimental

} // end thrust

