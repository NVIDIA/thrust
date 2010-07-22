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

// default
template<typename Range>
  struct range_const_iterator
{
  // XXX Boost treats const_iterator as an optional typedef
  //     we may wish to provide this option as well
  typename Range::const_iterator type;
};


// std::pair
template<typename Iterator>
  struct range_const_iterator< std::pair<Iterator,Iterator> >
{
  typedef Iterator type;
};


// thrust::pair
template<typename Iterator>
  struct range_const_iterator< thrust::pair<Iterator,Iterator> >
{
  typedef Iterator type;
};


// thrust::tuple
template<typename Iterator>
  struct range_const_iterator< thrust::tuple<Iterator,Iterator> >
{
  typedef Iterator type;
};


// array
template<typename T, std::size_t sz>
struct range_const_iterator< T[sz] >
{
  typedef const T* type;
};


} // end experimental

} // end thrust

