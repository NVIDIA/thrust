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
#include <thrust/detail/has_nested_type.h>

#include <cstddef> // for std::size_t
#include <utility> // for std::pair
#include <thrust/pair.h>
#include <thrust/tuple.h>

namespace thrust
{

namespace experimental
{

namespace detail
{


// define a has_iterator trait, which checks for a nested type named "iterator"
__THRUST_DEFINE_HAS_NESTED_TYPE(has_iterator, iterator);


// for now, just check if there is a nested iterator type and specialize as appropriate
// XXX implement this for the general case upon arrival of c++0x SFINAE for expressions:
//     http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html
template<typename T> struct is_range : has_iterator<T> {};


// std::pair
// XXX check is_iterator<Iterator>
template<typename Iterator>
  struct is_range< std::pair<Iterator,Iterator> >
    : thrust::detail::true_type
{};


// thrust::pair
// XXX check is_iterator<Iterator>
template<typename Iterator>
  struct is_range< thrust::pair<Iterator,Iterator> >
    : thrust::detail::true_type
{};


// thrust::tuple
// XXX check is_iterator<Iterator>
template<typename Iterator>
  struct is_range< thrust::tuple<Iterator,Iterator> >
    : thrust::detail::true_type
{};


// array
template<typename T, std::size_t sz>
  struct is_range<T[sz]>
    : thrust::detail::true_type
{};


} // end detail

} // end experimental

} // end thrust

