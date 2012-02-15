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


/*! \file iterator_traits.h
 *  \brief Traits and metafunctions for reasoning about the traits of iterators
 */

/*
 * (C) Copyright David Abrahams 2003.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <iterator>

namespace thrust
{

template<typename T>
  struct iterator_traits
    : public std::iterator_traits<T>
{
}; // end iterator_traits


// define space tags
struct host_space_tag {};

struct device_space_tag {};


// define Boost's traversal tags
struct no_traversal_tag {};

struct incrementable_traversal_tag
  : no_traversal_tag {};

struct single_pass_traversal_tag
  : incrementable_traversal_tag {};

struct forward_traversal_tag
  : single_pass_traversal_tag {};

struct bidirectional_traversal_tag
  : forward_traversal_tag {};

struct random_access_traversal_tag
  : bidirectional_traversal_tag {};


template<typename Iterator> struct iterator_value;

template<typename Iterator> struct iterator_pointer;

template<typename Iterator> struct iterator_reference;

template<typename Iterator> struct iterator_difference;

template<typename Iterator> struct iterator_traversal;

template<typename Iterator> struct iterator_space;

} // end thrust

#include <thrust/iterator/detail/any_space_tag.h>
#include <thrust/iterator/detail/iterator_traits.inl>

