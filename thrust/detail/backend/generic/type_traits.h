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


/*! \file generic/type_traits.h
 *  \brief Introspection for free functions defined in generic.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{
namespace detail
{
namespace backend
{

// we must define these traits outside of generic's namespace
namespace generic_type_traits_ns
{

namespace get_temporary_buffer_exists_ns
{
  typedef char yes;
  typedef char (&no)[2];

  struct any_conversion
  {
    template<typename T> any_conversion(const T &);
  };

  template<typename T>
  no get_temporary_buffer(const any_conversion &, const any_conversion &);

  template<typename T> yes check(const T &);

  no check(no);

  template<typename T, typename Tag, typename Size>
    struct get_temporary_buffer_exists
  {
    static Tag  &tag;
    static Size &n;

    static const bool value = sizeof(check(get_temporary_buffer<T>(tag,n))) == sizeof(yes);
  };
} // end get_temporary_buffer_ns

} // end generic_type_traits_ns

namespace generic
{

template<typename T, typename Tag, typename Size>
  struct get_temporary_buffer_exists
    : generic_type_traits_ns::get_temporary_buffer_exists_ns::get_temporary_buffer_exists<T,Tag,Size>
{};

} // end backend
} // end backend
} // end detail
} // end thrust

