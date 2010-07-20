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

#pragma once

#include <thrust/iterator/detail/placement/has_place.h>
#include <thrust/iterator/detail/placement/get_place.h>

namespace thrust
{

namespace detail
{

namespace has_place_detail
{
  struct tag {};
  struct any { template <class T> any(T const&); };

  static tag get_place(any const &);

  char (& test(tag) )[2];

  template<typename T>
  char test(T const &);

  template<typename Iterator>
    struct impl
  {
    static Iterator &test_me;
    static const bool value = sizeof(has_place_detail::test(get_place(test_me))) == 1;
  };
}

template<typename Iterator>
  struct has_place
{
  public:
    static const bool value = has_place_detail::impl<Iterator>::value;
}; // end has_place


} // end detail

} // end thrust

