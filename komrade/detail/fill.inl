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


/*! \file fill.inl
 *  \brief Inline file for fill.h.
 */

#include <komrade/fill.h>
#include <komrade/iterator/iterator_traits.h>
#include <komrade/detail/dispatch/fill.h>

namespace komrade
{

template<typename InputIterator, typename T>
  void fill(InputIterator first,
            InputIterator last,
            const T &exemplar)
{
  // dispatch on category
  komrade::detail::dispatch::fill(first, last, exemplar,
    typename komrade::iterator_traits<InputIterator>::iterator_category());
} // end fill()

} // end namespace komrade

