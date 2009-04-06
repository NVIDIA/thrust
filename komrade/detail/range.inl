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


/*! \file range.inl
 *  \brief Inline file for range.h.
 */

#include <komrade/range.h>
#include <komrade/iterator/iterator_traits.h>
#include <komrade/detail/dispatch/range.h>

namespace komrade
{

template<typename ForwardIterator>
  void range(ForwardIterator first,
             ForwardIterator last)
{
  typedef typename komrade::iterator_traits<ForwardIterator>::value_type OutputType;
  komrade::range(first, last, OutputType(0), OutputType(1));
} // end range()


template<typename ForwardIterator, typename T>
  void range(ForwardIterator first,
             ForwardIterator last,
             T init)
{
  komrade::range(first, last, init, T(1));
} // end range()


template<typename ForwardIterator, typename T>
  void range(ForwardIterator first,
             ForwardIterator last,
             T init,
             T step)
{
  komrade::detail::dispatch::range(first, last, init, step,
    typename komrade::iterator_traits<ForwardIterator>::iterator_category());
} // end range()

} // end namespace komrade


