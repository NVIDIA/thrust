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


/*! \file copy.inl
 *  \brief Inline file for copy.h.
 */

#include <komrade/copy.h>
#include <komrade/functional.h>
#include <komrade/iterator/iterator_categories.h>
#include <komrade/iterator/iterator_traits.h>
#include <komrade/detail/dispatch/copy.h>

namespace komrade
{

//////////////////
// Entry Points //
//////////////////
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result)
{
  // XXX make sure this isn't necessary
  if(begin == end) 
    return result;

  // dispatch on category
  return komrade::detail::dispatch::copy(begin, end, result,
           typename komrade::iterator_traits<InputIterator>::iterator_category(),
           typename komrade::iterator_traits<OutputIterator>::iterator_category());
} // end copy()


template<typename InputIterator,
         typename PredicateIterator,
         typename OutputIterator>
  OutputIterator copy_if(InputIterator begin,
                         InputIterator end,
                         PredicateIterator stencil,
                         OutputIterator result)
{
  // default predicate is identity
  typedef typename komrade::iterator_traits<PredicateIterator>::value_type StencilType;
  return komrade::copy_if(begin, end, stencil, result, komrade::identity<StencilType>());
} // end copy_if()


template<typename InputIterator,
         typename PredicateIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator begin,
                         InputIterator end,
                         PredicateIterator stencil,
                         OutputIterator result,
                         Predicate pred)
{
  // dispatch on category
  return komrade::detail::dispatch::copy_if(begin, end, stencil, result, pred,
           typename komrade::iterator_traits<InputIterator>::iterator_category(),
           typename komrade::iterator_traits<PredicateIterator>::iterator_category(),
           typename komrade::iterator_traits<OutputIterator>::iterator_category());
} // end copy_if()

} // end namespace komrade

