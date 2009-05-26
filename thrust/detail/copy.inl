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

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/copy.h>
#include <thrust/transform.h>

namespace thrust
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
  return thrust::detail::dispatch::copy(begin, end, result,
           typename thrust::iterator_traits<InputIterator>::iterator_category(),
           typename thrust::iterator_traits<OutputIterator>::iterator_category());
} // end copy()


template<typename InputIterator,
         typename PredicateIterator,
         typename OutputIterator>
  OutputIterator copy_when(InputIterator begin,
                           InputIterator end,
                           PredicateIterator stencil,
                           OutputIterator result)
{
  // default predicate is identity
  typedef typename thrust::iterator_traits<PredicateIterator>::value_type StencilType;
  return thrust::copy_when(begin, end, stencil, result, thrust::identity<StencilType>());
} // end copy_when()


template<typename InputIterator,
         typename PredicateIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_when(InputIterator begin,
                           InputIterator end,
                           PredicateIterator stencil,
                           OutputIterator result,
                           Predicate pred)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  return thrust::transform_if(begin, end, stencil, result, thrust::identity<InputType>(), pred);
} // end copy_when()

} // end namespace thrust

