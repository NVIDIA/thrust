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


/*! \file gather.inl
 *  \brief Inline file for gather.h.
 */

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>

#include <thrust/transform.h>
#include <thrust/functional.h>

namespace thrust
{

template<typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator gather(InputIterator        map_first,
                        InputIterator        map_last,
                        RandomAccessIterator input_first,
                        OutputIterator       result)
{
  return thrust::transform(thrust::make_permutation_iterator(input_first, map_first),
                           thrust::make_permutation_iterator(input_first, map_last),
                           result,
                           thrust::identity<typename thrust::iterator_value<RandomAccessIterator>::type>());
} // end gather()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator gather_if(InputIterator1       map_first,
                           InputIterator1       map_last,
                           InputIterator2       stencil,
                           RandomAccessIterator input_first,
                           OutputIterator       result)
{
  typedef typename thrust::iterator_value<InputIterator2>::type StencilType;
  return thrust::gather_if(map_first,
                           map_last,
                           stencil,
                           input_first,
                           result,
                           thrust::identity<StencilType>());
} // end gather_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator gather_if(InputIterator1       map_first,
                           InputIterator1       map_last,
                           InputIterator2       stencil,
                           RandomAccessIterator input_first,
                           OutputIterator       result,
                           Predicate            pred)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type InputType;
  return thrust::transform_if(thrust::make_permutation_iterator(input_first, map_first),
                              thrust::make_permutation_iterator(input_first, map_last),
                              stencil,
                              result,
                              thrust::identity<InputType>(),
                              pred);
} // end gather_if()

} // end namespace thrust

