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


/*! \file gather.inl
 *  \brief Inline file for gather.h.
 */

#include <thrust/gather.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>

// XXX remove these two when we no longer have to support the old
//     gather interface
#include <thrust/advance.h>
#include <thrust/distance.h>

#include <thrust/copy.h>

namespace thrust
{

// XXX remove this namespace in Thrust v1.3
namespace next
{

template<typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator gather(InputIterator        map_first,
                        InputIterator        map_last,
                        RandomAccessIterator input_first,
                        OutputIterator       result)
{
  return thrust::copy(thrust::make_permutation_iterator(input_first, map_first),
                      thrust::make_permutation_iterator(input_first, map_last),
                      result);
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
  return thrust::next::gather_if(map_first,
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
  return thrust::copy_when(thrust::make_permutation_iterator(input_first, map_first),
                           thrust::make_permutation_iterator(input_first, map_last),
                           stencil,
                           result,
                           pred);
} // end gather_if()

} // end next


// XXX remove this in Thrust v1.3
namespace deprecated
{


// XXX remove this in Thrust v1.3
template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input)
{
  // find the end of the map range
  typedef typename thrust::iterator_difference<ForwardIterator>::type difference_type;
  difference_type n = thrust::distance(first,last);
  InputIterator map_last = map;
  thrust::advance(map_last, n);

  thrust::next::gather(map, map_last, input, first);
} // end gather()


// XXX remove this in Thrust v1.3
template<typename ForwardIterator,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void gather_if(ForwardIterator first,
                 ForwardIterator last,
                 InputIterator1 map,
                 InputIterator2 stencil,
                 RandomAccessIterator input)
{
  typedef typename thrust::iterator_traits<InputIterator2>::value_type StencilType;
  thrust::deprecated::gather_if(first, last, map, stencil, input, thrust::identity<StencilType>());
} // end gather_if()


// XXX remove this in Thrust v1.3
template<typename ForwardIterator,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename Predicate>
  void gather_if(ForwardIterator first,
                 ForwardIterator last,
                 InputIterator1 map,
                 InputIterator2 stencil,
                 RandomAccessIterator input,
                 Predicate pred)
{
  // find the end of the map range
  typedef typename thrust::iterator_difference<ForwardIterator>::type difference_type;
  difference_type n = thrust::distance(first,last);
  InputIterator1 map_last = map;
  thrust::advance(map_last, n);

  thrust::next::gather_if(map, map_last, stencil, input, first, pred);
} // end gather_if()

} // end deprecated

} // end namespace thrust

