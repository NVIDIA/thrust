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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/copy_if.h>
#include <thrust/detail/copy_if.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/transform.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/type_traits.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <limits>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{

template<typename IndexType,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
OutputIterator copy_if(InputIterator1 first,
                       InputIterator1 last,
                       InputIterator2 stencil,
                       OutputIterator result,
                       Predicate pred)
{
    typedef typename thrust::detail::minimum_system<
      typename thrust::iterator_system<InputIterator1>::type,
      typename thrust::iterator_system<InputIterator2>::type,
      typename thrust::iterator_system<OutputIterator>::type
    >::type System;

    __THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING(IndexType n = thrust::distance(first, last));

    // compute {0,1} predicates
    thrust::detail::temporary_array<IndexType, System> predicates(n);
    thrust::transform(stencil,
                      stencil + n,
                      predicates.begin(),
                      thrust::detail::predicate_to_integral<Predicate,IndexType>(pred));

    // scan {0,1} predicates
    thrust::detail::temporary_array<IndexType, System> scatter_indices(n);
    thrust::exclusive_scan(predicates.begin(),
                           predicates.end(),
                           scatter_indices.begin(),
                           static_cast<IndexType>(0),
                           thrust::plus<IndexType>());

    // scatter the true elements
    thrust::scatter_if(first,
                       last,
                       scatter_indices.begin(),
                       predicates.begin(),
                       result,
                       thrust::identity<IndexType>());

    // find the end of the new sequence
    IndexType output_size = scatter_indices[n - 1] + predicates[n - 1];

    return result + output_size;
}

} // end namespace detail


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(tag,
                         InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         Predicate pred)
{
  // XXX it's potentially expensive to send [first,last) twice
  //     we should probably specialize this case for POD
  //     since we can safely keep the input in a temporary instead
  //     of doing two loads
  return thrust::copy_if(first, last, first, result, pred);
} // end copy_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
   OutputIterator copy_if(tag,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator result,
                          Predicate pred)
{
  typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;
  
  // empty sequence
  if(first == last)
    return result;
  
  difference_type n = thrust::distance(first, last);
  
  // create an unsigned version of n (we know n is positive from the comparison above)
  // to avoid a warning in the compare below
  typename thrust::detail::make_unsigned<difference_type>::type unsigned_n(n);
  
  // use 32-bit indices when possible (almost always)
  if(sizeof(difference_type) > sizeof(unsigned int) && unsigned_n > (std::numeric_limits<unsigned int>::max)())
  {
    result = detail::copy_if<difference_type>(first, last, stencil, result, pred);
  } // end if
  else
  {
    result = detail::copy_if<unsigned int>(first, last, stencil, result, pred);
  } // end else

  return result;
} // end copy_if()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

