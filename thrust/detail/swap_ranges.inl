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


/*! \file swap_ranges.inl
 *  \brief Inline file for swap_ranges.h.
 */

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/for_each.h>

namespace thrust
{

namespace detail
{

// XXX define this here rather than in internal_functional.h
// to avoid circular dependence between swap.h & internal_functional.h
struct swap_pair_elements
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  { 
    thrust::swap(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end swap_pair_elements



// XXX WAR circular #inclusion problems with this forward declaration
template<typename I, typename F>
I for_each(I, I, F);

} // end detail


template<typename ForwardIterator1,
         typename ForwardIterator2>
  ForwardIterator2 swap_ranges(ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2)
{
  typedef thrust::tuple<ForwardIterator1,ForwardIterator2> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple>              ZipIterator;

  ZipIterator result = thrust::detail::for_each(thrust::make_zip_iterator(thrust::make_tuple(first1, first2)),
                                                thrust::make_zip_iterator(thrust::make_tuple(last1,  first2)),
                                                detail::swap_pair_elements());
  return thrust::get<1>(result.get_iterator_tuple());
} // end swap_ranges()

} // end namespace thrust

