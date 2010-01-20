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

/*! \file permutation_iterator.h
 *  \brief An iterator which iterates over a permutation of a range.
 */

// thrust::permutation_iterator is derived from
// boost::permutation_iterator of the Boost Iterator
// Library, which is the work of
// David Abrahams, Jeremy Siek, & Thomas Witt.
// See http://www.boost.org for details.

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/permutation_iterator_base.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace experimental
{

template <typename ElementIterator,
          typename IndexIterator>
  class permutation_iterator
    : public detail::permutation_iterator_base<
        ElementIterator,
        IndexIterator
      >::type
{
  private:
    typedef detail::permutation_iterator_base<ElementIterator,IndexIterator>::type super_t;

    friend class iterator_core_access;

  public:
    __host__ __device__
    permutation_iterator()
      : m_element_iterator() {}

    __host__ __device__
    explicit permutation_iterator(ElementIterator x, IndexIterator y)
      : super_t(y), m_element_iterator(x) {}

    template<typename OtherElementIterator, typename OtherIndexIterator>
    __host__ __device__
    permutation_iterator(permutation_iterator<OtherElementIterator,OtherIndexIterator> const &r
    , typename detail::enable_if_convertible<OtherElementIterator, ElementIterator>::type* = 0
    , typename detail::enable_if_convertible<OtherIndexIterator, IndexIterator>::type* = 0
    )
      : super_t(r.base()), m_element_iterator(r.m_element_iterator)
    {}

  private:
    __host__ __device__
    typename super_t::reference dereference() const
    {
      return *(m_element_iterator + *this->base());
    }

    // make friends for the copy constructor
    template<typename,typename> friend class permutation_iterator;

    // make friends with the dereferencer
    friend template<typename, typename>
      inline __host__ __device__
        typename iterator_device_reference< thrust::experimental::permutation_iterator<ElementIterator, IndexIterator> >::type
          dereference(thrust::experimental::permutation_iterator<ElementIterator, IndexIterator> iter);

    ElementIterator m_element_iterator;
} // end permutation_iterator


template<typename ElementIterator, typename IndexIterator>
__host__ __device__
permutation_iterator<ElementIterator,IndexIterator> make_permutation_iterator(ElementIterator e, IndexIterator i)
{
  return permutation_iterator<ElementIterator,IndexIterator>(e,i);
}


} // end experimental

} // end thrust

#include <thrust/iterator/detail/permutation_iterator.inl>

