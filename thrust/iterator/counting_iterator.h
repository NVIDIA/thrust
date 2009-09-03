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


/*! \file counting_iterator.h
 *  \brief Defines the interface to an iterator
 *         which adapts an incrementable type
 *         to return the current value of the incrementable
 *         upon operator*(). Based on Boost's counting_iterator
 *         class.
 */

// thrust::counting_iterator is derived from
// boost::counting_iterator of the Boost Iterator
// Library, which is the work of
// David Abrahams, Jeremy Siek, & Thomas Witt.
// See http://www.boost.org for details.

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_categories.h>

// #include the details first
#include <thrust/iterator/detail/counting_iterator.inl>

namespace thrust
{

template<typename Incrementable,
         typename Space = use_default,
         typename Traversal = use_default,
         typename Difference = use_default>
  class counting_iterator
    : public detail::counting_iterator_base<Incrementable, Space, Traversal, Difference>::type
{
    typedef typename detail::counting_iterator_base<Incrementable, Space, Traversal, Difference>::type super_t;

    friend class thrust::experimental::iterator_core_access;

  public:
    typedef Incrementable const & reference;
    typedef typename super_t::difference_type difference_type;
    typedef counting_iterator device_dereferenceable_type;

    __host__ __device__
    counting_iterator(void){};

    __host__ __device__
    counting_iterator(counting_iterator const &rhs):super_t(rhs.base()){}

    __host__ __device__
    explicit counting_iterator(Incrementable x):super_t(x){}

  private:
    __host__ __device__
    reference dereference(void) const
    {
      return this->base_reference();
    }

    template <class OtherIncrementable>
    __host__ __device__
    difference_type
    distance_to(counting_iterator<OtherIncrementable, Space, Traversal, Difference> const& y) const
    {
      typedef typename
      thrust::detail::eval_if<
        thrust::detail::is_numeric<Incrementable>::value,
        thrust::detail::identity_<thrust::detail::number_distance<difference_type, Incrementable, OtherIncrementable> >,
        thrust::detail::identity_<thrust::detail::iterator_distance<difference_type, Incrementable, OtherIncrementable> >
      >::type d;

      return d::distance(this->base(), y.base());
    }
}; // end counting_iterator

template <typename Incrementable>
inline counting_iterator<Incrementable>
make_counting_iterator(Incrementable x)
{
  return counting_iterator<Incrementable>(x);
}

} // end thrust

