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

#pragma once

#include <komrade/detail/config.h>
#include <komrade/iterator/iterator_adaptor.h>
#include <komrade/iterator/iterator_categories.h>

namespace komrade
{

namespace experimental
{

template<typename Incrementable,
         // XXX TODO figure out whether we need CategoryOrTraversal
         // XXX TODO infer Difference type automatically
         typename Difference>
  class counting_iterator
    : public iterator_adaptor<counting_iterator<Incrementable,Difference>,
                              Incrementable,
                              Incrementable,
                              // XXX TODO infer the category automatically
                              komrade::experimental::random_access_universal_iterator_tag,
                              Incrementable const &,
                              Incrementable *,
                              Difference>
{
    friend class iterator_core_access;

    typedef iterator_adaptor<counting_iterator<Incrementable,Difference>,
                             Incrementable,
                             Incrementable,
                             // XXX TODO infer the category automatically
                             komrade::experimental::random_access_universal_iterator_tag,
                             Incrementable const &,
                             Incrementable *,
                             Difference> super_t;


  public:
    typedef Incrementable const & reference;

    __host__ __device__
    counting_iterator(void){};

    // XXX nvcc can't compile this at the moment
    //__host__ __device__
    //counting_iterator(counting_iterator const &rhs):super_t(rhs.base()){}

    __host__ __device__
    explicit counting_iterator(Incrementable x):super_t(x){}

  private:
    __host__ __device__
    reference dereference(void) const
    {
      return this->base_reference();
    }
}; // end counting_iterator

// XXX TODO infer the Difference type automatically
template <typename Incrementable, typename Difference>
inline counting_iterator<Incrementable, Difference>
make_counting_iterator(Incrementable x)
{
  return counting_iterator<Incrementable, Difference>(x);
}

} // end experimental

namespace detail
{

template<typename Incrementable, typename Difference>
  struct make_device_dereferenceable< komrade::experimental::counting_iterator<Incrementable,Difference> >
{
  __host__ __device__
  static
  komrade::experimental::counting_iterator<Incrementable,Difference> &
  transform(komrade::experimental::counting_iterator<Incrementable,Difference> &x)
  {
    return x;
  } // end transform()

  __host__ __device__
  static
  const komrade::experimental::counting_iterator<Incrementable,Difference> &
  transform(const komrade::experimental::counting_iterator<Incrementable,Difference> &x)
  {
    return x;
  } // end transform()
}; // end make_device_dereferenceable

} // end detail

} // end komrade

