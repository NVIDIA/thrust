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

/*! \file iterator_facade.h
 *  \brief A class which exposes a public interface for iterators
 */

/*
 * (C) Copyright David Abrahams 2002.
 * (C) Copyright Jeremy Siek    2002.
 * (C) Copyright Thomas Witt    2002.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */


#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/iterator_facade.inl>
#include <thrust/iterator/detail/distance_from_result.h>

#define ITERATOR_FACADE_FORMAL_PARMS      typename    Derived, typename    Pointer, typename    Value, typename    Space, typename    Traversal, typename    Reference, typename    Difference
#define ITERATOR_FACADE_FORMAL_PARMS_I(i) typename Derived##i, typename Pointer##i, typename Value##i, typename Space##i, typename Traversal##i, typename Reference##i, typename Difference##i

#define ITERATOR_FACADE_ARGS         Derived,    Pointer,    Value,    Space,    Traversal,    Reference,    Difference
#define ITERATOR_FACADE_ARGS_I(i) Derived##i, Pointer##i, Value##i, Space##i, Traversal##i, Reference##i, Difference##i

namespace thrust
{

namespace experimental
{

// This forward declaration is required for the friend declaration
// in iterator_core_access
template<ITERATOR_FACADE_FORMAL_PARMS> class iterator_facade;

class iterator_core_access
{
    // declare our friends
    template<ITERATOR_FACADE_FORMAL_PARMS> friend class iterator_facade;

    // iterator comparisons are our friends
    template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
              ITERATOR_FACADE_FORMAL_PARMS_I(2)>
    inline __host__ __device__
    friend bool
    operator ==(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
                iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs);

    template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
              ITERATOR_FACADE_FORMAL_PARMS_I(2)>
    inline __host__ __device__
    friend bool
    operator !=(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
                iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs);

    template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
              ITERATOR_FACADE_FORMAL_PARMS_I(2)>
    inline __host__ __device__
    friend bool
    operator <(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
               iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs);

    template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
              ITERATOR_FACADE_FORMAL_PARMS_I(2)>
    inline __host__ __device__
    friend bool
    operator >(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
               iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs);

    template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
              ITERATOR_FACADE_FORMAL_PARMS_I(2)>
    inline __host__ __device__
    friend bool
    operator <=(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
                iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs);

    template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
              ITERATOR_FACADE_FORMAL_PARMS_I(2)>
    inline __host__ __device__
    friend bool
    operator >=(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
                iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs);

    // iterator difference is our friend
    template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
              ITERATOR_FACADE_FORMAL_PARMS_I(2)>
    inline __host__ __device__
    friend
      typename thrust::detail::distance_from_result<
        iterator_facade<ITERATOR_FACADE_ARGS_I(1)>,
        iterator_facade<ITERATOR_FACADE_ARGS_I(2)>
      >::type
    operator-(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
              iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs);

    template<typename Facade>
    static typename Facade::reference dereference(Facade const& f)
    {
      return f.dereference();
    }

    template<typename Facade>
    __host__ __device__
    static void increment(Facade& f)
    {
      f.increment();
    }

    template<typename Facade>
    __host__ __device__
    static void decrement(Facade& f)
    {
      f.decrement();
    }

    template <class Facade1, class Facade2>
    __host__ __device__
    static bool equal(Facade1 const& f1, Facade2 const& f2)
    {
      return f1.equal(f2);
    }

    // XXX TODO: Investigate whether we need both of these cases
    //template <class Facade1, class Facade2>
    //__host__ __device__
    //static bool equal(Facade1 const& f1, Facade2 const& f2, mpl::true_)
    //{
    //  return f1.equal(f2);
    //}

    //template <class Facade1, class Facade2>
    //__host__ __device__
    //static bool equal(Facade1 const& f1, Facade2 const& f2, mpl::false_)
    //{
    //  return f2.equal(f1);
    //}

    template <class Facade>
    __host__ __device__
    static void advance(Facade& f, typename Facade::difference_type n)
    {
      f.advance(n);
    }

    // Facade2 is convertible to Facade1,
    // so return Facade1's difference_type
    template <class Facade1, class Facade2>
    __host__ __device__
    static typename Facade1::difference_type
      distance_from(Facade1 const& f1, Facade2 const& f2, thrust::detail::true_type)
    {
      return -f1.distance_to(f2);
    }

    // Facade2 is not convertible to Facade1,
    // so return Facade2's difference_type
    template <class Facade1, class Facade2>
    __host__ __device__
    static typename Facade2::difference_type
      distance_from(Facade1 const& f1, Facade2 const& f2, thrust::detail::false_type)
    {
      return f2.distance_to(f1);
    }
    
    template <class Facade1, class Facade2>
    __host__ __device__
    static typename thrust::detail::distance_from_result<Facade1,Facade2>::type
      distance_from(Facade1 const& f1, Facade2 const& f2)
    {
      // dispatch the implementation of this method upon whether or not
      // Facade2 is convertible to Facade1
      return distance_from(f1, f2,
        typename thrust::detail::is_convertible<Facade2,Facade1>::type());
    }

    //
    // Curiously Recurring Template interface.
    //
    template <ITERATOR_FACADE_FORMAL_PARMS>
    __host__ __device__
    static Derived& derived(iterator_facade<ITERATOR_FACADE_ARGS>& facade)
    {
      return *static_cast<Derived*>(&facade);
    }

    template <ITERATOR_FACADE_FORMAL_PARMS>
    __host__ __device__
    static Derived const& derived(iterator_facade<ITERATOR_FACADE_ARGS> const& facade)
    {
      return *static_cast<Derived const*>(&facade);
    }

  private:
    //// objects of this class are useless
    //__host__ __device__
    //iterator_core_access(); //undefined
}; // end iterator_core_access


template<ITERATOR_FACADE_FORMAL_PARMS>
  class iterator_facade
{
  private:
    //
    // Curiously Recurring Template interface.
    //
    __host__ __device__
    Derived& derived()
    {
      return *static_cast<Derived*>(this);
    }

    __host__ __device__
    Derived const& derived() const
    {
      return *static_cast<Derived const*>(this);
    }

    typedef detail::iterator_facade_types<
      Value, Space, Traversal, Reference, Difference
    > associated_types;

  public:
    typedef typename associated_types::value_type value_type;
    typedef Reference                             reference;
    typedef Pointer                               pointer;
    typedef Difference                            difference_type;
    typedef typename associated_types::iterator_category       iterator_category;

    reference operator*() const
    {
      return iterator_core_access::dereference(this->derived());
    }

    // XXX investigate whether or not we need to go to the lengths
    //     boost does to determine the return type
    pointer operator->() const
    {
      return this->derived();
    }

    // XXX investigate whether or not we need to go to the lengths
    //     boost does to determine the return type
    reference operator[](difference_type n) const
    {
      return *(this->derived() + n);
    }

    __host__ __device__
    Derived& operator++()
    {
      iterator_core_access::increment(this->derived());
      return this->derived();
    }

    __host__ __device__
    Derived  operator++(int)
    {
      Derived tmp(this->derived());
      ++*this;
      return tmp;
    }

    __host__ __device__
    Derived& operator--()
    {
      iterator_core_access::decrement(this->derived());
      return this->derived();
    }

    __host__ __device__
    Derived  operator--(int)
    {
      Derived tmp(this->derived());
      --*this;
      return tmp;
    }

    __host__ __device__
    Derived& operator+=(difference_type n)
    {
      iterator_core_access::advance(this->derived(), n);
      return this->derived();
    }

    __host__ __device__
    Derived& operator-=(difference_type n)
    {
      iterator_core_access::advance(this->derived(), -n);
      return this->derived();
    }

    __host__ __device__
    Derived  operator-(difference_type n) const
    {
      Derived result(this->derived());
      return result -= n;
    }

  protected:
    typedef iterator_facade iterator_facade_;

}; // end iterator_facade

// Comparison operators
template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
          ITERATOR_FACADE_FORMAL_PARMS_I(2)>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator ==(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
            iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs)
{
  return iterator_core_access
    ::equal(*static_cast<Derived1 const*>(&lhs),
            *static_cast<Derived2 const*>(&rhs));
}

template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
          ITERATOR_FACADE_FORMAL_PARMS_I(2)>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator !=(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
            iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs)
{
  return !iterator_core_access
    ::equal(*static_cast<Derived1 const*>(&lhs),
            *static_cast<Derived2 const*>(&rhs));
}

template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
          ITERATOR_FACADE_FORMAL_PARMS_I(2)>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator <(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
           iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs)
{
  return 0 > iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
          ITERATOR_FACADE_FORMAL_PARMS_I(2)>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator >(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
           iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs)
{
  return 0 < iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
          ITERATOR_FACADE_FORMAL_PARMS_I(2)>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator <=(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
            iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs)
{
  return 0 >= iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
          ITERATOR_FACADE_FORMAL_PARMS_I(2)>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator >=(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
            iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs)
{
  return 0 <= iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

// Iterator difference
template <ITERATOR_FACADE_FORMAL_PARMS_I(1),
          ITERATOR_FACADE_FORMAL_PARMS_I(2)>
inline __host__ __device__

// divine the type this operator returns
typename thrust::detail::distance_from_result<
  iterator_facade<ITERATOR_FACADE_ARGS_I(1)>,
  iterator_facade<ITERATOR_FACADE_ARGS_I(2)>
>::type

operator-(iterator_facade<ITERATOR_FACADE_ARGS_I(1)> const& lhs,
          iterator_facade<ITERATOR_FACADE_ARGS_I(2)> const& rhs)
{
  return iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

// Iterator addition
template <ITERATOR_FACADE_FORMAL_PARMS>
inline __host__ __device__
Derived operator+ (iterator_facade<ITERATOR_FACADE_ARGS> const& i,
                   typename Derived::difference_type n)
{
  Derived tmp(static_cast<Derived const&>(i));
  return tmp += n;
}

template <ITERATOR_FACADE_FORMAL_PARMS>
inline __host__ __device__
Derived operator+ (typename Derived::difference_type n,
                   iterator_facade<ITERATOR_FACADE_ARGS> const& i)
{
  Derived tmp(static_cast<Derived const&>(i));
  return tmp += n;
}

} // end experimental

} // end thrust

#undef ITERATOR_FACADE_FORMAL_PARMS
#undef ITERATOR_FACADE_FORMAL_PARMS_I
#undef ITERATOR_FACADE_ARGS
#undef ITERATOR_FACADE_ARGS_I

