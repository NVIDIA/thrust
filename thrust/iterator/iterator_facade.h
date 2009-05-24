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


/*! \file iterator_facade.h
 *  \brief Defines a class which exposes the public
 *         interface that all iterators accessable from the
 *         host and device must implement.  Based on
 *         Boost's iterator_facade class.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

namespace experimental
{

// This forward declaration is required for the friend declaration
// in iterator_core_access
template<typename I, typename V, typename CT, typename R, typename P, typename D> class iterator_facade;

class iterator_core_access
{
    // declare our friends
    template<typename I, typename V, typename CT, typename R, typename P, typename D> friend class iterator_facade;

    // iterator comparisons are our friends
    template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
              class Dr2, class V2, class TC2, class R2, class P2, class D2>
    friend bool
    operator ==(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
                iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs);

    template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
              class Dr2, class V2, class TC2, class R2, class P2, class D2>
    friend bool
    operator !=(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
                iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs);

    template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
              class Dr2, class V2, class TC2, class R2, class P2, class D2>
    friend bool
    operator <(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
               iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs);

    template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
              class Dr2, class V2, class TC2, class R2, class P2, class D2>
    friend bool
    operator >(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
               iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs);

    template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
              class Dr2, class V2, class TC2, class R2, class P2, class D2>
    friend bool
    operator <=(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
               iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs);

    template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
              class Dr2, class V2, class TC2, class R2, class P2, class D2>
    friend bool
    operator >=(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
               iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs);

    // iterator difference is our friend
    template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
              class Dr2, class V2, class TC2, class R2, class P2, class D2>
    friend typename iterator_facade<Dr1,V1,TC1,R1,P1,D1>::difference_type
    operator-(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
              iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs);

    template<typename Facade>
    __host__ __device__
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

    // XXX TODO: Investigate whether we need both of these cases
    //template <class Facade1, class Facade2>
    //__host__ __device__
    //static typename Facade1::difference_type distance_from(Facade1 const& f1, Facade2 const& f2, mpl::true_)
    //{
    //  return -f1.distance_to(f2);
    //}

    //template <class Facade1, class Facade2>
    //__host__ __device__
    //static typename Facade2::difference_type distance_from(Facade1 const& f1, Facade2 const& f2, mpl::false_)
    //{
    //  return f2.distance_to(f1);
    //}
    
    template <class Facade1, class Facade2>
    __host__ __device__
    static typename Facade2::difference_type distance_from(Facade1 const& f1, Facade2 const& f2)
    {
      return f2.distance_to(f1);
    }

    //
    // Curiously Recurring Template interface.
    //
    template <class I, class V, class TC, class R, class P, class D>
    __host__ __device__
    static I& derived(iterator_facade<I,V,TC,R,P,D>& facade)
    {
      return *static_cast<I*>(&facade);
    }

    template <class I, class V, class TC, class R, class P, class D>
    __host__ __device__
    static I const& derived(iterator_facade<I,V,TC,R,P,D> const& facade)
    {
      return *static_cast<I const*>(&facade);
    }

  private:
    //// objects of this class are useless
    //__host__ __device__
    //iterator_core_access(); //undefined
}; // end iterator_core_access

template<typename Derived,
         typename Value,
         typename CategoryOrTraversal,
         typename Reference,
         typename Pointer,
         typename Difference>
  class iterator_facade
{
  private:
    // XXX move these somewhere else
    template<typename T>
      struct remove_const
    {
      typedef T type;
    }; // end remove_const
    
    template<typename T>
      struct remove_const<const T>
    {
      typedef T type;
    }; // end remove_const

    //
    // Curiously Recurring Template interface.
    //
    Derived& derived()
    {
      return *static_cast<Derived*>(this);
    }

    Derived const& derived() const
    {
      return *static_cast<Derived const*>(this);
    }

  public:
    typedef typename remove_const<Value>::type value_type;
    typedef Reference                         reference;
    typedef Pointer                           pointer;
    typedef Difference                        difference_type;

    // XXX investigate whether or not we need to go to the lengths
    //     boost does to determine this type
    typedef CategoryOrTraversal               iterator_category;

    __host__ __device__
    reference operator*() const
    {
      return iterator_core_access::dereference(this->derived());
    }

    // XXX investigate whether or not we need to go to the lengths
    //     boost does to determine the return type
    __host__ __device__
    pointer operator->() const
    {
      return this->derived();
    }

    // XXX investigate whether or not we need to go to the lengths
    //     boost does to determine the return type
    __host__ __device__
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
template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
          class Dr2, class V2, class TC2, class R2, class P2, class D2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator ==(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
            iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs)
{
  return iterator_core_access
    ::equal(*static_cast<Dr1 const*>(&lhs),
            *static_cast<Dr2 const*>(&rhs));
}

template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
          class Dr2, class V2, class TC2, class R2, class P2, class D2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator !=(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
            iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs)
{
  return !iterator_core_access
    ::equal(*static_cast<Dr1 const*>(&lhs),
            *static_cast<Dr2 const*>(&rhs));
}

template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
          class Dr2, class V2, class TC2, class R2, class P2, class D2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator <(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
           iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs)
{
  return 0 > iterator_core_access
    ::distance_from(*static_cast<Dr1 const*>(&lhs),
                    *static_cast<Dr2 const*>(&rhs));
}

template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
          class Dr2, class V2, class TC2, class R2, class P2, class D2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator >(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
           iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs)
{
  return 0 < iterator_core_access
    ::distance_from(*static_cast<Dr1 const*>(&lhs),
                    *static_cast<Dr2 const*>(&rhs));
}

template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
          class Dr2, class V2, class TC2, class R2, class P2, class D2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator <=(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
            iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs)
{
  return 0 >= iterator_core_access
    ::distance_from(*static_cast<Dr1 const*>(&lhs),
                    *static_cast<Dr2 const*>(&rhs));
}

template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
          class Dr2, class V2, class TC2, class R2, class P2, class D2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator >=(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
            iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs)
{
  return 0 <= iterator_core_access
    ::distance_from(*static_cast<Dr1 const*>(&lhs),
                    *static_cast<Dr2 const*>(&rhs));
}

// Iterator difference
template <class Dr1, class V1, class TC1, class R1, class P1, class D1,
          class Dr2, class V2, class TC2, class R2, class P2, class D2>
inline __host__ __device__
// XXX investigate whether we need to do extra work to determine the return type
//     as Boost does
typename iterator_facade<Dr1,V1,TC1,R1,P1,D1>::difference_type
operator-(iterator_facade<Dr1,V1,TC1,R1,P1,D1> const& lhs,
          iterator_facade<Dr2,V2,TC2,R2,P2,D2> const& rhs)
{
  return iterator_core_access
    ::distance_from(*static_cast<Dr1 const*>(&lhs),
                    *static_cast<Dr2 const*>(&rhs));
}

// Iterator addition
template <class Derived, class V, class TC, class R, class P, class D>
inline __host__ __device__
Derived operator+ (iterator_facade<Derived,V,TC,R,P,D> const& i,
                   typename Derived::difference_type n)
{
  Derived tmp(static_cast<Derived const&>(i));
  return tmp += n;
}

template <class Derived, class V, class TC, class R, class P, class D>
inline __host__ __device__
Derived operator+ (typename Derived::difference_type n,
                   iterator_facade<Derived,V,TC,R,P,D> const& i)
{
  Derived tmp(static_cast<Derived const&>(i));
  return tmp += n;
}

} // end experimental

} // end thrust

