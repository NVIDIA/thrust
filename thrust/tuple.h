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


/*! \file tuple.h
 *  \brief Defines a tuple type similar to std::tr1::tuple.
 */

// thrust::tuple is derived from boost::tuple of the
// Boost Tuples Library, which is the work of
// Jaako Järvi.
// See http://www.boost.org for details.

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/tuple.inl>
#include <thrust/pair.h>

namespace thrust
{

// forward declaration of null_type
struct null_type;

// define tuple_element
template<int N, class T>
  struct tuple_element
{
  private:
    typedef typename T::tail_type Next;

  public:
    typedef typename tuple_element<N-1, Next>::type type;
}; // end tuple_element

// define tuple_size
template<class T>
  struct tuple_size
{
  static const int value = 1 + tuple_size<typename T::tail_type>::value;
}; // end tuple_size

// get function for non-const cons-lists, returns a reference to the element

template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::non_const_type
get(detail::cons<HT, TT>& c);


// get function for const cons-lists, returns a const reference to
// the element. If the element is a reference, returns the reference
// as such (that is, can return a non-const reference)
template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::const_type
get(const detail::cons<HT, TT>& c);



template <class T0, class T1, class T2, class T3, class T4,
          class T5, class T6, class T7, class T8, class T9>
  class tuple :
    public detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
{
  private:
  typedef typename detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type inherited;

  public:
// access_traits<T>::parameter_type takes non-reference types as const T&
  inline __host__ __device__
  tuple(void) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0)
    : inherited(t0, detail::cnull(), detail::cnull(), detail::cnull(),
                detail::cnull(), detail::cnull(), detail::cnull(),
                detail::cnull(), detail::cnull(), detail::cnull()) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1)
    : inherited(t0, t1, detail::cnull(), detail::cnull(),
                detail::cnull(), detail::cnull(), detail::cnull(),
                detail::cnull(), detail::cnull(), detail::cnull()) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2)
    : inherited(t0, t1, t2, detail::cnull(), detail::cnull(),
                detail::cnull(), detail::cnull(), detail::cnull(),
                detail::cnull(), detail::cnull()) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3)
    : inherited(t0, t1, t2, t3, detail::cnull(), detail::cnull(),
                detail::cnull(), detail::cnull(), detail::cnull(),
                detail::cnull()) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4)
    : inherited(t0, t1, t2, t3, t4, detail::cnull(), detail::cnull(),
                detail::cnull(), detail::cnull(), detail::cnull()) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5)
    : inherited(t0, t1, t2, t3, t4, t5, detail::cnull(), detail::cnull(),
                detail::cnull(), detail::cnull()) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6)
    : inherited(t0, t1, t2, t3, t4, t5, t6, detail::cnull(),
                detail::cnull(), detail::cnull()) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6,
        typename access_traits<T7>::parameter_type t7)
    : inherited(t0, t1, t2, t3, t4, t5, t6, t7, detail::cnull(),
                detail::cnull()) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6,
        typename access_traits<T7>::parameter_type t7,
        typename access_traits<T8>::parameter_type t8)
    : inherited(t0, t1, t2, t3, t4, t5, t6, t7, t8, detail::cnull()) {}

  inline __host__ __device__ 
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6,
        typename access_traits<T7>::parameter_type t7,
        typename access_traits<T8>::parameter_type t8,
        typename access_traits<T9>::parameter_type t9)
    : inherited(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9) {}


  template<class U1, class U2>
  inline __host__ __device__ 
  tuple(const detail::cons<U1, U2>& p) : inherited(p) {}

  template <class U1, class U2>
  inline __host__ __device__ 
  tuple& operator=(const detail::cons<U1, U2>& k)
  {
    inherited::operator=(k);
    return *this;
  }

  template <class U1, class U2>
  __host__ __device__ inline
  tuple& operator=(const thrust::pair<U1, U2>& k) {
    //BOOST_STATIC_ASSERT(length<tuple>::value == 2);// check_length = 2
    this->head = k.first;
    this->tail.head = k.second;
    return *this;
  }
};

// The empty tuple
template <>
class tuple<null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type>  :
  public null_type
{
public:
  typedef null_type inherited;
};



template<class T0>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0>::type
    make_tuple(const T0& t0);

template<class T0, class T1>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1>::type
    make_tuple(const T0& t0, const T1& t1);

template<class T0, class T1, class T2>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2);

template<class T0, class T1, class T2, class T3>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3);

template<class T0, class T1, class T2, class T3, class T4>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4);

template<class T0, class T1, class T2, class T3, class T4, class T5>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8, const T9& t9);


// null_type comparison
__host__ __device__ inline
bool operator==(const null_type&, const null_type&);

__host__ __device__ inline
bool operator>=(const null_type&, const null_type&);

__host__ __device__ inline
bool operator<=(const null_type&, const null_type&);

__host__ __device__ inline
bool operator!=(const null_type&, const null_type&);

__host__ __device__ inline
bool operator<(const null_type&, const null_type&);

__host__ __device__ inline
bool operator>(const null_type&, const null_type&);

} // end thrust

