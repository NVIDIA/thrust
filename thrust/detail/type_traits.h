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


/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
 */

#pragma once

#include <thrust/detail/util/static.h>

// XXX nvcc 2.2 closed beta can't compile type_traits
//// find type_traits
//
//#ifdef __GNUC__
//
//#if __GNUC__ == 4 && __GNUC_MINOR__ == 2
//#include <tr1/type_traits>
//#elif __GNUC__ == 4 && __GNUC_MINOR__ > 2
//#include <type_traits>
//#endif // GCC version
//
//#endif // GCC
//
//#ifdef _MSC_VER
//#include <type_traits>
//#endif // MSVC

namespace thrust
{

namespace detail
{

//typedef std::tr1::true_type  true_type;
//typedef std::tr1::false_type false_type;

typedef thrust::detail::util::Bool2Type<true> true_type;
typedef thrust::detail::util::Bool2Type<false> false_type;

template<typename T>
  struct is_pod
    : public false_type
{
}; // end is_pod

// all pointers are pod
template<typename T> struct is_pod<T*> : public true_type {};

// bool types are pod
template<> struct is_pod<bool> : public true_type {};

// char types are pod
template<> struct is_pod<char> : public true_type {};
template<> struct is_pod<unsigned char> : public true_type {};
template<> struct is_pod<signed char> : public true_type {};

// short types are pod
template<> struct is_pod<short> : public true_type {};
template<> struct is_pod<unsigned short> : public true_type {};

// int types are pod
template<> struct is_pod<int> : public true_type {};
template<> struct is_pod<unsigned int> : public true_type {};

// long types are pod
template<> struct is_pod<long> : public true_type {};
template<> struct is_pod<unsigned long> : public true_type {};

// long long types are pod
template<> struct is_pod<long long> : public true_type {};
template<> struct is_pod<unsigned long long> : public true_type {};

// real types are pod
template<> struct is_pod<float> : public true_type {};
template<> struct is_pod<double> : public true_type {};


//template<typename T> struct is_integral : public std::tr1::is_integral<T> {};
template<typename U> struct is_integral : public false_type {};

template<> struct is_integral<bool>               : public true_type {};
template<> struct is_integral<char>               : public true_type {};
template<> struct is_integral<unsigned char>      : public true_type {};
template<> struct is_integral<short>              : public true_type {};
template<> struct is_integral<unsigned short>     : public true_type {};
template<> struct is_integral<int>                : public true_type {};
template<> struct is_integral<unsigned int>       : public true_type {};
template<> struct is_integral<long>               : public true_type {};
template<> struct is_integral<unsigned long>      : public true_type {};
template<> struct is_integral<long long>          : public true_type {};
template<> struct is_integral<unsigned long long> : public true_type {};

// these two are synonyms for each other
//template<typename T> struct has_trivial_copy : public std::tr1::has_trivial_copy<T> {};
//template<typename T> struct has_trivial_copy_constructor : public std::tr1::has_trivial_copy<T> {};
//
//template<typename T> struct has_trivial_destructor : public std::tr1::has_trivial_destructor<T> {};
//template<typename T> struct has_trivial_assign : public std::tr1::has_trivial_assign<T> {};

template<typename T> struct has_trivial_copy : public is_pod<T> {};
template<typename T> struct has_trivial_copy_constructor : public is_pod<T> {};

template<typename T> struct has_trivial_destructor : public is_pod<T> {};
template<typename T> struct has_trivial_assign : public is_pod<T> {};

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

template<typename T>
  struct remove_volatile
{
  typedef T type;
}; // end remove_volatile

template<typename T>
  struct remove_volatile<volatile T>
{
  typedef T type;
}; // end remove_volatile

template<typename T>
  struct remove_cv
{
  typedef typename remove_const<typename remove_volatile<T>::type>::type type;
}; // end remove_cv

template<typename T>
  struct remove_reference
{
  typedef T type;
}; // end remove_reference

template<typename T>
  struct remove_reference<T&>
{
  typedef T type;
}; // end remove_reference

template<typename T1, typename T2>
  struct is_same
    : public false_type
{
}; // end is_same

template<typename T>
  struct is_same<T,T>
    : public true_type
{
}; // end is_same

} // end detail

} // end thrust

