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
#include <cstddef>

namespace thrust
{
namespace detail
{

template<typename Ptr> struct pointer_element;

template<template<typename> class Ptr, typename Arg>
  struct pointer_element<Ptr<Arg> >
{
  typedef Arg type;
};

template<template<typename,typename> class Ptr, typename Arg1, typename Arg2>
  struct pointer_element<Ptr<Arg1,Arg2> >
{
  typedef Arg1 type;
};

template<typename T>
  struct pointer_element<T*>
{
  typedef T type;
};

template<typename Ptr, typename T> struct rebind_pointer;

template<typename T, typename U>
  struct rebind_pointer<T*,U>
{
  typedef U* type;
};

template<template<typename> class Ptr, typename Arg, typename T>
  struct rebind_pointer<Ptr<Arg>,T>
{
  typedef Ptr<T> type;
};

template<template<typename, typename> class Ptr, typename Arg1, typename Arg2, typename T>
  struct rebind_pointer<Ptr<Arg1,Arg2>,T>
{
  typedef Ptr<T,Arg2> type;
};

template<typename Ptr>
  struct pointer_traits
{
  typedef Ptr                                 pointer;
  typedef typename pointer_element<Ptr>::type element_type;
  typedef typename Ptr::difference_type       difference_type;

  template<typename U>
    struct rebind 
      : rebind_pointer<Ptr,U>
  {};

  __host__ __device__
  inline static pointer pointer_to(element_type &r)
  {
    // XXX this is supposed to be pointer::pointer_to(&r); (i.e., call a static member function of pointer called pointer_to)
    //     assume that pointer has a constructor from raw pointer instead
    
    return pointer(&r);
  }

  // thrust additions follow
  typedef element_type *                      raw_pointer;

  __host__ __device__
  inline static raw_pointer get(pointer ptr)
  {
    return ptr.get();
  }
};

template<typename T>
  struct pointer_traits<T*>
{
  typedef T*             pointer;
  typedef T              element_type;
  typedef std::ptrdiff_t difference_type;

  template<typename U>
    struct rebind
  {
    typedef U* other;
  };

  __host__ __device__
  inline static pointer pointer_to(T& r)
  {
    return &r;
  }

  // thrust additions follow
  typedef pointer        raw_pointer;

  __host__ __device__
  inline static raw_pointer get(pointer ptr)
  {
    return ptr;
  }
};

} // end detail
} // end thrust

