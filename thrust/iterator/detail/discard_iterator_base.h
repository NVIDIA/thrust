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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <cstddef> // for std::ptrdiff_t

namespace thrust
{

// forward declaration of discard_iterator
template<typename> class discard_iterator;

namespace detail
{

// a type which may be assigned any other type
struct any_assign
{
  inline __host__ __device__ any_assign(void)
  {}

  template<typename T>
  inline __host__ __device__ any_assign(T)
  {}

  template<typename T>
  inline __host__ __device__
  any_assign &operator=(T)
  {
    if(0)
    {
      // trick the compiler into silencing "warning: this expression has no effect"
      int *x = 0;
      *x = 13;
    } // end if

    return *this;
  }
};

template<typename Space>
  struct discard_iterator_base
{
  // XXX value_type should actually be void
  //     but this interferes with zip_iterator<discard_iterator>
  typedef any_assign        value_type;
  typedef any_assign        reference;
  typedef void              pointer;
  typedef std::ptrdiff_t    incrementable;

  typedef typename thrust::counting_iterator<
    incrementable,
    Space,
    thrust::random_access_traversal_tag
  > base_iterator;

  typedef typename thrust::experimental::iterator_adaptor<
    discard_iterator<Space>,
    base_iterator,
    pointer,
    value_type,
    typename thrust::iterator_space<base_iterator>::type,
    typename thrust::iterator_traversal<base_iterator>::type,
    reference
  > type;
}; // end discard_iterator_base

} // end detail
  
} // end thrust


