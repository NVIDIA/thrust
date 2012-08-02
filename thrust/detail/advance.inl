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


/*! \file advance.inl
 *  \brief Inline file for advance.h
 */

#include <thrust/detail/config.h>
#include <thrust/advance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/advance.h>
#include <thrust/system/detail/adl/advance.h>

namespace thrust
{


template <typename System, typename InputIterator, typename Distance>
void advance(thrust::detail::dispatchable_base<System> &system, InputIterator& i, Distance n)
{
  using thrust::system::detail::generic::advance;

  advance(system.derived(), i, n);
} // end advance()


namespace detail
{


template <typename System, typename InputIterator, typename Distance>
void strip_const_advance(const System &system, InputIterator& i, Distance n)
{
  System &non_const_system = const_cast<System&>(system);

  thrust::advance(non_const_system, i, n);
} // end strip_const_advance()


namespace advance_detail
{


// This helper exists purely to avoid warnings concerning null references
// from the null pointer used in advance's dispatch below
template<typename System>
inline __host__ __device__
System &deref(System *ptr)
{
  return *ptr;
} // end deref()

} // end advance_detail


} // end detail


template <typename InputIterator, typename Distance>
void advance(InputIterator& i, Distance n)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  // XXX avoid default-constructing a System
  // XXX we justify this hack for advance's dispatch (though, not the dispatch
  // XXX of other algorithms) because advance is more similar to the pointer
  // XXX manipulation functions dispatched by thrust::reference
  // XXX than it is to the algorithms
  System *system = 0;

  thrust::detail::strip_const_advance(select_system(detail::advance_detail::deref(system)), i, n);
} // end advance()


} // end namespace thrust

