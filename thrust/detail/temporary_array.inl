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

#include <thrust/detail/temporary_array.h>
#include <thrust/distance.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/detail/copy.h>


namespace thrust
{

namespace detail
{


template<typename T, typename System>
  temporary_array<T,System>
    ::temporary_array(thrust::dispatchable<System> &system, size_type n)
      :super_t(n, alloc_type(temporary_allocator<T,System>(system)))
{
  ;
} // end temporary_array::temporary_array()


namespace temporary_array_detail
{


template<typename System, typename Iterator1, typename Iterator2>
Iterator2 strip_const_copy(const System &system, Iterator1 first, Iterator1 last, Iterator2 result)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::copy(non_const_system, first, last, result);
} // end strip_const_copy()


} // end temporary_array_detail


template<typename T, typename System>
  template<typename InputIterator>
    temporary_array<T,System>
      ::temporary_array(thrust::dispatchable<System> &system, InputIterator first, InputIterator last,
                        typename disable_if<
                          is_same<System, typename thrust::iterator_system<InputIterator>::type>::value
                        >::type *)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  typedef typename thrust::iterator_system<InputIterator>::type InputSystem;
  InputSystem input_system;

  super_t::allocate(thrust::distance(input_system,first,last));

  // since this copy is cross-system,
  // use select_system to get the system to dispatch on

  using thrust::system::detail::generic::select_system;

  // XXX this copy should actually be copy construct via allocator
  temporary_array_detail::strip_const_copy(select_system(input_system, system.derived()), first, last, super_t::begin());
} // end temporary_array::temporary_array()


template<typename T, typename System>
  template<typename InputIterator>
    temporary_array<T,System>
      ::temporary_array(thrust::dispatchable<System> &system, InputIterator first, InputIterator last,
                        typename enable_if<
                          is_same<System, typename thrust::iterator_system<InputIterator>::type>::value
                        >::type *)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  super_t::allocate(thrust::distance(system,first,last));

  thrust::copy(system, first, last, super_t::begin());
} // end temporary_array::temporary_array()


} // end detail

} // end thrust

