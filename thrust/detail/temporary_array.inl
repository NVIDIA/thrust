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


template<typename T, typename System>
  template<typename InputIterator>
    temporary_array<T,System>
      ::temporary_array(thrust::dispatchable<System> &system,
                        InputIterator first,
                        size_type n)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  super_t::allocate(n);

  // XXX this copy should actually be copy construct via allocator
  thrust::copy_n(system, first, n, super_t::begin());
} // end temporary_array::temporary_array()


template<typename T, typename System>
  template<typename InputIterator, typename InputSystem>
    temporary_array<T,System>
      ::temporary_array(thrust::dispatchable<System> &system,
                        thrust::dispatchable<InputSystem> &input_system,
                        InputIterator first,
                        size_type n)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  super_t::allocate(n);

  // XXX this copy should actually be copy construct via allocator
  thrust::detail::two_system_copy_n(input_system, system, first, n, super_t::begin());
} // end temporary_array::temporary_array()


template<typename T, typename System>
  template<typename InputIterator>
    temporary_array<T,System>
      ::temporary_array(thrust::dispatchable<System> &system,
                        InputIterator first,
                        InputIterator last)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  super_t::allocate(thrust::distance(system,first,last));

  // XXX this copy should actually be copy construct via allocator
  thrust::copy(system, first, last, super_t::begin());
} // end temporary_array::temporary_array()


template<typename T, typename System>
  template<typename InputSystem, typename InputIterator>
    temporary_array<T,System>
      ::temporary_array(thrust::dispatchable<System> &system,
                        thrust::dispatchable<InputSystem> &input_system,
                        InputIterator first,
                        InputIterator last)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  super_t::allocate(thrust::distance(input_system,first,last));

  // XXX this copy should actually be copy construct via allocator
  thrust::detail::two_system_copy(input_system, system, first, last, super_t::begin());
} // end temporary_array::temporary_array()


} // end detail

} // end thrust

