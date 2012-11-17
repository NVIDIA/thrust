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


/*! \file equal.inl
 *  \brief Inline file for equal.h.
 */

#include <thrust/equal.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/equal.h>
#include <thrust/system/detail/adl/equal.h>

namespace thrust
{


template<typename System, typename InputIterator1, typename InputIterator2>
bool equal(thrust::detail::dispatchable_base<System> &system, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  using thrust::system::detail::generic::equal;
  return equal(thrust::detail::derived_cast(system), first1, last1, first2);
} // end equal()


template<typename System, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
bool equal(thrust::detail::dispatchable_base<System> &system, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::equal;
  return equal(thrust::detail::derived_cast(system), first1, last1, first2, binary_pred);
} // end equal()


namespace detail
{


template<typename System, typename InputIterator1, typename InputIterator2>
bool strip_const_equal(const System &system, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::equal(non_const_system, first1, last1, first2);
} // end equal()


template<typename System, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
bool strip_const_equal(const System &system, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate binary_pred)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::equal(non_const_system, first1, last1, first2, binary_pred);
} // end equal()


} // end detail


template <typename InputIterator1, typename InputIterator2>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;

  System1 system1;
  System2 system2;

  return thrust::detail::strip_const_equal(select_system(system1,system2), first1, last1, first2);
}


template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;

  System1 system1;
  System2 system2;

  return thrust::detail::strip_const_equal(select_system(system1,system2), first1, last1, first2, binary_pred);
}


} // end namespace thrust

