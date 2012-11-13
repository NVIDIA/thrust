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

/*! \file merge.inl
 *  \brief Inline file for merge.h.
 */

#include <thrust/merge.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/merge.h>
#include <thrust/system/detail/adl/merge.h>

namespace thrust
{


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator merge(thrust::detail::dispatchable_base<System> &system,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result)
{
  using thrust::system::detail::generic::merge;
  return merge(system.derived(), first1, last1, first2, last2, result);
} // end merge()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator merge(thrust::detail::dispatchable_base<System> &system,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result,
                       StrictWeakCompare comp)
{
  using thrust::system::detail::generic::merge;
  return merge(system.derived(), first1, last1, first2, last2, result, comp);
} // end merge()


template<typename System, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(thrust::detail::dispatchable_base<System> &system,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result)
{
  using thrust::system::detail::generic::merge_by_key;
  return merge_by_key(system.derived(), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
} // end merge_by_key()


template<typename System, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2, typename Compare>
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(thrust::detail::dispatchable_base<System> &system,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result,
                 Compare comp)
{
  using thrust::system::detail::generic::merge_by_key;
  return merge_by_key(system.derived(), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
} // end merge_by_key()


namespace detail
{


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator strip_const_merge(const System &system,
                                   InputIterator1 first1,
                                   InputIterator1 last1,
                                   InputIterator2 first2,
                                   InputIterator2 last2,
                                   OutputIterator result)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::merge(non_const_system, first1, last1, first2, last2, result);
} // end strip_const_merge()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator strip_const_merge(const System &system,
                                   InputIterator1 first1,
                                   InputIterator1 last1,
                                   InputIterator2 first2,
                                   InputIterator2 last2,
                                   OutputIterator result,
                                   StrictWeakCompare comp)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::merge(non_const_system, first1, last1, first2, last2, result, comp);
} // end strip_const_merge()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    strip_const_merge_by_key(const System &system,
                             InputIterator1 keys_first1,
                             InputIterator1 keys_last1,
                             InputIterator2 keys_first2,
                             InputIterator2 keys_last2,
                             InputIterator3 values_first1,
                             InputIterator4 values_first2,
                             OutputIterator1 keys_result,
                             OutputIterator2 values_result)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::merge_by_key(non_const_system, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
} // end strip_const_merge_by_key()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Compare>
  thrust::pair<OutputIterator1,OutputIterator2>
    strip_const_merge_by_key(const System &system,
                             InputIterator1 keys_first1,
                             InputIterator1 keys_last1,
                             InputIterator2 keys_first2,
                             InputIterator2 keys_last2,
                             InputIterator3 values_first1,
                             InputIterator4 values_first2,
                             OutputIterator1 keys_result,
                             OutputIterator2 values_result,
                             Compare comp)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::merge_by_key(non_const_system, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
} // end strip_const_merge_by_key()


} // end detail


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator merge(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result,
                       StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::detail::strip_const_merge(select_system(system1,system2,system3), first1, last1, first2, last2, result, comp);
} // end merge()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator merge(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::detail::strip_const_merge(select_system(system1,system2,system3), first1, last1, first2, last2, result);
} // end merge()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakOrdering>
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(InputIterator1 keys_first1,
                 InputIterator1 keys_last1,
                 InputIterator2 keys_first2,
                 InputIterator2 keys_last2,
                 InputIterator3 values_first1,
                 InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result,
                 StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<InputIterator4>::type  System4;
  typedef typename thrust::iterator_system<OutputIterator1>::type System5;
  typedef typename thrust::iterator_system<OutputIterator2>::type System6;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::detail::strip_const_merge_by_key(select_system(system1,system2,system3,system4,system5,system6), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
} // end merge_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(InputIterator1 keys_first1,
                 InputIterator1 keys_last1,
                 InputIterator2 keys_first2,
                 InputIterator2 keys_last2,
                 InputIterator3 values_first1,
                 InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<InputIterator4>::type  System4;
  typedef typename thrust::iterator_system<OutputIterator1>::type System5;
  typedef typename thrust::iterator_system<OutputIterator2>::type System6;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::detail::strip_const_merge_by_key(select_system(system1,system2,system3,system4,system5,system6), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
} // end merge_by_key()


} // end thrust

