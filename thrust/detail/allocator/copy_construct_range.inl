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

#include <thrust/detail/config.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/copy.h>
#include <thrust/tuple.h>
#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <memory>

namespace thrust
{
namespace detail
{
namespace allocator_traits_detail
{


template<typename Allocator, typename InputType, typename OutputType>
  struct copy_construct_with_allocator
{
  Allocator &a;

  copy_construct_with_allocator(Allocator &a)
    : a(a)
  {}

  template<typename Tuple>
  inline __host__ __device__
  void operator()(Tuple t)
  {
    const InputType &in = thrust::get<0>(t);
    OutputType &out = thrust::get<1>(t);

    allocator_traits<Allocator>::construct(a, &out, in);
  }
};


template<typename Allocator, typename T>
  struct needs_copy_construct_via_allocator
    : has_member_construct2<
        Allocator,
        T,
        T
      >
{};


// we know that std::allocator::construct's only effect is to call T's
// copy constructor, so we needn't use it for copy construction
template<typename U, typename T>
  struct needs_copy_construct_via_allocator<std::allocator<U>, T>
    : thrust::detail::false_type
{};


// XXX it's regrettable that this implementation is copied almost
//     exactly from system::detail::generic::uninitialized_copy
//     perhaps generic::uninitialized_copy could call this routine
//     with a default allocator
template<typename Allocator, typename FromSystem, typename ToSystem, typename InputIterator, typename Pointer>
  typename enable_if_convertible<
    FromSystem,
    ToSystem,
    Pointer
  >::type
    uninitialized_copy_with_allocator(Allocator &a,
                                      thrust::execution_policy<FromSystem> &from_system,
                                      thrust::execution_policy<ToSystem> &to_system,
                                      InputIterator first,
                                      InputIterator last,
                                      Pointer result)
{
  // zip up the iterators
  typedef thrust::tuple<InputIterator,Pointer> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple>  ZipIterator;

  ZipIterator begin = thrust::make_zip_iterator(thrust::make_tuple(first,result));
  ZipIterator end = begin;

  // get a zip_iterator pointing to the end
  const typename thrust::iterator_difference<InputIterator>::type n = thrust::distance(first,last);
  thrust::advance(end,n);

  // create a functor
  typedef typename iterator_traits<InputIterator>::value_type InputType;
  typedef typename iterator_traits<Pointer>::value_type       OutputType;

  // do the for_each
  // note we use to_system to dispatch the for_each
  thrust::for_each(to_system, begin, end, copy_construct_with_allocator<Allocator,InputType,OutputType>(a));

  // return the end of the output range
  return thrust::get<1>(end.get_iterator_tuple());
}


// XXX it's regrettable that this implementation is copied almost
//     exactly from system::detail::generic::uninitialized_copy_n
//     perhaps generic::uninitialized_copy_n could call this routine
//     with a default allocator
template<typename Allocator, typename FromSystem, typename ToSystem, typename InputIterator, typename Size, typename Pointer>
  typename enable_if_convertible<
    FromSystem,
    ToSystem,
    Pointer
  >::type
    uninitialized_copy_with_allocator_n(Allocator &a,
                                        thrust::execution_policy<FromSystem> &from_system,
                                        thrust::execution_policy<ToSystem> &to_system,
                                        InputIterator first,
                                        Size n,
                                        Pointer result)
{
  // zip up the iterators
  typedef thrust::tuple<InputIterator,Pointer> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple>  ZipIterator;

  ZipIterator begin = thrust::make_zip_iterator(thrust::make_tuple(first,result));

  // create a functor
  typedef typename iterator_traits<InputIterator>::value_type InputType;
  typedef typename iterator_traits<Pointer>::value_type       OutputType;

  // do the for_each_n
  // note we use to_system to dispatch the for_each_n
  ZipIterator end = thrust::for_each_n(to_system, begin, n, copy_construct_with_allocator<Allocator,InputType,OutputType>(a));

  // return the end of the output range
  return thrust::get<1>(end.get_iterator_tuple());
}


template<typename Allocator, typename FromSystem, typename ToSystem, typename InputIterator, typename Pointer>
  typename disable_if_convertible<
    FromSystem,
    ToSystem,
    Pointer
  >::type
    uninitialized_copy_with_allocator(Allocator &,
                                      thrust::execution_policy<FromSystem> &from_system,
                                      thrust::execution_policy<ToSystem> &to_system,
                                      InputIterator first,
                                      InputIterator last,
                                      Pointer result)
{
  // the systems aren't trivially interoperable
  // just call two_system_copy and hope for the best
  return thrust::detail::two_system_copy(from_system, to_system, first, last, result);
} // end uninitialized_copy_with_allocator()


template<typename Allocator, typename FromSystem, typename ToSystem, typename InputIterator, typename Size, typename Pointer>
  typename disable_if_convertible<
    FromSystem,
    ToSystem,
    Pointer
  >::type
    uninitialized_copy_with_allocator_n(Allocator &,
                                        thrust::execution_policy<FromSystem> &from_system,
                                        thrust::execution_policy<ToSystem> &to_system,
                                        InputIterator first,
                                        Size n,
                                        Pointer result)
{
  // the systems aren't trivially interoperable
  // just call two_system_copy_n and hope for the best
  return thrust::detail::two_system_copy_n(from_system, to_system, first, n, result);
} // end uninitialized_copy_with_allocator_n()


template<typename FromSystem, typename Allocator, typename InputIterator, typename Pointer>
  typename disable_if<
    needs_copy_construct_via_allocator<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value,
    Pointer
  >::type
    copy_construct_range(thrust::execution_policy<FromSystem> &from_system,
                         Allocator &a,
                         InputIterator first,
                         InputIterator last,
                         Pointer result)
{
  typename allocator_system<Allocator>::type &to_system = allocator_system<Allocator>::get(a);

  // just call two_system_copy
  return thrust::detail::two_system_copy(from_system, to_system, first, last, result);
}


template<typename FromSystem, typename Allocator, typename InputIterator, typename Size, typename Pointer>
  typename disable_if<
    needs_copy_construct_via_allocator<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value,
    Pointer
  >::type
    copy_construct_range_n(thrust::execution_policy<FromSystem> &from_system,
                           Allocator &a,
                           InputIterator first,
                           Size n,
                           Pointer result)
{
  typename allocator_system<Allocator>::type &to_system = allocator_system<Allocator>::get(a);

  // just call two_system_copy_n
  return thrust::detail::two_system_copy_n(from_system, to_system, first, n, result);
}


template<typename FromSystem, typename Allocator, typename InputIterator, typename Pointer>
  typename enable_if<
    needs_copy_construct_via_allocator<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value,
    Pointer
  >::type
    copy_construct_range(thrust::execution_policy<FromSystem> &from_system,
                         Allocator &a,
                         InputIterator first,
                         InputIterator last,
                         Pointer result)
{
  typename allocator_system<Allocator>::type &to_system = allocator_system<Allocator>::get(a);
  return uninitialized_copy_with_allocator(a, from_system, to_system, first, last, result);
}


template<typename FromSystem, typename Allocator, typename InputIterator, typename Size, typename Pointer>
  typename enable_if<
    needs_copy_construct_via_allocator<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value,
    Pointer
  >::type
    copy_construct_range_n(thrust::execution_policy<FromSystem> &from_system,
                           Allocator &a,
                           InputIterator first,
                           Size n,
                           Pointer result)
{
  typename allocator_system<Allocator>::type &to_system = allocator_system<Allocator>::get(a);
  return uninitialized_copy_with_allocator_n(a, from_system, to_system, first, n, result);
}


} // end allocator_traits_detail


template<typename System, typename Allocator, typename InputIterator, typename Pointer>
  Pointer copy_construct_range(thrust::execution_policy<System> &from_system,
                               Allocator &a,
                               InputIterator first,
                               InputIterator last,
                               Pointer result)
{
  return allocator_traits_detail::copy_construct_range(from_system, a, first, last, result);
}


template<typename System, typename Allocator, typename InputIterator, typename Size, typename Pointer>
  Pointer copy_construct_range_n(thrust::execution_policy<System> &from_system,
                                 Allocator &a,
                                 InputIterator first,
                                 Size n,
                                 Pointer result)
{
  return allocator_traits_detail::copy_construct_range_n(from_system, a, first, n, result);
}


} // end detail
} // end thrust

