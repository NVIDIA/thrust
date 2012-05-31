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
#include <thrust/detail/temporary_array.h>
#include <memory>

namespace thrust
{
namespace detail
{


// XXX WAR circular dependency with this forward declaration
template<typename Iterator, typename Tag> class move_to_system;


namespace allocator_traits_detail
{


template<typename Allocator, typename InputType, typename OutputType>
  struct copy_construct_with_allocator
{
  Allocator &a;

  copy_construct_with_allocator(Allocator &a)
    : a(a)
  {}

  inline __host__ __device__
  void operator()(thrust::tuple<const InputType&,OutputType&> t)
  {
    const InputType &in = thrust::get<0>(t);
    OutputType &out = thrust::get<1>(t);

    allocator_traits<Allocator>::construct(a, &out, in);
  }
};


// XXX it's regrettable that this implementation is copied almost
//     exactly from system::detail::generic::uninitialized_copy
//     perhaps generic::uninitialized_copy could call this routine
//     with a default allocator
template<typename Allocator, typename InputIterator, typename Pointer>
  Pointer unintialized_copy_with_allocator(Allocator &a, InputIterator first, InputIterator last, Pointer result)
{
  // zip up the iterators
  typedef thrust::tuple<InputIterator,Pointer> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple>  ZipIterator;

  ZipIterator begin = thrust::make_zip_iterator(thrust::make_tuple(first,result));
  ZipIterator end = begin;

  // get a zip_iterator pointing to the end
  const typename thrust::iterator_difference<InputIterator>::type n = thrust::distance(first,last);
  thrust::advance(end, n);

  // create a functor
  typedef typename iterator_traits<InputIterator>::value_type InputType;
  typedef typename iterator_traits<Pointer>::value_type       OutputType;

  // do the for_each
  thrust::for_each(begin, end, copy_construct_with_allocator<Allocator,InputType,OutputType>(a));

  // return the end of the output range
  return thrust::get<1>(end.get_iterator_tuple());
}


template<typename Allocator, typename T>
  struct is_trivially_copy_constructible
    : integral_constant<
        bool,
        !has_member_construct2<Allocator,T,T>::value && has_trivial_copy_constructor<T>::value
      >
{};

// we know that std::allocator::construct's only effect is to
// call T's constructor, so we needn't use it when constructing T
template<typename U, typename T>
  struct is_trivially_copy_constructible<std::allocator<U>, T>
    : has_trivial_copy_constructor<T>
{};


template<typename Allocator, typename InputIterator, typename Pointer>
  typename enable_if<
    is_trivially_copy_constructible<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value,
    Pointer
  >::type
    copy_construct_range(Allocator &, InputIterator first, InputIterator last, Pointer result)
{
  // just call copy
  return thrust::copy(first, last, result);
}


template<typename Allocator, typename InputIterator, typename Pointer>
  typename disable_if<
    is_trivially_copy_constructible<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value,
    Pointer
  >::type
    copy_construct_range(Allocator &a, InputIterator first, InputIterator last, Pointer result)
{
  // move input to the same system as the output
  typedef typename thrust::iterator_system<Pointer>::type system;

  // this is a no-op if both ranges are in the same system
  thrust::detail::move_to_system<InputIterator,system> temp(first,last);

  return unintialized_copy_with_allocator(a, temp.begin(), temp.end(), result);
}


} // end allocator_traits_detail


template<typename Allocator, typename InputIterator, typename Pointer>
  Pointer copy_construct_range(Allocator &a, InputIterator first, InputIterator last, Pointer result)
{
  return allocator_traits_detail::copy_construct_range(a, first, last, result);
}


} // end detail
} // end thrust

