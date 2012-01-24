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

#include <thrust/detail/dispatch/is_trivial_copy.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/detail/internal/scalar/general_copy.h>
#include <thrust/system/detail/internal/scalar/trivial_copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace internal
{
namespace scalar
{
namespace copy_detail
{


// returns the raw pointer associated with a Pointer-like thing
template<typename Pointer>
  typename thrust::detail::pointer_traits<Pointer>::raw_pointer
    get(Pointer ptr)
{
  return thrust::detail::pointer_traits<Pointer>::get(ptr);
}


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::true_type)  // is_trivial_copy
{
  typedef typename thrust::iterator_difference<InputIterator>::type Size;

  const Size n = last - first;
  thrust::system::detail::internal::scalar::trivial_copy_n(get(&*first), n, get(&*result));
  return result + n;
} // end copy()


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::false_type)  // is_trivial_copy
{
  return thrust::system::detail::internal::scalar::general_copy(first,last,result);
} // end copy()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::true_type)  // is_trivial_copy
{
  thrust::system::detail::internal::scalar::trivial_copy_n(get(&*first), n, get(&*result));
  return result + n;
} // end copy_n()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::false_type)  // is_trivial_copy
{
  return thrust::system::detail::internal::scalar::general_copy_n(first,n,result);
} // end copy_n()

} // end namespace copy_detail


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  return thrust::system::detail::internal::scalar::copy_detail::copy(first, last, result,
    typename thrust::detail::dispatch::is_trivial_copy<InputIterator,OutputIterator>::type());
} // end copy()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result)
{
  return thrust::system::detail::internal::scalar::copy_detail::copy_n(first, n, result,
    typename thrust::detail::dispatch::is_trivial_copy<InputIterator,OutputIterator>::type());
} // end copy_n()

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

