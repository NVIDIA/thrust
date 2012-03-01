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
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/system/cpp/detail/malloc_and_free.h>
#include <thrust/detail/swap.h>
#include <limits>

namespace thrust
{

// XXX WAR an issue with MSVC 2005 (cl v14.00) incorrectly implementing
//     pointer_raw_pointer for pointer by specializing it here
//     note that we specialize it here, before the use of raw_pointer_cast
//     below, which causes pointer_raw_pointer's instantiation
#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && (_MSC_VER <= 1400)
namespace detail
{

template<typename T>
  struct pointer_raw_pointer< thrust::cpp::pointer<T> >
{
  typedef typename thrust::cpp::pointer<T>::raw_pointer type;
}; // end pointer_raw_pointer

} // end detail
#endif

namespace system
{
namespace cpp
{
namespace detail
{

template<typename Pointer1, typename Pointer2>
__host__ __device__
  void assign_value(tag, Pointer1 dst, Pointer2 src)
{
  *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
} // end assign_value()

template<typename Pointer>
__host__ __device__
  typename thrust::iterator_value<Pointer>::type
    get_value(tag, Pointer ptr)
{
  return *thrust::raw_pointer_cast(ptr);
} // end get_value()

template<typename Pointer1, typename Pointer2>
__host__ __device__
  void iter_swap(tag, Pointer1 a, Pointer2 b)
{
  using thrust::swap;
  swap(*thrust::raw_pointer_cast(a), *thrust::raw_pointer_cast(b));
} // end iter_swap()

} // end detail


template<typename T>
  template<typename OtherT>
    reference<T> &
      reference<T>
        ::operator=(const reference<OtherT> &other)
{
  return super_t::operator=(other);
} // end reference::operator=()

template<typename T>
  reference<T> &
    reference<T>
      ::operator=(const value_type &x)
{
  return super_t::operator=(x);
} // end reference::operator=()

template<typename T>
__host__ __device__
void swap(reference<T> a, reference<T> b)
{
  a.swap(b);
} // end swap()

pointer<void> malloc(std::size_t n)
{
  return pointer<void>(thrust::system::cpp::detail::malloc(tag(), n));
} // end malloc()

void free(pointer<void> ptr)
{
  return thrust::system::cpp::detail::free(tag(), ptr);
} // end free()

} // end cpp
} // end system

namespace detail
{

// XXX iterator_facade tries to instantiate the Reference
//     type when computing the answer to is_convertible<Reference,Value>
//     we can't do that at that point because reference
//     is not complete
//     WAR the problem by specializing is_convertible
template<typename T>
  struct is_convertible<thrust::cpp::reference<T>, T>
    : thrust::detail::true_type
{};

} // end detail
} // end thrust

