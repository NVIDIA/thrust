/*
 *  Copyright 2008-2011 NVIDIA Corporation
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
#include <thrust/system/cpp/memory.h>
#include <thrust/swap.h>
#include <cstdlib> // for malloc & free
#include <limits>

namespace thrust
{

namespace detail
{
namespace backend
{
namespace cpp
{

// XXX malloc should be moved into thrust::system::cpp::detail
inline thrust::system::cpp::pointer<void> malloc(tag, std::size_t n)
{
  void *result = std::malloc(n);

  return thrust::system::cpp::pointer<void>(result);
} // end malloc()

// XXX free should be moved into thrust::system::cpp::detail
inline void free(tag, thrust::system::cpp::pointer<void> ptr)
{
  std::free(ptr.get());
} // end free()

// XXX assign_value should be moved into thrust::system::cpp::detail
template<typename Pointer1, typename Pointer2>
__host__ __device__
  void assign_value(tag, Pointer1 dst, Pointer2 src)
{
  *thrust::detail::pointer_traits<Pointer1>::get(dst)
    = *thrust::detail::pointer_traits<Pointer2>::get(src);
} // end assign_value()

// XXX get_value should be moved into thrust::system::cpp::detail
template<typename Pointer>
__host__ __device__
  typename thrust::iterator_value<Pointer>::type
    get_value(tag, Pointer ptr)
{
  return *thrust::detail::pointer_traits<Pointer>::get(ptr);
} // end get_value()

// XXX iter_swap should be moved into thrust::system::cpp::detail
template<typename Pointer1, typename Pointer2>
__host__ __device__
  void iter_swap(tag, Pointer1 a, Pointer2 b)
{
  using thrust::swap;
  swap(*thrust::detail::pointer_traits<Pointer1>::get(a),
       *thrust::detail::pointer_traits<Pointer2>::get(b));
} // end iter_swap()

} // end cpp
} // end backend
} // end detail

namespace system
{
namespace cpp
{

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
  return thrust::detail::backend::cpp::malloc(tag(), n);
} // end malloc()

void free(pointer<void> ptr)
{
  return thrust::detail::backend::cpp::free(tag(), ptr);
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

namespace backend
{

// specialize dereference_result and overload dereference
template<typename> struct dereference_result;

template<typename T>
  struct dereference_result<thrust::cpp::pointer<T> >
{
  typedef T& type;
}; // end dereference_result

template<typename T>
  struct dereference_result<thrust::cpp::pointer<const T> >
{
  typedef const T& type;
}; // end dereference_result

template<typename T>
  typename dereference_result< thrust::cpp::pointer<T> >::type
    dereference(thrust::cpp::pointer<T> ptr)
{
  return *ptr.get();
} // end dereference()

template<typename T, typename IndexType>
  typename dereference_result< thrust::cpp::pointer<T> >::type
    dereference(thrust::cpp::pointer<T> ptr, IndexType n)
{
  return ptr.get()[n];
} // end dereference()

} // end backend
} // end detail

} // end thrust

