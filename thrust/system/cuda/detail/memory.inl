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
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/detail/malloc_and_free.h>
#include <thrust/detail/copy.h>
#include <thrust/swap.h>
#include <thrust/detail/raw_pointer_cast.h>
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
  struct pointer_raw_pointer< thrust::cuda::pointer<T> >
{
  typedef typename thrust::cuda::pointer<T>::raw_pointer type;
}; // end pointer_raw_pointer

} // end detail
#endif

namespace system
{
namespace cuda
{
namespace detail
{

// XXX WAR an issue with msvc 2005 (cl v14.00) which creates multiply-defined
//     symbols resulting from assign_value
#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER) && (_MSC_VER <= 1400)

namespace
{

template<typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value_msvc2005_war(Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(Pointer1 dst, Pointer2 src)
    {
      thrust::copy(src, src + 1, dst);
    }

    __device__ inline static void device_path(Pointer1 dst, Pointer2 src)
    {
      *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
    }
  };

#ifndef __CUDA_ARCH__
  war_nvbugs_881631::host_path(dst,src);
#else
  war_nvbugs_881631::device_path(dst,src);
#endif // __CUDA_ARCH__
} // end assign_value_msvc2005_war()

} // end anon namespace

template<typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(cuda::tag, Pointer1 dst, Pointer2 src)
{
  return assign_value_msvc2005_war(dst,src);
} // end assign_value()

#else

template<typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(cuda::tag, Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(Pointer1 dst, Pointer2 src)
    {
      thrust::copy(src, src + 1, dst);
    }

    __device__ inline static void device_path(Pointer1 dst, Pointer2 src)
    {
      *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
    }
  };

#ifndef __CUDA_ARCH__
  war_nvbugs_881631::host_path(dst,src);
#else
  war_nvbugs_881631::device_path(dst,src);
#endif // __CUDA_ARCH__
} // end assign_value()

#endif // msvc 2005 WAR


template<typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(cpp_to_cuda, Pointer1 dst, Pointer2 src)
{
#if __CUDA_ARCH__
  thrust::system::cuda::detail::assign_value(cuda::tag(), dst, src);
#else
  thrust::copy(src, src + 1, dst);
#endif
} // end assign_value()

template<typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(cuda_to_cpp, Pointer1 dst, Pointer2 src)
{
#if __CUDA_ARCH__
  thrust::system::cuda::detail::assign_value(cuda::tag(), dst, src);
#else
  thrust::copy(src, src + 1, dst);
#endif
} // end assign_value()

namespace
{

template<typename Pointer>
inline __host__ __device__
  typename thrust::iterator_value<Pointer>::type
    get_value_msvc2005_war(Pointer ptr)
{
  typedef typename thrust::iterator_value<Pointer>::type result_type;

  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static result_type host_path(Pointer ptr)
    {
      // when called from host code, implement with assign_value
      // note that this requires a type with default constructor
      result_type result;
      assign_value(cuda_to_cpp(), &result, ptr);
      return result;
    }

    __device__ inline static result_type device_path(Pointer ptr)
    {
      // when called from device code, just do simple deref
      // XXX consider deferring to get_value(cpp::tag, ptr) here
      return *thrust::detail::pointer_traits<Pointer>::get(ptr);
    }
  };

#ifndef __CUDA_ARCH__
  return war_nvbugs_881631::host_path(ptr);
#else
  return war_nvbugs_881631::device_path(ptr);
#endif // __CUDA_ARCH__
} // end get_value_msvc2005_war()

} // end anon namespace

template<typename Pointer>
inline __host__ __device__
  typename thrust::iterator_value<Pointer>::type
    get_value(tag, Pointer ptr)
{
  return get_value_msvc2005_war(ptr);
} // end get_value()

template<typename Pointer1, typename Pointer2>
inline __host__ __device__
void iter_swap(tag, Pointer1 a, Pointer2 b)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(Pointer1 a, Pointer2 b)
    {
      thrust::swap_ranges(a, a + 1, b);
    }

    __device__ inline static void device_path(Pointer1 a, Pointer2 b)
    {
      using thrust::swap;
      swap(*thrust::detail::pointer_traits<Pointer1>::get(a),
           *thrust::detail::pointer_traits<Pointer2>::get(b));
    }
  };

#ifndef __CUDA_ARCH__
  return war_nvbugs_881631::host_path(a,b);
#else
  return war_nvbugs_881631::device_path(a,b);
#endif // __CUDA_ARCH__
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
  return pointer<void>(thrust::system::cuda::detail::malloc(tag(), n));
} // end malloc()

void free(pointer<void> ptr)
{
  return thrust::system::cuda::detail::free(tag(), ptr.get());
} // end free()

} // end cuda
} // end system

namespace detail
{

// XXX iterator_facade tries to instantiate the Reference
//     type when computing the answer to is_convertible<Reference,Value>
//     we can't do that at that point because reference
//     is not complete
//     WAR the problem by specializing is_convertible
template<typename T>
  struct is_convertible<thrust::cuda::reference<T>, T>
    : thrust::detail::true_type
{};

} // end detail
} // end thrust

