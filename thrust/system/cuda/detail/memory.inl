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
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/copy.h>
#include <thrust/swap.h>
#include <limits>

#include <cuda_runtime_api.h>
#include <thrust/system/cuda_error.h>
#include <thrust/system/detail/bad_alloc.h>

namespace thrust
{

namespace detail
{
namespace backend
{
namespace cuda
{

struct cuda_to_cpp  {};
struct cpp_to_cuda  {};

__host__ __device__
inline tag select_system(tag, tag)
{
  return tag();
}

__host__ __device__
inline cuda_to_cpp select_system(cuda::tag, thrust::system::cpp::tag)
{
  return cuda_to_cpp();
}

__host__ __device__
inline cpp_to_cuda select_system(thrust::system::cpp::tag, cuda::tag)
{
  return cpp_to_cuda();
}

// XXX malloc should be moved into thrust::system::cuda::detail
inline thrust::system::cuda::pointer<void> malloc(tag, std::size_t n)
{
  void *result = 0;

  cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&result), n);

  if(error)
  {
    throw thrust::system::detail::bad_alloc(thrust::system::cuda_category().message(error).c_str());
  } // end if

  return thrust::system::cuda::pointer<void>(result);
} // end malloc()

// XXX free should be moved into thrust::system::cuda::detail
inline void free(tag, thrust::system::cuda::pointer<void> ptr)
{
  cudaError_t error = cudaFree(ptr.get());

  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category());
  } // end error
} // end free()

// XXX assign_value should be moved into thrust::system::cpp::detail
template<typename Pointer1, typename Pointer2>
__host__ __device__
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
      // when called from device code, just do simple deref & assign
      // XXX consider deferring to assign_value(cpp::tag, dst, src) here
      *thrust::detail::pointer_traits<Pointer1>::get(dst)
        = *thrust::detail::pointer_traits<Pointer2>::get(src);
    }
  };

#ifndef __CUDA_ARCH__
  return war_nvbugs_881631::host_path(dst,src);
#else
  return war_nvbugs_881631::device_path(dst,src);
#endif // __CUDA_ARCH__
} // end assign_value()

template<typename Pointer1, typename Pointer2>
  void assign_value(cpp_to_cuda, Pointer1 dst, Pointer2 src)
{
  thrust::copy(src, src + 1, dst);
} // end assign_value()

template<typename Pointer1, typename Pointer2>
  void assign_value(cuda_to_cpp, Pointer1 dst, Pointer2 src)
{
  thrust::copy(src, src + 1, dst);
} // end assign_value()

// XXX get_value should be moved into thrust::system::cuda::detail
template<typename Pointer>
__host__ __device__
  typename thrust::iterator_value<Pointer>::type
    get_value(tag, Pointer ptr)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static typename thrust::iterator_value<Pointer>::type host_path(Pointer ptr)
    {
      // when called from host code, implement with assign_value
      // note that this requires a type with default constructor
      typename thrust::iterator_value<Pointer>::type result;
      assign_value(cuda_to_cpp(), &result, ptr);
      return result;
    }

    __device__ inline static typename thrust::iterator_value<Pointer>::type device_path(Pointer ptr)
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
} // end get_value()

// XXX iter_swap should be moved into thrust::system::cuda::detail
template<typename Pointer1, typename Pointer2>
__host__ __device__
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

} // end cuda
} // end backend
} // end detail

namespace system
{
namespace cuda
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
  return thrust::detail::backend::cuda::malloc(tag(), n);
} // end malloc()

void free(pointer<void> ptr)
{
  return thrust::detail::backend::cuda::free(tag(), ptr);
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

namespace backend
{

// specialize dereference_result and overload dereference
template<typename> struct dereference_result;

template<typename T>
  struct dereference_result<thrust::cuda::pointer<T> >
{
  typedef T& type;
}; // end dereference_result

template<typename T>
  struct dereference_result<thrust::cuda::pointer<const T> >
{
  typedef const T& type;
}; // end dereference_result

template<typename T>
  typename dereference_result< thrust::cuda::pointer<T> >::type
    dereference(thrust::cuda::pointer<T> ptr)
{
  return *ptr.get();
} // end dereference()

template<typename T, typename IndexType>
  typename dereference_result< thrust::cuda::pointer<T> >::type
    dereference(thrust::cuda::pointer<T> ptr, IndexType n)
{
  return ptr.get()[n];
} // end dereference()

} // end backend
} // end detail

} // end thrust

