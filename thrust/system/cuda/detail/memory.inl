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

// XXX malloc should be moved into thrust::system::cuda::detail
thrust::system::cuda::pointer<void> malloc(tag, std::size_t n)
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
void free(tag, thrust::system::cuda::pointer<void> ptr)
{
  cudaError_t error = cudaFree(ptr.get());

  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category());
  } // end error
} // end free()

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
void swap(reference<T> &a, reference<T> &b)
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

