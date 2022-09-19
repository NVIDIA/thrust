/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights meserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <cstdio>
#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

#include <cub/detail/device_synchronize.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>

#include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

inline THRUST_HOST_DEVICE
cudaStream_t
default_stream()
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return cudaStreamPerThread;
#else
  return cudaStreamLegacy;
#endif
}

// Fallback implementation of the customization point.
template <class Derived>
THRUST_HOST_DEVICE
cudaStream_t
get_stream(execution_policy<Derived> &)
{
  return default_stream();
}

// Entry point/interface.
template <class Derived>
THRUST_HOST_DEVICE cudaStream_t
stream(execution_policy<Derived> &policy)
{
  return get_stream(derived_cast(policy));
}


// Fallback implementation of the customization point.
template <class Derived>
THRUST_HOST_DEVICE
bool
must_perform_optional_stream_synchronization(execution_policy<Derived> &)
{
  return true;
}

// Entry point/interface.
template <class Derived>
THRUST_HOST_DEVICE bool
must_perform_optional_synchronization(execution_policy<Derived> &policy)
{
  return must_perform_optional_stream_synchronization(derived_cast(policy));
}


// Fallback implementation of the customization point.
__thrust_exec_check_disable__
template <class Derived>
THRUST_HOST_DEVICE
cudaError_t
synchronize_stream(execution_policy<Derived> &policy)
{
  return cub::SyncStream(stream(policy));
}

// Entry point/interface.
template <class Policy>
THRUST_HOST_DEVICE
cudaError_t
synchronize(Policy &policy)
{
  return synchronize_stream(derived_cast(policy));
}

// Fallback implementation of the customization point.
__thrust_exec_check_disable__
template <class Derived>
THRUST_HOST_DEVICE
cudaError_t
synchronize_stream_optional(execution_policy<Derived> &policy)
{
  cudaError_t result;

  if (must_perform_optional_synchronization(policy))
  {
    result = synchronize_stream(policy);
  }
  else
  {
    result = cudaSuccess;
  }

  return result;
}

// Entry point/interface.
template <class Policy>
THRUST_HOST_DEVICE
cudaError_t
synchronize_optional(Policy &policy)
{
  return synchronize_stream_optional(derived_cast(policy));
}

template <class Type>
THRUST_HOST_FUNCTION cudaError_t
trivial_copy_from_device(Type *       dst,
                         Type const * src,
                         size_t       count,
                         cudaStream_t stream)
{
  cudaError status = cudaSuccess;
  if (count == 0) return status;

  status = ::cudaMemcpyAsync(dst,
                             src,
                             sizeof(Type) * count,
                             cudaMemcpyDeviceToHost,
                             stream);
  cudaStreamSynchronize(stream);
  return status;
}

template <class Type>
THRUST_HOST_FUNCTION cudaError_t
trivial_copy_to_device(Type *       dst,
                       Type const * src,
                       size_t       count,
                       cudaStream_t stream)
{
  cudaError status = cudaSuccess;
  if (count == 0) return status;

  status = ::cudaMemcpyAsync(dst,
                             src,
                             sizeof(Type) * count,
                             cudaMemcpyHostToDevice,
                             stream);
  cudaStreamSynchronize(stream);
  return status;
}

template <class Policy, class Type>
THRUST_HOST_DEVICE cudaError_t
trivial_copy_device_to_device(Policy &    policy,
                              Type *      dst,
                              Type const *src,
                              size_t      count)
{
  cudaError_t  status = cudaSuccess;
  if (count == 0) return status;

  cudaStream_t stream = cuda_cub::stream(policy);
  //
  status = ::cudaMemcpyAsync(dst,
                             src,
                             sizeof(Type) * count,
                             cudaMemcpyDeviceToDevice,
                             stream);
  cuda_cub::synchronize(policy);
  return status;
}

inline void THRUST_HOST_DEVICE
terminate()
{
  NV_IF_TARGET(NV_IS_HOST, (std::terminate();), (asm("trap;");));
}

__host__  __device__
inline void throw_on_error(cudaError_t status)
{
  // Clear the global CUDA error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.
#ifdef THRUST_RDC_ENABLED
  cudaGetLastError();
#else
  NV_IF_TARGET(NV_IS_HOST, (cudaGetLastError();));
#endif

  if (cudaSuccess != status)
  {

    // Can't use #if inside NV_IF_TARGET, use a temp macro to hoist the device
    // instructions out of the target logic.
#ifdef THRUST_RDC_ENABLED

#define THRUST_TEMP_DEVICE_CODE \
  printf("Thrust CUDA backend error: %s: %s\n", \
         cudaGetErrorName(status), \
         cudaGetErrorString(status))

#else

#define THRUST_TEMP_DEVICE_CODE \
  printf("Thrust CUDA backend error: %d\n", \
         static_cast<int>(status))

#endif

    NV_IF_TARGET(NV_IS_HOST, (
      throw thrust::system_error(status, thrust::cuda_category());
    ), (
      THRUST_TEMP_DEVICE_CODE;
      cuda_cub::terminate();
    ));

#undef THRUST_TEMP_DEVICE_CODE

  }
}

THRUST_HOST_DEVICE
inline void throw_on_error(cudaError_t status, char const *msg)
{
  // Clear the global CUDA error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.
#ifdef THRUST_RDC_ENABLED
  cudaGetLastError();
#else
  NV_IF_TARGET(NV_IS_HOST, (cudaGetLastError();));
#endif

  if (cudaSuccess != status)
  {
    // Can't use #if inside NV_IF_TARGET, use a temp macro to hoist the device
    // instructions out of the target logic.
#ifdef THRUST_RDC_ENABLED

#define THRUST_TEMP_DEVICE_CODE \
  printf("Thrust CUDA backend error: %s: %s: %s\n", \
         cudaGetErrorName(status), \
         cudaGetErrorString(status),\
         msg)

#else

#define THRUST_TEMP_DEVICE_CODE \
  printf("Thrust CUDA backend error: %d: %s\n", \
         static_cast<int>(status),              \
         msg)

#endif

    NV_IF_TARGET(NV_IS_HOST, (
      throw thrust::system_error(status, thrust::cuda_category(), msg);
    ), (
      THRUST_TEMP_DEVICE_CODE;
      cuda_cub::terminate();
    ));

#undef THRUST_TEMP_DEVICE_CODE

  }
}

// FIXME: Move the iterators elsewhere.

template <class ValueType,
          class InputIt,
          class UnaryOp>
struct transform_input_iterator_t
{
  typedef transform_input_iterator_t                         self_t;
  typedef typename iterator_traits<InputIt>::difference_type difference_type;
  typedef ValueType                                          value_type;
  typedef void                                               pointer;
  typedef value_type                                         reference;
  typedef std::random_access_iterator_tag                    iterator_category;

  InputIt         input;
  mutable UnaryOp op;

  THRUST_HOST_DEVICE __forceinline__
  transform_input_iterator_t(InputIt input, UnaryOp op)
      : input(input), op(op) {}

#if THRUST_CPP_DIALECT >= 2011
  transform_input_iterator_t(const self_t &) = default;
#endif

  // UnaryOp might not be copy assignable, such as when it is a lambda.  Define
  // an explicit copy assignment operator that doesn't try to assign it.
  self_t& operator=(const self_t& o)
  {
    input = o.input;
    return *this;
  }

  /// Postfix increment
  THRUST_HOST_DEVICE __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++input;
    return retval;
  }

  /// Prefix increment
  THRUST_HOST_DEVICE __forceinline__ self_t operator++()
  {
    ++input;
    return *this;
  }

  /// Indirection
  THRUST_HOST_DEVICE __forceinline__ reference operator*() const
  {
    typename thrust::iterator_value<InputIt>::type x = *input;
    return op(x);
  }
  /// Indirection
  THRUST_HOST_DEVICE __forceinline__ reference operator*()
  {
    typename thrust::iterator_value<InputIt>::type x = *input;
    return op(x);
  }

  /// Addition
  THRUST_HOST_DEVICE __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(input + n, op);
  }

  /// Addition assignment
  THRUST_HOST_DEVICE __forceinline__ self_t &operator+=(difference_type n)
  {
    input += n;
    return *this;
  }

  /// Subtraction
  THRUST_HOST_DEVICE __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(input - n, op);
  }

  /// Subtraction assignment
  THRUST_HOST_DEVICE __forceinline__ self_t &operator-=(difference_type n)
  {
    input -= n;
    return *this;
  }

  /// Distance
  THRUST_HOST_DEVICE __forceinline__ difference_type operator-(self_t other) const
  {
    return input - other.input;
  }

  /// Array subscript
  THRUST_HOST_DEVICE __forceinline__ reference operator[](difference_type n) const
  {
    return op(input[n]);
  }

  /// Equal to
  THRUST_HOST_DEVICE __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (input == rhs.input);
  }

  /// Not equal to
  THRUST_HOST_DEVICE __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (input != rhs.input);
  }
};    // struct transform_input_iterarot_t

template <class ValueType,
          class InputIt1,
          class InputIt2,
          class BinaryOp>
struct transform_pair_of_input_iterators_t
{
  typedef transform_pair_of_input_iterators_t                 self_t;
  typedef typename iterator_traits<InputIt1>::difference_type difference_type;
  typedef ValueType                                           value_type;
  typedef void                                                pointer;
  typedef value_type                                          reference;
  typedef std::random_access_iterator_tag                     iterator_category;

  InputIt1         input1;
  InputIt2         input2;
  mutable BinaryOp op;

  THRUST_HOST_DEVICE __forceinline__
  transform_pair_of_input_iterators_t(InputIt1 input1_,
                                      InputIt2 input2_,
                                      BinaryOp op_)
      : input1(input1_), input2(input2_), op(op_) {}

#if THRUST_CPP_DIALECT >= 2011
  transform_pair_of_input_iterators_t(const self_t &) = default;
#endif

  // BinaryOp might not be copy assignable, such as when it is a lambda.
  // Define an explicit copy assignment operator that doesn't try to assign it.
  self_t& operator=(const self_t& o)
  {
    input1 = o.input1;
    input2 = o.input2;
    return *this;
  }

  /// Postfix increment
  THRUST_HOST_DEVICE __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++input1;
    ++input2;
    return retval;
  }

  /// Prefix increment
  THRUST_HOST_DEVICE __forceinline__ self_t operator++()
  {
    ++input1;
    ++input2;
    return *this;
  }

  /// Indirection
  THRUST_HOST_DEVICE __forceinline__ reference operator*() const
  {
    return op(*input1, *input2);
  }
  /// Indirection
  THRUST_HOST_DEVICE __forceinline__ reference operator*()
  {
    return op(*input1, *input2);
  }

  /// Addition
  THRUST_HOST_DEVICE __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(input1 + n, input2 + n, op);
  }

  /// Addition assignment
  THRUST_HOST_DEVICE __forceinline__ self_t &operator+=(difference_type n)
  {
    input1 += n;
    input2 += n;
    return *this;
  }

  /// Subtraction
  THRUST_HOST_DEVICE __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(input1 - n, input2 - n, op);
  }

  /// Subtraction assignment
  THRUST_HOST_DEVICE __forceinline__ self_t &operator-=(difference_type n)
  {
    input1 -= n;
    input2 -= n;
    return *this;
  }

  /// Distance
  THRUST_HOST_DEVICE __forceinline__ difference_type operator-(self_t other) const
  {
    return input1 - other.input1;
  }

  /// Array subscript
  THRUST_HOST_DEVICE __forceinline__ reference operator[](difference_type n) const
  {
    return op(input1[n], input2[n]);
  }

  /// Equal to
  THRUST_HOST_DEVICE __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (input1 == rhs.input1) && (input2 == rhs.input2);
  }

  /// Not equal to
  THRUST_HOST_DEVICE __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (input1 != rhs.input1) || (input2 != rhs.input2);
  }

};    // struct transform_pair_of_input_iterators_t


struct identity
{
  template <class T>
  THRUST_HOST_DEVICE T const &
  operator()(T const &t) const
  {
    return t;
  }

  template <class T>
  THRUST_HOST_DEVICE T &
  operator()(T &t) const
  {
    return t;
  }
};


template <class T>
struct counting_iterator_t
{
  typedef counting_iterator_t             self_t;
  typedef T                               difference_type;
  typedef T                               value_type;
  typedef void                            pointer;
  typedef T                               reference;
  typedef std::random_access_iterator_tag iterator_category;

  T count;

  THRUST_HOST_DEVICE __forceinline__
  counting_iterator_t(T count_) : count(count_) {}

  /// Postfix increment
  THRUST_HOST_DEVICE __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++count;
    return retval;
  }

  /// Prefix increment
  THRUST_HOST_DEVICE __forceinline__ self_t operator++()
  {
    ++count;
    return *this;
  }

  /// Indirection
  THRUST_HOST_DEVICE __forceinline__ reference operator*() const
  {
    return count;
  }

  /// Indirection
  THRUST_HOST_DEVICE __forceinline__ reference operator*()
  {
    return count;
  }

  /// Addition
  THRUST_HOST_DEVICE __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(count + n);
  }

  /// Addition assignment
  THRUST_HOST_DEVICE __forceinline__ self_t &operator+=(difference_type n)
  {
    count += n;
    return *this;
  }

  /// Subtraction
  THRUST_HOST_DEVICE __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(count - n);
  }

  /// Subtraction assignment
  THRUST_HOST_DEVICE __forceinline__ self_t &operator-=(difference_type n)
  {
    count -= n;
    return *this;
  }

  /// Distance
  THRUST_HOST_DEVICE __forceinline__ difference_type operator-(self_t other) const
  {
    return count - other.count;
  }

  /// Array subscript
  THRUST_HOST_DEVICE __forceinline__ reference operator[](difference_type n) const
  {
    return count + n;
  }

  /// Equal to
  THRUST_HOST_DEVICE __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (count == rhs.count);
  }

  /// Not equal to
  THRUST_HOST_DEVICE __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (count != rhs.count);
  }

};    // struct count_iterator_t

}    // cuda_

THRUST_NAMESPACE_END
