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
#include <thrust/system/cuda/detail/cub/util_arch.cuh>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

THRUST_BEGIN_NS

namespace cuda_cub {

template <class Policy>
__host__ __device__
cudaError_t
synchronize(Policy &policy)
{
  return synchronize_stream(derived_cast(policy));
}

template <class Derived>
__host__ __device__ cudaStream_t
stream(execution_policy<Derived> &policy)
{
  return get_stream(derived_cast(policy));
}


#if 0
template <class Policy, class Type>
CUB_RUNTIME_FUNCTION cudaError_t
trivial_copy_from_device(Policy &    policy,
                         Type *      dst,
                         Type const *src,
                         size_t      count)
{
  cudaError status = cudaSuccess;
  if (count == 0) return status;
#ifdef __CUDA_ARCH__
  for (size_t i = 0; i != count; ++i)
  {
    dst[i] = src[i];
  }
#else
  cudaStream_t stream = cuda_cub::stream(policy);
  //
  status = ::cudaMemcpyAsync(dst,
                             src,
                             sizeof(Type) * count,
                             cudaMemcpyDeviceToHost,
                             stream);
  cuda_cub::synchronize(policy);

#endif
  return status;
}
#endif

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

#if 0
template <class Policy, class Type>
CUB_RUNTIME_FUNCTION cudaError_t
trivial_copy_to_device(Policy &    ,
                       Type *      dst,
                       Type const *src,
                       size_t      count)
{
  cudaError status = cudaSuccess;
  if (count == 0) return status;
#ifdef __CUDA_ARCH__
  for (size_t i = 0; i != count; ++i)
  {
    dst[i] = src[i];
  }
#else
  cudaStream_t stream = cuda_cub::stream(policy);
  //
  status = ::cudaMemcpyAsync(dst,
                             src,
                             sizeof(Type) * count,
                             cudaMemcpyHostToDevice,
                             stream);
  cuda_cub::synchronize(policy);
#endif
  return status;
}
#else
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
#endif


template <class Policy, class Type>
__host__ __device__ cudaError_t
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


inline void __host__ __device__
terminate()
{
#ifdef __CUDA_ARCH__
  asm("trap;");
#else
  std::terminate();
#endif
}

__host__  __device__
inline void throw_on_error(cudaError_t status)
{
  if (cudaSuccess != status)
  {
#if !defined(__CUDA_ARCH__)
    throw thrust::system_error(status, thrust::cuda_category());
#else
#if __THRUST_HAS_CUDART__
    printf("Thrust CUDA backend error: %s\n",
           cudaGetErrorString(status));
#else
    printf("Thrust CUDA backend error: %d\n",
           static_cast<int>(status));
#endif
    cuda_cub::terminate();
#endif
  }
}

__host__ __device__ 
inline void throw_on_error(cudaError_t status, char const *msg)
{
  if (cudaSuccess != status)
  {
#if !defined(__CUDA_ARCH__)
    throw thrust::system_error(status, thrust::cuda_category(), msg);
#else
#if __THRUST_HAS_CUDART__
    printf("Thrust CUDA backend error: %s: %s\n",
           cudaGetErrorString(status),
           msg);
#else
    printf("Thrust CUDA backend error: %d: %s \n",
           static_cast<int>(status),
           msg);
#endif
    cuda_cub::terminate();
#endif
  }
}

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

  __host__ __device__ __forceinline__
  transform_input_iterator_t(InputIt input, UnaryOp op)
      : input(input), op(op) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++input;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_t operator++()
  {
    ++input;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const
  {
    typename thrust::iterator_value<InputIt>::type x = *input;
    return op(x);
  }
  /// Indirection
  __host__ __device__ __forceinline__ reference operator*()
  {
    typename thrust::iterator_value<InputIt>::type x = *input;
    return op(x);
  }

  /// Addition
  __host__ __device__ __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(input + n, op);
  }

  /// Addition assignment
  __host__ __device__ __forceinline__ self_t &operator+=(difference_type n)
  {
    input += n;
    return *this;
  }

  /// Subtraction
  __host__ __device__ __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(input - n, op);
  }

  /// Subtraction assignment
  __host__ __device__ __forceinline__ self_t &operator-=(difference_type n)
  {
    input -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_t other) const
  {
    return input - other.input;
  }

  /// Array subscript
  __host__ __device__ __forceinline__ reference operator[](difference_type n) const
  {
    return op(input[n]);
  }

#if 0
    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &op(*input_itr);
    }
#endif

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (input == rhs.input);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (input != rhs.input);
  }

#if 0
    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self& itr)
    {
        return os;
    }
#endif
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

  __host__ __device__ __forceinline__
  transform_pair_of_input_iterators_t(InputIt1 input1_,
                                      InputIt2 input2_,
                                      BinaryOp op_)
      : input1(input1_), input2(input2_), op(op_) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++input1;
    ++input2;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_t operator++()
  {
    ++input1;
    ++input2;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const
  {
    return op(*input1, *input2);
  }
  /// Indirection
  __host__ __device__ __forceinline__ reference operator*()
  {
    return op(*input1, *input2);
  }

  /// Addition
  __host__ __device__ __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(input1 + n, input2 + n, op);
  }

  /// Addition assignment
  __host__ __device__ __forceinline__ self_t &operator+=(difference_type n)
  {
    input1 += n;
    input2 += n;
    return *this;
  }

  /// Subtraction
  __host__ __device__ __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(input1 - n, input2 - n, op);
  }

  /// Subtraction assignment
  __host__ __device__ __forceinline__ self_t &operator-=(difference_type n)
  {
    input1 -= n;
    input2 -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_t other) const
  {
    return input1 - other.input1;
  }

  /// Array subscript
  __host__ __device__ __forceinline__ reference operator[](difference_type n) const
  {
    return op(input1[n], input2[n]);
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (input1 == rhs.input1) && (input2 == rhs.input2);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (input1 != rhs.input1) || (input2 != rhs.input2);
  }

};    // struct transform_pair_of_input_iterators_t

template <class ValueType,
          class InputIt1,
          class InputIt2,
          class InputIt3,
          class TransformOp>
struct transform_triple_of_input_iterators_t
{
  typedef transform_triple_of_input_iterators_t               self_t;
  typedef typename iterator_traits<InputIt1>::difference_type difference_type;
  typedef ValueType                                           value_type;
  typedef value_type *                                        pointer;
  typedef value_type                                          reference;
  typedef std::random_access_iterator_tag                     iterator_category;

  InputIt1            input1;
  InputIt2            input2;
  InputIt3            input3;
  mutable TransformOp op;

  __host__ __device__ __forceinline__
  transform_triple_of_input_iterators_t(InputIt1    input1_,
                                        InputIt2    input2_,
                                        InputIt3    input3_,
                                        TransformOp op_)
      : input1(input1_), input2(input2_), input3(input3_), op(op_) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++input1;
    ++input2;
    ++input3;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_t operator++()
  {
    ++input1;
    ++input2;
    ++input3;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const
  {
    return op(*input1, *input2, *input3);
  }
  /// Indirection
  __host__ __device__ __forceinline__ reference operator*()
  {
    return op(*input1, *input2, *input3);
  }

  /// Addition
  __host__ __device__ __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(input1 + n, input2 + n, input3 + n, op);
  }

  /// Addition assignment
  __host__ __device__ __forceinline__ self_t &operator+=(difference_type n)
  {
    input1 += n;
    input2 += n;
    input3 += n;
    return *this;
  }

  /// Subtraction
  __host__ __device__ __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(input1 - n, input2 - n, input3 - n, op);
  }

  /// Subtraction assignment
  __host__ __device__ __forceinline__ self_t &operator-=(difference_type n)
  {
    input1 -= n;
    input2 -= n;
    input3 -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_t other) const
  {
    return input1 - other.input1;
  }

  /// Array subscript
  __host__ __device__ __forceinline__ reference operator[](difference_type n) const
  {
    return op(input1[n], input2[n], input3[n]);
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (input1 == rhs.input1) &&
           (input2 == rhs.input2) &&
           (input3 == rhs.input3);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (input1 != rhs.input1) ||
           (input2 != rhs.input2) ||
           (input3 != rhs.input3);
  }

};    // struct trasnform_triple_of_input_iterators_t

struct identity
{
  template <class T>
  __host__ __device__ T const &
  operator()(T const &t) const
  {
    return t;
  }

  template <class T>
  __host__ __device__ T &
  operator()(T &t) const
  {
    return t;
  }
};

template <class ValueType,
          class OutputIt,
          class TransformOp = identity>
struct transform_output_iterator_t
{
  struct proxy_reference
  {
  private:
    OutputIt    output;
    TransformOp op;

  public:
    __host__ __device__
    proxy_reference(OutputIt const &output_, TransformOp op_)
        : output(output_), op(op_) {}

    proxy_reference __host__ __device__
    operator=(ValueType const &x)
    {
      *output = op(x);
      return *this;
    }
  };

  typedef transform_output_iterator_t                         self_t;
  typedef typename iterator_traits<OutputIt>::difference_type difference_type;
  typedef void                                                value_type;
  typedef proxy_reference                                     reference;
  typedef std::output_iterator_tag                            iterator_category;

  OutputIt    output;
  TransformOp op;

  __host__ __device__ __forceinline__
  transform_output_iterator_t(OutputIt output)
      : output(output) {}

  __host__ __device__ __forceinline__
  transform_output_iterator_t(OutputIt output, TransformOp op)
      : output(output), op(op) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++output;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_t operator++()
  {
    ++output;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const
  {
    return proxy_reference(output, op);
  }
  /// Indirection
  __host__ __device__ __forceinline__ reference operator*()
  {
    return proxy_reference(output, op);
  }

  /// Addition
  __host__ __device__ __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(output + n, op);
  }

  /// Addition assignment
  __host__ __device__ __forceinline__ self_t &operator+=(difference_type n)
  {
    output += n;
    return *this;
  }

  /// Subtraction
  __host__ __device__ __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(output - n, op);
  }

  /// Subtraction assignment
  __host__ __device__ __forceinline__ self_t &operator-=(difference_type n)
  {
    output -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_t other) const
  {
    return output - other.output;
  }

  /// Array subscript
  __host__ __device__ __forceinline__ reference operator[](difference_type n) const
  {
    return *(output + n);
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (output == rhs.output);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (output != rhs.output);
  }
};    // struct transform_output_iterator_

template <class T, T VALUE>
struct static_integer_iterator
{
  typedef static_integer_iterator         self_t;
  typedef int                             difference_type;
  typedef T                               value_type;
  typedef T                               reference;
  typedef std::random_access_iterator_tag iterator_category;

  __host__ __device__ __forceinline__
  static_integer_iterator() {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_t operator++(int)
  {
    return *this;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_t operator++()
  {
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const
  {
    return VALUE;
  }
  /// Indirection
  __host__ __device__ __forceinline__ reference operator*()
  {
    return VALUE;
  }

  /// Addition
  __host__ __device__ __forceinline__ self_t operator+(difference_type ) const
  {
    return self_t();
  }

  /// Addition assignment
  __host__ __device__ __forceinline__ self_t &operator+=(difference_type )
  {
    return *this;
  }

  /// Subtraction
  __host__ __device__ __forceinline__ self_t operator-(difference_type ) const
  {
    return self_t();
  }

  /// Subtraction assignment
  __host__ __device__ __forceinline__ self_t &operator-=(difference_type )
  {
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_t ) const
  {
    return 0;
  }

  /// Array subscript
  __host__ __device__ __forceinline__ reference operator[](difference_type ) const
  {
    return VALUE;
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_t &) const
  {
    return true;
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_t &) const
  {
    return false;
  }

};    // struct static_bool_iterator

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

  __host__ __device__ __forceinline__
  counting_iterator_t(T count_) : count(count_) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++count;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_t operator++()
  {
    ++count;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const
  {
    return count;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*()
  {
    return count;
  }

  /// Addition
  __host__ __device__ __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(count + n);
  }

  /// Addition assignment
  __host__ __device__ __forceinline__ self_t &operator+=(difference_type n)
  {
    count += n;
    return *this;
  }

  /// Subtraction
  __host__ __device__ __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(count - n);
  }

  /// Subtraction assignment
  __host__ __device__ __forceinline__ self_t &operator-=(difference_type n)
  {
    count -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_t other) const
  {
    return count - other.count;
  }

  /// Array subscript
  __host__ __device__ __forceinline__ reference operator[](difference_type n) const
  {
    return count + n;
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (count == rhs.count);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (count != rhs.count);
  }

};    // struct count_iterator_t


}    // cuda_

THRUST_END_NS
