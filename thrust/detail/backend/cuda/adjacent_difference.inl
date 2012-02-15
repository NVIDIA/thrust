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

#include <thrust/gather.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/detail/backend_iterator_spaces.h>

#include <thrust/detail/uninitialized_array.h>

#include <thrust/detail/backend/decompose.h>

#include <thrust/detail/backend/cuda/default_decomposition.h>
#include <thrust/detail/backend/cuda/detail/launch_closure.h>
#include <thrust/detail/backend/cuda/detail/launch_calculator.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace detail
{

template <typename Decomposition>
struct last_index_in_each_interval : public thrust::unary_function<typename Decomposition::index_type, typename Decomposition::index_type>
{
  typedef typename Decomposition::index_type index_type;

  Decomposition decomp;

  last_index_in_each_interval(Decomposition decomp) : decomp(decomp) {}

  __host__ __device__
  index_type operator()(index_type interval)
  {
    return decomp[interval].end() - 1;
  }
};

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
struct adjacent_difference_closure
{
  InputIterator1  input;
  InputIterator2  input_copy;
  OutputIterator output;
  BinaryFunction binary_op;
  Decomposition  decomp;
  
  adjacent_difference_closure(InputIterator1  input,
                              InputIterator2  input_copy,
                              OutputIterator output,
                              BinaryFunction binary_op,
                              Decomposition  decomp)
    : input(input), input_copy(input_copy), output(output), binary_op(binary_op), decomp(decomp) {}

  __device__
  void operator()(void)
  {
// uses built-in variables
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    typedef typename thrust::iterator_value<InputIterator1>::type  InputType;
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;
    typedef typename Decomposition::index_type index_type;

    // this block processes results in [range.begin(), range.end())
    thrust::detail::backend::index_range<index_type> range = decomp[blockIdx.x];
    
    input_copy += blockIdx.x - 1;
      
    // prime the temp values for all threads so we don't need to launch a default constructor
    InputType next_left = (blockIdx.x == 0) ? dereference(input) : dereference(input_copy);

    index_type base = range.begin();
    index_type i    = range.begin() + threadIdx.x;
    
    if (i < range.end())
    {
      if (threadIdx.x > 0)
      {
        InputIterator1 temp = input + (i - 1);
        next_left = dereference(temp);
      }
    }
    
    input  += i;
    output += i;

    while (base < range.end())
    {
      InputType curr_left = next_left;

      if (i + blockDim.x < range.end())
      {
        InputIterator1 temp = input + (blockDim.x - 1);
        next_left = dereference(temp);
      }

      __syncthreads();

      if (i < range.end())
      {
        if (i == 0)
          dereference(output) = dereference(input);
        else
          dereference(output) = binary_op(dereference(input), curr_left);
      }

      i      += blockDim.x;
      base   += blockDim.x;
      input  += blockDim.x;
      output += blockDim.x;
    }

#endif // THRUST_DEVICE_COMPILER_NVCC
  }
};

} // end namespace detail


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
  typedef typename thrust::iterator_value<InputIterator>::type               InputType;
  typedef typename thrust::iterator_difference<InputIterator>::type          IndexType;
  typedef          thrust::detail::cuda_device_space_tag                     Space;
  typedef          thrust::detail::backend::uniform_decomposition<IndexType> Decomposition;

  IndexType n = last - first;

  if (n == 0)
    return result;

  Decomposition decomp = thrust::detail::backend::cuda::default_decomposition(last - first);

  // allocate temporary storage
  thrust::detail::uninitialized_array<InputType,Space> temp(decomp.size() - 1);

  // gather last value in each interval
  detail::last_index_in_each_interval<Decomposition> unary_op(decomp);
  thrust::gather(thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), unary_op),
                 thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), unary_op) + (decomp.size() - 1),
                 first,
                 temp.begin());

  
  typedef typename thrust::detail::uninitialized_array<InputType,Space>::iterator InputIterator2;
  typedef detail::adjacent_difference_closure<InputIterator,InputIterator2,OutputIterator,BinaryFunction,Decomposition> Closure;

  Closure closure(first, temp.begin(), result, binary_op, decomp); 

  thrust::detail::backend::cuda::detail::launch_closure(closure, decomp.size());
  
  return result + n;
}

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

