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

// don't attempt to #include this file without omp support
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
#include <omp.h>
#endif // omp support

#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/device/dereference.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace omp
{


template<typename RandomAccessIterator1,
         typename SizeType1,
         typename SizeType2,
         typename BinaryFunction,
         typename RandomAccessIterator2>
  void unordered_blocked_reduce_n(RandomAccessIterator1 first,
                                  SizeType1 n,
                                  SizeType2 num_blocks,
                                  BinaryFunction binary_op,
                                  RandomAccessIterator2 result)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to OpenMP support in your compiler.                         X
  // ========================================================================
  THRUST_STATIC_ASSERT( (depend_on_instantiation<RandomAccessIterator1,
                        (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value) );

  // check for empty input or output
  if(n == 0 || num_blocks == 0) return;

  typedef typename thrust::iterator_value<RandomAccessIterator2>::type OutputType;

// do not attempt to compile the body of this function, which calls omp functions, without
// support from the compiler
// XXX implement the body of this function in another file to eliminate this ugliness
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
# pragma omp parallel num_threads(num_blocks)
  {
    int thread_id = omp_get_thread_num();
    
    RandomAccessIterator1 temp = first + thread_id;
    OutputType thread_sum = thrust::detail::device::dereference(temp);

#   pragma omp for 
    for(SizeType1 i = SizeType1(num_blocks); i < n; i++)
    {
      RandomAccessIterator1 temp = first + i;
      thread_sum = binary_op(thread_sum, thrust::detail::device::dereference(temp));
    }

    result[thread_id] = thread_sum;
  }
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
} // end unordered_blocked_reduce_n()


template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  SizeType get_unordered_blocked_reduce_n_schedule(RandomAccessIterator first,
                                                   SizeType n,
                                                   OutputType init,
                                                   BinaryFunction binary_op)
{
  SizeType result = 0;

#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
  result = std::min<SizeType>(omp_get_max_threads(), n);
#endif

  return result;
} // end get_unordered_blocked_reduce_n_schedule()


} // end namespace omp
} // end namespace device
} // end namespace detail
} // end namespace thrust

