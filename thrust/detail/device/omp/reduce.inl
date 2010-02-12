/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

template <typename InputIterator,
          typename OutputType,
          typename BinaryFunction>
OutputType reduce(InputIterator first,
                  InputIterator last,
                  OutputType init,
                  BinaryFunction binary_op)
{
    // we're attempting to launch an omp kernel, assert we're compiling with omp support
    // ========================================================================
    // X Note to the user: If you've found this line due to a compiler error, X
    // X you need to OpenMP support in your compiler.                         X
    // ========================================================================
    THRUST_STATIC_ASSERT( (depend_on_instantiation<InputIterator,
                          (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value) );

    typedef typename thrust::iterator_difference<InputIterator>::type difference_type;

    if (first == last)
        return init;

    difference_type N = thrust::distance(first, last);

// do not attempt to compile the body of this function, which calls omp functions, without
// support from the compiler
// XXX implement the body of this function in another file to eliminate this ugliness
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
    int num_threads = std::min<difference_type>(omp_get_max_threads(), N);

    thrust::detail::raw_omp_device_buffer<OutputType> thread_results(first, first + num_threads);

#   pragma omp parallel num_threads(num_threads)
    {

        int thread_id = omp_get_thread_num();

        OutputType thread_sum = thread_results[thread_id];

#      pragma omp for 
        for (difference_type i = num_threads; i < N; i++)
        {
            InputIterator temp = first + i;
            thread_sum = binary_op(thread_sum, thrust::detail::device::dereference(temp));
        }

        thread_results[thread_id] = thread_sum;
    }
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE

    OutputType total_sum = init;

#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
    for (typename thrust::detail::raw_omp_device_buffer<OutputType>::iterator result = thread_results.begin();
         result != thread_results.end();
         ++result)
        total_sum = binary_op(total_sum, thrust::detail::device::dereference(result));
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE

    return total_sum;
}

} // end namespace omp
} // end namespace device
} // end namespace detail
} // end namespace thrust

