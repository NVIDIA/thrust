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
#include <thrust/system/cuda/detail/copy_device_to_device.h>
#include <thrust/system/cuda/detail/copy_cross_system.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/trivial_copy.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{

template<typename System,
         typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_device(dispatchable<System> &system,
                                       InputIterator begin, 
                                       InputIterator end, 
                                       OutputIterator result,
                                       thrust::detail::false_type)
{
    // general case (mixed types)
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    return thrust::transform(system, begin, end, result, thrust::identity<InputType>());
#else
    // we're not compiling with nvcc: copy [begin, end) to temp host memory
    typename thrust::iterator_traits<InputIterator>::difference_type n = thrust::distance(begin, end);

    thrust::host_system_tag temp_system;
    thrust::detail::temporary_array<InputType, thrust::host_system_tag> temp1(temp_system, begin, end);

    // transform temp1 to OutputType in host memory
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    thrust::detail::temporary_array<OutputType, thrust::host_system_tag> temp2(temp_system, temp1.begin(), temp1.end());

    // copy temp2 to device
    result = thrust::system::cuda::detail::copy_cross_system(temp2.begin(), temp2.end(), result);

    return result;
#endif // THRUST_DEVICE_COMPILER_NVCC
}


template<typename System,
         typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_device(dispatchable<System> &system,
                                       InputIterator begin, 
                                       InputIterator end, 
                                       OutputIterator result,
                                       thrust::detail::true_type)
{
    // specialization for device to device when the value_types match, operator= is not overloaded,
    // and the iterators are pointers

    // how many elements to copy?
    typename thrust::iterator_traits<OutputIterator>::difference_type n = end - begin;

    thrust::system::cuda::detail::trivial_copy_n(system, begin, n, result);

    return result + n;
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////

template<typename System,
         typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_device(dispatchable<System> &system,
                                       InputIterator begin, 
                                       InputIterator end, 
                                       OutputIterator result)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    const bool use_trivial_copy = 
        thrust::detail::is_same<InputType, OutputType>::value
        && thrust::detail::is_trivial_iterator<InputIterator>::value 
        && thrust::detail::is_trivial_iterator<OutputIterator>::value;

    // XXX WAR unused variable warning
    (void) use_trivial_copy;

    return detail::copy_device_to_device(system, begin, end, result,
            thrust::detail::integral_constant<bool, use_trivial_copy>());

}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

