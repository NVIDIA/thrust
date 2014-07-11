/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
#include <thrust/system/cuda/detail/copy.h>
#include <thrust/system/cuda/detail/copy_device_to_device.h>
#include <thrust/system/cuda/detail/copy_cross_system.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


template<typename System,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
OutputIterator copy(execution_policy<System> &system,
                    InputIterator first,
                    InputIterator last,
                    OutputIterator result)
{
  return thrust::system::cuda::detail::copy_device_to_device(system,first,last,result);
} // end copy()


template<typename System1,
         typename System2,
         typename InputIterator,
         typename OutputIterator>
OutputIterator copy(cross_system<System1,System2> systems,
                    InputIterator first,
                    InputIterator last,
                    OutputIterator result)
{
  return thrust::system::cuda::detail::copy_cross_system(systems,first,last,result);
} // end copy()


template<typename System,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
__host__ __device__
OutputIterator copy_n(execution_policy<System> &system,
                      InputIterator first,
                      Size n,
                      OutputIterator result)
{
  return thrust::system::cuda::detail::copy_device_to_device(system,first,first+n,result);
} // end copy_n()


template<typename System1,
         typename System2,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
OutputIterator copy_n(cross_system<System1,System2> systems,
                      InputIterator first,
                      Size n,
                      OutputIterator result)
{
  return thrust::system::cuda::detail::copy_cross_system_n(systems,first,n,result);
} // end copy_n()


} // end detail
} // end cuda
} // end system
} // end thrust

