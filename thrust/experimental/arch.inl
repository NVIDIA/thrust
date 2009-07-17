/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file arch.inl
 *  \brief Inline file for arch.h.
 */

#include <assert.h>
#include <stdio.h>

#include <thrust/experimental/arch.h>

// #include this for make_uint3
#include <vector_functions.h>

namespace thrust
{

namespace experimental
{

namespace arch
{

namespace detail
{

inline void checked_get_current_device_properties(cudaDeviceProp &props)
{
  int current_device = -1;
  cudaError_t error = cudaGetDevice(&current_device);
  if(error)
  {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
  } // end if

  if(current_device < 0)
  {
    throw std::runtime_error(std::string("No CUDA device found."));
  } // end if
  
  error = cudaGetDeviceProperties(&props, current_device);
  if(error)
  {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
  } // end if
} // end checked_get_current_device_properties()

} // end detail


size_t num_multiprocessors(void)
{
  size_t result = 0;

  cudaDeviceProp properties;  

  detail::checked_get_current_device_properties(properties);
  result = properties.multiProcessorCount;

  return result;
} // end num_multiprocessors()


size_t max_active_threads_per_multiprocessor(void)
{
  // index this array by [major, minor] revision
  // \see NVIDIA_CUDA_Programming_Guide_2.1.pdf pp 82--83
  static const size_t max_active_threads_by_compute_capability[2][4] = {{   0,   0,   0,    0},
                                                                        {   768, 768, 1024, 1024}};
  size_t result = 0;

  cudaDeviceProp properties;  
  detail::checked_get_current_device_properties(properties);

  assert(properties.major == 1);
  assert(properties.minor >= 0 && properties.minor <= 3);

  result = max_active_threads_by_compute_capability[properties.major][properties.minor];

  return result;
} // end max_active_threads_per_multiprocessor()


size_t max_active_threads(void)
{
  return num_multiprocessors() * max_active_threads_per_multiprocessor();
} // end max_active_threads()


dim3 max_grid_dimensions(void)
{
  dim3 zero = make_uint3(0,0,0);
  dim3 result = zero;

  // \see NVIDIA_CUDA_Programming_Guide_2.1.pdf pp 82--83, A.1.1
  dim3 max_dim = make_uint3(65535, 65535, 65535);

  // index this array by [major, minor] revision
  // \see NVIDIA_CUDA_Programming_Guide_2.1.pdf pp 82--83
  static const dim3 max_grid_dimensions_by_compute_capability[2][4] = {{   zero,    zero,    zero,    zero},
                                                                       {max_dim, max_dim, max_dim, max_dim}};

  cudaDeviceProp properties;  
  detail::checked_get_current_device_properties(properties);

  assert(properties.major == 1);
  assert(properties.minor >= 0 && properties.minor <= 3);

  result = max_grid_dimensions_by_compute_capability[properties.major][properties.minor];

  return result;
} // end max_grid_dimensions()

}; // end arch

}; // end experimental

}; // end thrust

