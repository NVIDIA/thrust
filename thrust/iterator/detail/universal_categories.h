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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/detail/backend_iterator_categories.h>

namespace thrust
{

// define these types without inheritance to avoid ambiguous conversion to base classes

struct input_universal_iterator_tag
{
  operator input_host_iterator_tag () {return input_host_iterator_tag();}

  operator thrust::detail::input_cuda_device_iterator_tag () {return thrust::detail::input_cuda_device_iterator_tag();}

  operator detail::input_omp_device_iterator_tag () {return detail::input_omp_device_iterator_tag();}
};

struct output_universal_iterator_tag
{
  operator output_host_iterator_tag () {return output_host_iterator_tag();}

  operator detail::output_cuda_device_iterator_tag () {return detail::output_cuda_device_iterator_tag();}

  operator detail::output_omp_device_iterator_tag () {return detail::output_omp_device_iterator_tag();}
};

struct forward_universal_iterator_tag
  : input_universal_iterator_tag
{
  operator forward_host_iterator_tag () {return forward_host_iterator_tag();};

  operator detail::forward_cuda_device_iterator_tag () {return detail::forward_cuda_device_iterator_tag();};

  operator detail::forward_omp_device_iterator_tag () {return detail::forward_omp_device_iterator_tag();};
};

struct bidirectional_universal_iterator_tag
  : forward_universal_iterator_tag
{
  operator bidirectional_host_iterator_tag () {return bidirectional_host_iterator_tag();};

  operator detail::bidirectional_cuda_device_iterator_tag () {return detail::bidirectional_cuda_device_iterator_tag();};

  operator detail::bidirectional_omp_device_iterator_tag () {return detail::bidirectional_omp_device_iterator_tag();};
};


namespace detail
{

// create this struct to control conversion precedence in random_access_universal_iterator_tag
template<typename T>
struct one_degree_of_separation
  : T
{
};

} // end detail


struct random_access_universal_iterator_tag
{
  // these conversions are all P0
  operator random_access_host_iterator_tag () {return random_access_host_iterator_tag();};

  operator random_access_device_iterator_tag () {return random_access_device_iterator_tag();};

  operator detail::random_access_cuda_device_iterator_tag () {return detail::random_access_cuda_device_iterator_tag();};

  operator detail::random_access_omp_device_iterator_tag () {return detail::random_access_omp_device_iterator_tag();};

  // bidirectional_universal_iterator_tag is P1
  operator detail::one_degree_of_separation<bidirectional_universal_iterator_tag> () {return detail::one_degree_of_separation<bidirectional_universal_iterator_tag>();}

};


} // end thrust

