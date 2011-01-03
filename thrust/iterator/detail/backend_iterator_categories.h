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

#include <thrust/iterator/iterator_categories.h>

namespace thrust
{
namespace detail
{

// define these in detail for now
struct cuda_device_iterator_tag {};

struct input_cuda_device_iterator_tag
  : cuda_device_iterator_tag
{
  operator input_device_iterator_tag () {return input_device_iterator_tag();}
};

struct output_cuda_device_iterator_tag
  : cuda_device_iterator_tag
{
  operator output_device_iterator_tag () {return output_device_iterator_tag();}
};

struct forward_cuda_device_iterator_tag
  : input_cuda_device_iterator_tag
{
  operator forward_device_iterator_tag () {return forward_device_iterator_tag();}
};

struct bidirectional_cuda_device_iterator_tag
  : forward_cuda_device_iterator_tag
{
  operator bidirectional_device_iterator_tag () {return bidirectional_device_iterator_tag();}
};

struct random_access_cuda_device_iterator_tag
  : bidirectional_cuda_device_iterator_tag
{
  operator random_access_device_iterator_tag () {return random_access_device_iterator_tag();} 
};



struct omp_device_iterator_tag {};

struct input_omp_device_iterator_tag
  : omp_device_iterator_tag
{
  operator input_device_iterator_tag () {return input_device_iterator_tag();}
};

struct output_omp_device_iterator_tag
  : omp_device_iterator_tag
{
  operator output_device_iterator_tag () {return output_device_iterator_tag();}
};

struct forward_omp_device_iterator_tag
  : input_omp_device_iterator_tag
{
  operator forward_device_iterator_tag () {return forward_device_iterator_tag();}
};

struct bidirectional_omp_device_iterator_tag
  : forward_omp_device_iterator_tag
{
  operator bidirectional_device_iterator_tag () {return bidirectional_device_iterator_tag();}
};

struct random_access_omp_device_iterator_tag
  : bidirectional_omp_device_iterator_tag
{
  operator random_access_device_iterator_tag () {return random_access_device_iterator_tag();} 
};

} // end namespace detail
} // end namespace thrust

