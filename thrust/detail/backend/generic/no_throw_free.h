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

#pragma once

#include <thrust/device_ptr.h>
#include <thrust/device_free.h>

namespace thrust
{

// XXX WAR circular #inclusion with forward declaration
void device_free(thrust::device_ptr<void> ptr);

namespace detail
{

namespace backend
{

namespace generic
{


template<unsigned int DummyParameterToAvoidInstantiation>
  void no_throw_free(thrust::device_ptr<void> ptr) throw()
{
  try
  {
    thrust::device_free(ptr);
  }
  catch(...)
  {
    ;
  }
} // end no_throw_free()


} // end generic

} // end backend

} // end detail

} // end namespace thrust

