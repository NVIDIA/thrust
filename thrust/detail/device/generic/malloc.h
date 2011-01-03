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


/*! \file malloc.h
 *  \brief Generic implementation of device malloc.
 */

#pragma once

#include <thrust/device_ptr.h>
#include <cstdlib>
#include <stdexcept>

namespace thrust
{

namespace detail
{

namespace device
{

namespace generic
{

template<unsigned int DummyParameterToAvoidInstantiation>
thrust::device_ptr<void> malloc(const std::size_t n)
{
  void *result = std::malloc(n);

  if(!result)
  {
    throw std::bad_alloc();
  } // end if

  return thrust::device_ptr<void>(result);
} // end malloc()

} // end namespace generic

} // end namespace device

} // end namespace detail

} // end namespace thrust

