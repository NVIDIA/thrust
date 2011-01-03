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

/*! \file free.h
 *  \brief Generic implementation of device free.
 */

#pragma once

#include <thrust/device_ptr.h>
#include <cstdlib>
#include <iostream>

namespace thrust
{

namespace detail
{

namespace device
{

namespace generic
{

template<unsigned int DummyParameterToAvoidInstantiation>
void free(thrust::device_ptr<void> ptr)
{
  return std::free(ptr.get());
} // end free()

} // end namespace generic

} // end namespace device

} // end namespace detail

} // end namespace thrust


