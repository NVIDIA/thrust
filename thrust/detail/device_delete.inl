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


/*! \file device_delete.inl
 *  \brief Inline file for device_delete.h.
 */

#include <thrust/device_delete.h>
#include <thrust/device_free.h>
#include <thrust/detail/destroy.h>

namespace thrust
{

template<typename T>
  void device_delete(device_ptr<T> ptr,
                     const size_t n)
{
  // XXX defer to cudaDelete once it is implemented
  detail::destroy(ptr, ptr + n);
  device_free(ptr);
} // end device_delete()

} // end thrust

