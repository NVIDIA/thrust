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


/*! \file device_malloc.h
 *  \brief Defines the entry point to a function
 *         for allocating device storage.
 */

#pragma once

#include <komrade/detail/config.h>
#include <komrade/device_ptr.h>
#include <cstddef> // for std::size_t

namespace komrade
{

/*! \addtogroup allocation_functions Allocation Functions
 *  \ingroup memory_management_functions
 *  \{
 */

/*! This version of \p device_malloc allocates sequential device storage
 *  for bytes.
 *
 *  \param n The number of bytes to allocate sequentially
 *           in device memory.
 *  \return A \p device_ptr to the newly allocated memory.
 *
 *  The following code snippet demonstrates how to use \p device_malloc to
 *  allocate a range of device memory.
 *
 *  \code
 *  #include <komrade/device_malloc.h>
 *  #include <komrade/device_free.h>
 *  ...
 *  // allocate some memory with device_malloc
 *  const int N = 100;
 *  komrade::device_ptr<void> void_ptr = komrade::device_malloc(N);
 *
 *  // manipulate memory
 *  ...
 *
 *  // deallocate with device_free
 *  komrade::device_free(void_ptr);
 *  \endcode
 *
 *  \see device_ptr
 *  \see device_free
 */
inline komrade::device_ptr<void> device_malloc(const std::size_t n);

/*! This version of \p device_malloc allocates sequential device storage for
 *  new objects of the given type.
 *
 *  \param n The number of objects of type T to allocate
 *           sequentially in device memory.
 *  \return A \p device_ptr to the newly allocated memory.
 *
 *  The following code snippet demonstrates how to use \p device_malloc to
 *  allocate a range of device memory.
 *
 *  \code
 *  #include <komrade/device_malloc.h>
 *  #include <komrade/device_free.h>
 *  ...
 *  // allocate some integers with device_malloc
 *  const int N = 100;
 *  komrade::device_ptr<int> int_array = komrade::device_malloc<int>(N);
 *
 *  // manipulate integers
 *  ...
 *
 *  // deallocate with device_free
 *  komrade::device_free(int_array);
 *  \endcode
 *
 *  \see device_ptr
 *  \see device_free
 */
template<typename T>
  inline komrade::device_ptr<T> device_malloc(const std::size_t n);

/*! \}
 */

} // end komrade

#include <komrade/detail/device_malloc.inl>

