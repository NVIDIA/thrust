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


/*! \file device_free.h
 *  \brief Defines the entry point to a function
 *         for deallocating device storage.
 */

#pragma once

#include <komrade/detail/config.h>
#include <komrade/device_ptr.h>

namespace komrade
{

/*! \addtogroup deallocation_functions Deallocation Functions
 *  \ingroup memory_management_functions
 *  \{
 */

/*! \p device_free deallocates memory allocated by the function \p device_malloc.
 *
 *  \param ptr A \p device_ptr pointing to memory to be deallocated.
 *
 *  The following code snippet demonstrates how to use \p device_free to
 *  deallocate memory allocated by \p device_malloc.
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
 *  \see device_malloc
 */
inline void device_free(komrade::device_ptr<void> ptr);

/*! \}
 */

} // end komrade

#include <komrade/detail/device_free.inl>

