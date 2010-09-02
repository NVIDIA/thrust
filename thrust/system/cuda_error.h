/*
 *  Copyright 2008-2010 NVIDIA Corporation
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


/*! \file cuda_error.h
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/error_code.h>
#include <cuda_runtime_api.h>

namespace thrust
{

namespace experimental
{

namespace system
{

/*! \addtogroup system
 *  \{
 */

// To construct an error_code after a CUDA Runtime error:
//
//   error_code(::cudaGetLastError(), cuda_category())

// XXX N3000 prefers enum class cuda_errc { ... }
namespace cuda_errc
{

// typedef cudaError_t to cuda_errc_t so we don't have to update the list
typedef cudaError_t cuda_errc_t;

} // end namespace cuda_errc


// XXX N3000 prefers is_error_code_enum<cuda_errc>
template<> struct is_error_code_enum<cuda_errc::cuda_errc_t> : thrust::detail::true_type {};


// XXX replace cuda_errc::cuda_errc_t with cuda_errc upon c++0x
/*! \return <tt>error_code(static_cast<int>(e), cuda_category())</tt>
 */
inline error_code make_error_code(cuda_errc::cuda_errc_t e);


// XXX replace cuda_errc::cuda_errc_t with cuda_error upon c++0x
/*! \return <tt>error_condition(static_cast<int>(e), cuda_category())</tt>.
 */
inline error_condition make_error_condition(cuda_errc::cuda_errc_t e);


/*! \return A reference to an object of a type derived from class \p error_category.
 *  \note The object's \p equivalent virtual functions shall behave as specified
 *        for the class \p error_category. The object's \p name virtual function shall
 *        return a pointer to the string <tt>"cuda"</tt>. The object's
 *        \p default_error_condition virtual function shall behave as follows:
 *
 *        If the argument <tt>ev</tt> corresponds to a CUDA error value, the function
 *        shall return <tt>error_condition(ev,cuda_category())</tt>.
 *        Otherwise, the function shall return <tt>system_category.default_error_condition(ev)</tt>.
 */
inline const error_category &cuda_category(void);

/*! \} // end system
 */


} // end system

using system::cuda_category;

// XXX replace with using system::cuda_errc upon c++0x
namespace cuda_errc = system::cuda_errc;

} // end namespace experimental

} // end namespace thrust

#include <thrust/system/detail/cuda_error.inl>

