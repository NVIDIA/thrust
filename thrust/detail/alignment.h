/*
 *  Copyright 2017 NVIDIA Corporation
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

/*! \file alignment.h
 *  \brief Type-alignment utilities.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h> // For `integral_constant`.

#include <cstddef> // For `std::size_t` and `std::max_align_t`.

#include <type_traits> // For `std::alignment_of` and `std::aligned_storage`.

namespace thrust
{
namespace detail
{

/// \p THRUST_ALIGNOF is a macro that takes a single type-id as a parameter,
/// and returns the alignment requirement of the type in bytes.
///
/// It is an approximation of C++11's `alignof` operator.
///
/// Note: MSVC does not allow the builtin used to implement this to be placed
/// inside of a `__declspec(align(#))` attribute. As a workaround, you can
/// assign the result of \p THRUST_ALIGNOF to a variable and pass the variable
/// as the argument to `__declspec(align(#))`.
#define THRUST_ALIGNOF(x) alignof(x)

/// \p alignment_of provides the member constant `value` which is equal to the
/// alignment requirement of the type `T`, as if obtained by a C++11 `alignof`
/// expression.
///
/// It is an implementation of C++11's \p std::alignment_of.
template <typename T>
using alignment_of = std::alignment_of<T>;

/// \p aligned_type provides the nested type `type`, which is a trivial
/// type whose alignment requirement is a divisor of `Align`.
///
/// The behavior is undefined if `Align` is not a power of 2.
template <std::size_t Align>
struct aligned_type
{
    struct alignas(Align) type {};
};

/// \p aligned_storage provides the nested type `type`, which is a trivial type
/// suitable for use as uninitialized storage for any object whose size is at
/// most `Len` bytes and whose alignment requirement is a divisor of `Align`.
///
/// The behavior is undefined if `Len` is 0 or `Align` is not a power of 2.
///
/// It is an implementation of C++11's \p std::aligned_storage.
template <std::size_t Len, std::size_t Align>
using aligned_storage = std::aligned_storage<Len, Align>;

/// \p max_align_t is a trivial type whose alignment requirement is at least as
/// strict (as large) as that of every scalar type.
///
/// It is an implementation of C++11's \p std::max_align_t.
using max_align_t = std::max_align_t;

/// \p aligned_reinterpret_cast `reinterpret_cast`s \p u of type \p U to `void*`
/// and then `reinterpret_cast`s the result to \p T. The indirection through
/// `void*` suppresses compiler warnings when the alignment requirement of \p *u
/// is less than the alignment requirement of \p *t. The caller of
/// \p aligned_reinterpret_cast is responsible for ensuring that the alignment
/// requirements are actually satisified.
template <typename T, typename U>
__host__ __device__
T aligned_reinterpret_cast(U u)
{
  return reinterpret_cast<T>(reinterpret_cast<void*>(u));
}

__host__ __device__
inline std::size_t aligned_storage_size(std::size_t n, std::size_t align)
{
  return ((n + align - 1) / align) * align;
}

} // end namespace detail
} // end namespace thrust

