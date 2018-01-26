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

#if __cplusplus >= 201103L
    #include <type_traits> // For `std::alignment_of`.
#endif

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
#if __cplusplus >= 201103L
    #define THRUST_ALIGNOF(x) alignof(x) 
#else
    #define THRUST_ALIGNOF(x) __alignof(x)
#endif

/// \p alignment_of provides the member constant `value` which is equal to the
/// alignment requirement of the type `T`, as if obtained by a C++11 `alignof`
/// expression.
/// 
/// It is an implementation of C++11's \p std::alignment_of.
#if __cplusplus >= 201103L
    template <typename T>
    using alignment_of = std::alignment_of<T>;
#else
    template <typename T>
    struct alignment_of;

    template <typename T, std::size_t size_diff>
    struct alignment_of_helper
    {
        static const std::size_t value =
            integral_constant<std::size_t, size_diff>::value;
    };

    template <typename T>
    struct alignment_of_helper<T, 0>
    {
        static const std::size_t value = alignment_of<T>::value;
    };

    template <typename T>
    struct alignment_of
    {
      private:
        struct impl
        {
            T    x;
            char c;
        };

      public:
        static const std::size_t value =
            alignment_of_helper<impl, sizeof(impl) - sizeof(T)>::value;
    };
#endif

/// \p aligned_byte provides the nested type `type`, which is a trivial
/// type whose alignment requirement is a divisor of `Align`.
///
/// The behavior is undefined if `Align` is not a power of 2.
template <std::size_t Align>
struct aligned_byte;

#if __cplusplus >= 201103L
    template <std::size_t Align>
    struct aligned_byte
    {
        struct alignas(Align) type {};
    };
#elif  (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC)                    \
    || (   (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC)                 \
        && (THRUST_GCC_VERSION < 40300))
    // We have to implement `aligned_byte` with specializations for MSVC
    // and GCC 4.2.x and older because they require literals as arguments to 
    // their alignment attribute.

    #if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC)
        #define THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(X)                  \
            template <>                                                       \
            struct aligned_byte<X>                                    \
            {                                                                 \
                __declspec(align(X)) struct type {};                          \
            };                                                                \
            /**/
    #else
        #define THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(X)                  \
            template <>                                                       \
            struct aligned_byte<X>                                    \
            {                                                                 \
                struct type {} __attribute__((aligned(X)));                   \
            };                                                                \
            /**/
    #endif
    
    THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(1);
    THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(2);
    THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(4);
    THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(8);
    THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(16);
    THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(32);
    THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(64);
    THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION(128);

    #undef THRUST_DEFINE_ALIGNED_BYTE_SPECIALIZATION
#else
    template <std::size_t Align>
    struct aligned_byte
    {
        struct type {} __attribute__((aligned(Align)));
    };
#endif

/// \p aligned_packed_byte provides the nested type `type`, which is a trivial
/// type whose size is 1 byte and alignment requirement is a divisor of `Align`.
///
/// The first element of a C-style or dynamic array of `aligned_packed_byte`s
/// will be aligned to the alignment requirement (assuming the alignment is
/// supported by the implementation and any allocators used). However,
/// subsequent elements will not be aligned.
///
/// It can be used when you have a pointer to storage allocated in bytes, and
/// you wish to cast the byte pointer (e.g. `max_aligned_packed_byte*`) to a
/// pointer type that has a greater alignment requirement without triggering
/// compiler warnings (`-Wcast-align`). You are responsible for ensuring that
/// the alignment requirements are actually satisified.
///
/// \p alignment_of will not necessarily work with \p aligned_packed_byte.
///
/// The behavior is undefined if `Align` is not a power of 2.
template <std::size_t Align>
struct aligned_packed_byte;

#if    (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC)                    \
    || (   (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC)                 \
        && (THRUST_GCC_VERSION < 40300))
    // We have to implement `aligned_byte` with specializations for MSVC and GCC
    // 4.2.x and older because they require literals as arguments to their
    // alignment attribute.

    #if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC)
        #define THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(X)           \
            template <>                                                       \
            struct aligned_packed_byte<X>                                     \
            {                                                                 \
              private:                                                        \
                struct underlying_type {};                                    \
              public:                                                         \
                typedef __declspec(align(X)) underlying_type type;            \
            };                                                                \
            /**/
    #else
        // `underlying_type` must be a dependent type, otherwise recent versions
        // of Clang complain because the alignment of `type` is dependent but
        // the type itself is not.
        #define THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(X)           \
            template <>                                                       \
            struct aligned_packed_byte<X>                                     \
            {                                                                 \
              private:                                                        \
                struct underlying_type {};                                    \
              public:                                                         \
                typedef underlying_type __attribute__((aligned(X))) type;     \
            };                                                                \
            /**/
    #endif
    
    THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(1);
    THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(2);
    THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(4);
    THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(8);
    THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(16);
    THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(32);
    THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(64);
    THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION(128);

    #undef THRUST_DEFINE_ALIGNED_PACKED_BYTE_SPECIALIZATION
#else
    template <std::size_t Align>
    struct aligned_packed_byte
    {
      private:
        struct underlying_type {};
      public:
        typedef underlying_type __attribute__((aligned(Align))) type;
    };
#endif

/// \p aligned_storage provides the nested type `type`, which is a trivial type
/// suitable for use as uninitialized storage for any object whose size is at
/// most `Len` bytes and whose alignment requirement is a divisor of `Align`.
/// 
/// The behavior is undefined if `Len` is 0 or `Align` is not a power of 2.
///
/// It is an implementation of C++11's \p std::alignment_of.
#if __cplusplus >= 201103L
    template <std::size_t Len, std::size_t Align>
    using aligned_storage = std::aligned_storage<Len, Align>;
#else
    template <std::size_t Len, std::size_t Align>
    struct aligned_storage
    {
        union type
        {
            unsigned char data[Len];
            // We put this into the union in case the alignment requirement of
            // an array of `unsigned char` of length `Len` is greater than
            // `Align`.

            typename aligned_byte<Align>::type align;
        };
    };
#endif

/// \p max_align_t is a trivial type whose alignment requirement is at least as
/// strict (as large) as that of every scalar type.
///
/// It is an implementation of C++11's \p std::max_align_t.
#if __cplusplus >= 201103L
    using max_align_t = std::max_align_t;
#else
    union max_align_t
    {
        // These cannot be private because C++03 POD types cannot have private
        // data members.
        char c;
        short s;
        int i;
        long l;
        float f;
        double d;
        long long ll;
        long double ld;
        void* p;
    };
#endif

/// \p max_aligned_packed_byte is a trivial type whose size is 1 and whose
/// alignment requirement is \p max_alignment.
/// 
/// It can be used when you have a pointer to storage allocated in bytes, and
/// you wish to cast the byte pointer (e.g. `max_aligned_packed_byte*`) to a
/// pointer type that has a greater alignment requirement without triggering
/// compiler warnings (`-Wcast-align`). You are responsible for ensuring that
/// the alignment requirements are actually satisified.
///
/// \p alignment_of will not necessarily work with \p max_aligned_packed_byte.
typedef aligned_packed_byte<alignment_of<max_align_t>::value>::type
        max_aligned_packed_byte;

} // end namespace detail
} // end namespace thrust

