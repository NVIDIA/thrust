/**
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 */

#pragma once

#include <vector_types.h>

namespace thrust {
namespace detail {
namespace device {
namespace cuda   {
namespace detail {
namespace b40c_thrust   {

//------------------------------------------------------------------------------
// Vector types
//------------------------------------------------------------------------------

template <typename K, int vec_elements> struct VecType;


//
// Define general vector types
//

template <typename K> 
struct VecType<K, 1> {
	K x;
	typedef K Type;
};

template <typename K> 
struct VecType<K, 2> {
	K x;
	K y;
	typedef VecType<K, 2> Type;
};

template <typename K> 
struct VecType<K, 4> {
	K x;
	K y;
	K z;
	K w;
	typedef VecType<K, 4> Type;
};

//
// Specialize certain built-in vector types
//

#define B40C_DEFINE_VECTOR_TYPE(base_type,short_type)                           \
  template<> struct VecType<base_type, 1> { typedef short_type##1 Type; };      \
  template<> struct VecType<base_type, 2> { typedef short_type##2 Type; };      \
  template<> struct VecType<base_type, 4> { typedef short_type##4 Type; };     

B40C_DEFINE_VECTOR_TYPE(char,               char)
B40C_DEFINE_VECTOR_TYPE(short,              short)
B40C_DEFINE_VECTOR_TYPE(int,                int)
B40C_DEFINE_VECTOR_TYPE(long,               long)
B40C_DEFINE_VECTOR_TYPE(long long,          longlong)
B40C_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
B40C_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
B40C_DEFINE_VECTOR_TYPE(unsigned int,       uint)
B40C_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
B40C_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
B40C_DEFINE_VECTOR_TYPE(float,              float)
B40C_DEFINE_VECTOR_TYPE(double,             double)

#undef B40C_DEFINE_VECTOR_TYPE

} // end namespace b40c_thrust
} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

