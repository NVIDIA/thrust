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

namespace b40c {

//------------------------------------------------------------------------------
// Vector types
//------------------------------------------------------------------------------

template <typename K, int vec_elements> struct VecType;


//
// Vectors of arbitrary types
//

template <typename K> 
struct VecType<K, 1> {
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
// Vectors of floats
//

template<>
struct VecType<float, 1> {
	typedef float Type;
};

template<>
struct VecType<float, 2> {
	typedef float2 Type;
};

template<>
struct VecType<float, 4> {
	typedef float4 Type;
};


//
// Vectors of doubles
//

template<>
struct VecType<double, 1> {
	typedef double Type;
};

template<>
struct VecType<double, 2> {
	typedef double2 Type;
};

template<>
struct VecType<double, 4> {
	typedef double4 Type;
};


//
// Vectors of chars
//

template<>
struct VecType<char, 1> {
	typedef char Type;
};

template<>
struct VecType<char, 2> {
	typedef char2 Type;
};

template<>
struct VecType<char, 4> {
	typedef char4 Type;
};


//
// Vectors of unsigned chars
//

template<>
struct VecType<unsigned char, 1> {
	typedef unsigned char Type;
};

template<>
struct VecType<unsigned char, 2> {
	typedef uchar2 Type;
};

template<>
struct VecType<unsigned char, 4> {
	typedef uchar4 Type;
};


//
// Vectors of shorts
//

template<>
struct VecType<short, 1> {
	typedef short Type;
};

template<>
struct VecType<short, 2> {
	typedef short2 Type;
};

template<>
struct VecType<short, 4> {
	typedef short4 Type;
};


//
// Vectors of unsigned shorts
//

template<>
struct VecType<unsigned short, 1> {
	typedef unsigned short Type;
};

template<>
struct VecType<unsigned short, 2> {
	typedef ushort2 Type;
};

template<>
struct VecType<unsigned short, 4> {
	typedef ushort4 Type;
};


//
// Vectors of ints
//

template<>
struct VecType<int, 1> {
	typedef int Type;
};

template<>
struct VecType<int, 2> {
	typedef int2 Type;
};

template<>
struct VecType<int, 4> {
	typedef int4 Type;
};


//
// Vectors of unsigned ints
//

template<>
struct VecType<unsigned int, 1> {
	typedef unsigned int Type;
};

template<>
struct VecType<unsigned int, 2> {
	typedef uint2 Type;
};

template<>
struct VecType<unsigned int, 4> {
	typedef uint4 Type;
};


//
// Vectors of longs
//

template<>
struct VecType<long, 1> {
	typedef long Type;
};

template<>
struct VecType<long, 2> {
	typedef long2 Type;
};

template<>
struct VecType<long, 4> {
	typedef long4 Type;
};


//
// Vectors of unsigned longs
//

template<>
struct VecType<unsigned long, 1> {
	typedef unsigned long Type;
};

template<>
struct VecType<unsigned long, 2> {
	typedef ulong2 Type;
};

template<>
struct VecType<unsigned long, 4> {
	typedef ulong4 Type;
};


//
// Vectors of long longs
//

template<>
struct VecType<long long, 1> {
	typedef long long Type;
};

template<>
struct VecType<long long, 2> {
	typedef longlong2 Type;
};

template<>
struct VecType<long long, 4> {
	typedef longlong4 Type;
};


//
// Vectors of unsigned long longs
//

template<>
struct VecType<unsigned long long, 1> {
	typedef unsigned long long Type;
};

template<>
struct VecType<unsigned long long, 2> {
	typedef ulonglong2 Type;
};

template<>
struct VecType<unsigned long long, 4> {
	typedef ulonglong4 Type;
};

} // namespace b40c

