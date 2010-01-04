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


#pragma once

//#if defined(_MSC_VER) // Microsoft Visual C++ doesn't have stdint.h
//#include <vadefs.h>
//#else 
//#include <stdint.h> 
//#endif
//
// TODO find a portable <stdint.h> and put it in the Thrust namespace 
// TODO replace unsigned long with uintptr_t


// functions to handle memory alignment

namespace thrust
{
namespace detail
{
namespace util
{

template <typename T>
T * align_up(T * ptr, unsigned long bytes)
{
    return (T *) ( bytes * ((reinterpret_cast<unsigned long>(ptr) + (bytes - 1)) / bytes) );
}

template <typename T>
T * align_down(T * ptr, unsigned long bytes)
{
    return (T *) ( bytes * reinterpret_cast<unsigned long>(ptr) / bytes);
}

template <typename T>
bool is_aligned(T * ptr, unsigned long bytes = sizeof(T))
{
    return reinterpret_cast<unsigned long>(ptr) % bytes == 0;
}

} // end namespace util
} // end namespace detail
} // end namespace thrust

