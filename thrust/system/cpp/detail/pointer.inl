/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

namespace thrust
{

// XXX WAR an issue with MSVC 2005 (cl v14.00) incorrectly implementing
//     pointer_raw_pointer for pointer by specializing it here
#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && (_MSC_VER <= 1400)
namespace detail
{

template<typename T>
  struct pointer_raw_pointer< thrust::cpp::pointer<T> >
{
  typedef typename thrust::cpp::pointer<T>::raw_pointer type;
}; // end pointer_raw_pointer

} // end detail
#endif

namespace system
{
namespace cpp
{

template<typename T>
  template<typename OtherT>
    reference<T> &
      reference<T>
        ::operator=(const reference<OtherT> &other)
{
  return super_t::operator=(other);
} // end reference::operator=()

template<typename T>
  reference<T> &
    reference<T>
      ::operator=(const value_type &x)
{
  return super_t::operator=(x);
} // end reference::operator=()

template<typename T>
__host__ __device__
void swap(reference<T> a, reference<T> b)
{
  a.swap(b);
} // end swap()

} // end cpp
} // end system
} // end thrust

