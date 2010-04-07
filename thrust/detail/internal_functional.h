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


/*! \file internal_functional.inl
 *  \brief Non-public functionals used to implement algorithm internals.
 */

namespace thrust
{
namespace detail
{

// note that equal_to_value does not force conversion from T2 -> T1 as equal_to does
template <typename T2>
struct equal_to_value
{
    const T2 rhs;

    equal_to_value(const T2& rhs) : rhs(rhs) {}

    template <typename T1>
        __host__ __device__
        bool operator()(const T1& lhs) const
        {
            return lhs == rhs;
        }
};

} // end namespace detail
} // end namespace thrust

