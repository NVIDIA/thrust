/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

namespace thrust
{
namespace detail
{
namespace backend
{
namespace generic
{

template<typename Tag>
__host__ __device__
  Tag select_system(Tag)
{
  return Tag();
} // end select_system()

// XXX this is just a placeholder
//     for now, return Tag1
template<typename Tag1, typename Tag2>
__host__ __device__
  Tag1 select_system(Tag1, Tag2)
{
  return Tag1();
} // end select_system()

} // end generic
} // end backend
} // end detail
} // end thrust

