/*
 *  Copyright 2008-2013 NVIDIA Corporation
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


/*! \file advance.inl
 *  \brief Inline file for advance.h
 */

#include <thrust/detail/config.h>
#include <thrust/advance.h>
#include <thrust/system/detail/generic/advance.h>

namespace thrust
{


template <typename InputIterator, typename Distance>
__host__ __device__
void advance(InputIterator& i, Distance n)
{
  thrust::system::detail::generic::advance(i, n);
} // end advance()


} // end namespace thrust

