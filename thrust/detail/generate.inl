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


/*! \file generate.inl
 *  \author Jared Hoberock
 *  \brief Inline file for generate.h.
 */

#include <thrust/generate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/dispatch/generate.h>

namespace thrust
{

template<typename ForwardIterator,
         typename Generator>
  void generate(ForwardIterator first,
                ForwardIterator last,
                Generator gen)
{
  detail::dispatch::generate(first, last, gen,
    typename thrust::iterator_space<ForwardIterator>::type());
} // end generate()

} // end thrust

