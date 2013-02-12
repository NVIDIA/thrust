/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/generate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/for_each.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename ExecutionPolicy,
         typename ForwardIterator,
         typename Generator>
  void generate(thrust::execution_policy<ExecutionPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                Generator gen)
{
  thrust::for_each(exec, first, last, typename thrust::detail::generate_functor<ExecutionPolicy,Generator>::type(gen));
} // end generate()

template<typename ExecutionPolicy,
         typename OutputIterator,
         typename Size,
         typename Generator>
  OutputIterator generate_n(thrust::execution_policy<ExecutionPolicy> &exec,
                            OutputIterator first,
                            Size n,
                            Generator gen)
{
  return thrust::for_each_n(exec, first, n, typename thrust::detail::generate_functor<ExecutionPolicy,Generator>::type(gen));
} // end generate()

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

