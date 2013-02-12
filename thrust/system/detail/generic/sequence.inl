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
#include <thrust/system/detail/generic/sequence.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/tabulate.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename ForwardIterator>
  void sequence(thrust::execution_policy<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type T;

  thrust::sequence(exec, first, last, T(0));
} // end sequence()


template<typename DerivedPolicy, typename ForwardIterator, typename T>
  void sequence(thrust::execution_policy<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                T init)
{
  thrust::sequence(exec, first, last, init, T(1));
} // end sequence()


template<typename DerivedPolicy, typename ForwardIterator, typename T>
  void sequence(thrust::execution_policy<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                T init,
                T step)
{
  thrust::tabulate(exec, first, last, init + step * thrust::placeholders::_1);
} // end sequence()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

