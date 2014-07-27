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

#pragma once

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/choose_sizes.hpp>
#include <thrust/system/cuda/detail/bulk/detail/closure.hpp>
#include <thrust/system/cuda/detail/bulk/detail/cuda_launcher/cuda_launcher.hpp>


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


template<typename Closure>
__host__ __device__
thrust::pair<typename parallel_group<concurrent_group<> >::size_type,
             typename concurrent_group<>::size_type>
  choose_sizes(parallel_group<concurrent_group<> > g, Closure)
{
  bulk::detail::cuda_launcher<
    parallel_group<concurrent_group<> >,
    Closure
  > launcher;

  return launcher.choose_sizes(g.size(), g.this_exec.size());
} // end choose_sizes()


} // end detail


template<typename Function>
__host__ __device__
thrust::pair<typename parallel_group<concurrent_group<> >::size_type,
             typename concurrent_group<>::size_type>
  choose_sizes(parallel_group<concurrent_group<> > g, Function f)
{
  return bulk::detail::choose_sizes(g, detail::make_closure(f));
}


template<typename Function, typename Arg1>
__host__ __device__
thrust::pair<typename parallel_group<concurrent_group<> >::size_type,
             typename concurrent_group<>::size_type>
  choose_sizes(parallel_group<concurrent_group<> > g, Function f, Arg1 arg1)
{
  return bulk::detail::choose_sizes(g, detail::make_closure(f,arg1));
}


template<typename Function, typename Arg1, typename Arg2>
__host__ __device__
thrust::pair<typename parallel_group<concurrent_group<> >::size_type,
             typename concurrent_group<>::size_type>
  choose_sizes(parallel_group<concurrent_group<> > g, Function f, Arg1 arg1, Arg2 arg2)
{
  return bulk::detail::choose_sizes(g, detail::make_closure(f,arg1,arg2));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3>
__host__ __device__
thrust::pair<typename parallel_group<concurrent_group<> >::size_type,
             typename concurrent_group<>::size_type>
  choose_sizes(parallel_group<concurrent_group<> > g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  return bulk::detail::choose_sizes(g, detail::make_closure(f,arg1,arg2,arg3));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
__host__ __device__
thrust::pair<typename parallel_group<concurrent_group<> >::size_type,
             typename concurrent_group<>::size_type>
  choose_sizes(parallel_group<concurrent_group<> > g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  return bulk::detail::choose_sizes(g, detail::make_closure(f,arg1,arg2,arg3,arg4));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
__host__ __device__
thrust::pair<typename parallel_group<concurrent_group<> >::size_type,
             typename concurrent_group<>::size_type>
  choose_sizes(parallel_group<concurrent_group<> > g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  return bulk::detail::choose_sizes(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
__host__ __device__
thrust::pair<typename parallel_group<concurrent_group<> >::size_type,
             typename concurrent_group<>::size_type>
  choose_sizes(parallel_group<concurrent_group<> > g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  return bulk::detail::choose_sizes(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6));
}


} // end bulk
BULK_NAMESPACE_SUFFIX

