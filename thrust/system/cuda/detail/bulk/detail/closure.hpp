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
#include <thrust/system/cuda/detail/bulk/detail/apply_from_tuple.hpp>

#include <thrust/detail/config.h>
#include <thrust/tuple.h>

BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


template<typename Function, typename Tuple>
class closure
{
  public:
    typedef Function function_type;

    typedef Tuple arguments_type;

    __host__ __device__
    closure(function_type f, const arguments_type &args)
      :f(f),
       args(args)
    {}


    __host__ __device__
    void operator()()
    {
      apply_from_tuple(f,args);
    }


    __host__ __device__
    function_type function() const
    {
      return f;
    }


    __host__ __device__
    arguments_type arguments() const
    {
      return args;
    }


  private:
    function_type   f;
    arguments_type args;
}; // end closure


template<typename Function, typename Arguments>
__host__ __device__
const closure<Function,Arguments> &make_closure(const closure<Function,Arguments> &c)
{
  return c;
}


template<typename Function>
__host__ __device__
closure<Function, thrust::tuple<> > make_closure(Function f)
{
  return closure<Function,thrust::tuple<> >(f, thrust::tuple<>());
}


template<typename Function, typename Arg1>
__host__ __device__
closure<Function, thrust::tuple<Arg1> > make_closure(Function f, const Arg1 &a1)
{
  return closure<Function,thrust::tuple<Arg1> >(f, thrust::make_tuple(a1));
}


template<typename Function, typename Arg1, typename Arg2>
__host__ __device__
closure<
  Function,
  thrust::tuple<Arg1,Arg2>
>
  make_closure(Function f, const Arg1 &a1, const Arg2 &a2)
{
  return closure<Function,thrust::tuple<Arg1,Arg2> >(f, thrust::make_tuple(a1,a2));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3>
__host__ __device__
closure<
  Function,
  thrust::tuple<Arg1,Arg2,Arg3>
>
  make_closure(Function f, const Arg1 &a1, const Arg2 &a2, const Arg3 &a3)
{
  return closure<Function,thrust::tuple<Arg1,Arg2,Arg3> >(f, thrust::make_tuple(a1,a2,a3));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
__host__ __device__
closure<
  Function,
  thrust::tuple<Arg1,Arg2,Arg3,Arg4>
>
  make_closure(Function f, const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4)
{
  return closure<Function,thrust::tuple<Arg1,Arg2,Arg3,Arg4> >(f, thrust::make_tuple(a1,a2,a3,a4));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
__host__ __device__
closure<
  Function,
  thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5>
>
  make_closure(Function f, const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5)
{
  return closure<Function,thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5> >(f, thrust::make_tuple(a1,a2,a3,a4,a5));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
__host__ __device__
closure<
  Function,
  thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6>
>
  make_closure(Function f, const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6)
{
  return closure<Function,thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6> >(f, thrust::make_tuple(a1,a2,a3,a4,a5,a6));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
__host__ __device__
closure<
  Function,
  thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7>
>
  make_closure(Function f, const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7)
{
  return closure<Function,thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7> >(f, thrust::make_tuple(a1,a2,a3,a4,a5,a6,a7));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
__host__ __device__
closure<
  Function,
  thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8>
>
  make_closure(Function f, const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8)
{
  return closure<Function,thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8> >(f, thrust::make_tuple(a1,a2,a3,a4,a5,a6,a7,a8));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
__host__ __device__
closure<
  Function,
  thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9>
>
  make_closure(Function f, const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8, const Arg9 &a9)
{
  return closure<Function,thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9> >(f, thrust::make_tuple(a1,a2,a3,a4,a5,a6,a7,a8,a9));
}


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__host__ __device__
closure<
  Function,
  thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9,Arg10>
>
  make_closure(Function f, const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8, const Arg9 &a9, const Arg10 &a10)
{
  return closure<Function,thrust::tuple<Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9,Arg10> >(f, thrust::make_tuple(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10));
}


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

