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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/async.hpp>
#include <thrust/system/cuda/detail/bulk/detail/cuda_launcher/cuda_launcher.hpp>
#include <thrust/system/cuda/detail/bulk/detail/closure.hpp>
#include <thrust/system/cuda/detail/bulk/detail/throw_on_error.hpp>
#include <thrust/system/cuda/detail/bulk/detail/terminate.hpp>


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


template<typename ExecutionGroup, typename Closure>
__host__ __device__
future<void> async_in_stream(ExecutionGroup g, Closure c, cudaStream_t s, cudaEvent_t before_event)
{
#if __BULK_HAS_CUDART__
  if(before_event != 0)
  {
    bulk::detail::throw_on_error(cudaStreamWaitEvent(s, before_event, 0), "cudaStreamWaitEvent in async_in_stream");
  }
#else
  bulk::detail::terminate_with_message("async_in_stream(): cudaStreamWaitEvent requires CUDART");
#endif

  bulk::detail::cuda_launcher<ExecutionGroup, Closure> launcher;
  launcher.launch(g, c, s);

  return future_core_access::create(s, false);
} // end async_in_stream()


template<typename ExecutionGroup, typename Closure>
__host__ __device__
future<void> async(ExecutionGroup g, Closure c, cudaEvent_t before_event)
{
  cudaStream_t s;

  // XXX cudaStreamCreate is __host__-only
  //     figure out a way to support this that does not require creating a new stream
#if (__BULK_HAS_CUDART__ && !defined(__CUDA_ARCH__))
  bulk::detail::throw_on_error(cudaStreamCreate(&s), "cudaStreamCreate in bulk::detail::async");
#else
  s = 0;
  bulk::detail::terminate_with_message("bulk::async(): cudaStreamCreate() is unsupported in __device__ code.");
#endif

#if __BULK_HAS_CUDART__
  if(before_event != 0)
  {
    bulk::detail::throw_on_error(cudaStreamWaitEvent(s, before_event, 0), "cudaStreamWaitEvent in bulk::detail::async");
  }
#else
  bulk::detail::terminate_with_message("async_in_stream(): cudaStreamWaitEvent requires CUDART");
#endif

  bulk::detail::cuda_launcher<ExecutionGroup, Closure> launcher;
  launcher.launch(g, c, s);

  // note we pass true here, unlike false above
  return future_core_access::create(s, true);
} // end async()


template<typename ExecutionGroup, typename Closure>
__host__ __device__
future<void> async(ExecutionGroup g, Closure c)
{
  return bulk::detail::async_in_stream(g, c, 0, 0);
} // end async()


template<typename ExecutionGroup, typename Closure>
__host__ __device__
future<void> async(async_launch<ExecutionGroup> launch, Closure c)
{
  return launch.is_stream_valid() ?
    bulk::detail::async_in_stream(launch.exec(), c, launch.stream(), launch.before_event()) :
    bulk::detail::async(launch.exec(), c, launch.before_event());
} // end async()


} // end detail


template<typename ExecutionGroup, typename Function>
__host__ __device__
future<void> async(ExecutionGroup g, Function f)
{
  return bulk::detail::async(g, detail::make_closure(f));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1,arg2));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1,arg2,arg3));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9));
} // end async()


template<typename ExecutionGroup, typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__host__ __device__
future<void> async(ExecutionGroup g, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9, Arg10 arg10)
{
  return bulk::detail::async(g, detail::make_closure(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10));
} // end async()


} // end bulk
BULK_NAMESPACE_SUFFIX

