/*
 *  Copyright 2018 NVIDIA Corporation
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

#include <thrust/mr/memory_resource.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/system/cuda/pointer.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system/cuda/detail/util.h>

#include <thrust/memory/detail/host_system_resource.h>

THRUST_BEGIN_NS

namespace system
{
namespace cuda
{
namespace detail
{

    typedef cudaError_t (*allocation_fn)(void **, std::size_t);
    typedef cudaError_t (*deallocation_fn)(void *);

    template<allocation_fn Alloc, deallocation_fn Dealloc, typename Pointer>
    class cuda_memory_resource THRUST_FINAL : public mr::memory_resource<Pointer>
    {
    public:
        Pointer do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) THRUST_OVERRIDE
        {
            (void)alignment;

            void * ret;
            cudaError_t status = Alloc(&ret, bytes);

            if (status != cudaSuccess)
            {
                throw thrust::system::detail::bad_alloc(thrust::cuda_category().message(status).c_str());
            }

            return Pointer(ret);
        }

        void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) THRUST_OVERRIDE
        {
            (void)bytes;
            (void)alignment;

            cudaError_t status = Dealloc(thrust::detail::pointer_traits<Pointer>::get(p));

            if (status != cudaSuccess)
            {
                thrust::cuda_cub::throw_on_error(status, "CUDA free failed");
            }
        }
    };

    inline cudaError_t cudaMallocManaged(void ** ptr, std::size_t bytes)
    {
        return ::cudaMallocManaged(ptr, bytes, cudaMemAttachGlobal);
    }

    typedef detail::cuda_memory_resource<cudaMalloc, cudaFree,
        thrust::cuda::pointer<void> >
        device_memory_resource;
    typedef detail::cuda_memory_resource<detail::cudaMallocManaged, cudaFree,
        thrust::cuda::pointer<void> >
        managed_memory_resource;
    typedef detail::cuda_memory_resource<cudaMallocHost, cudaFreeHost,
        thrust::host_memory_resource::pointer>
        pinned_memory_resource;

} // end detail

typedef detail::device_memory_resource memory_resource;
typedef detail::managed_memory_resource universal_memory_resource;
typedef detail::pinned_memory_resource universal_host_pinned_memory_resource;

} // end cuda
} // end system

THRUST_END_NS

