/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <cuda_occupancy.h>
#include <thrust/detail/config.h>
#include <thrust/system/cuda/config.h>
#include <thrust/iterator/detail/is_trivial_iterator.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/cub/block/block_load.cuh>
#include <thrust/system/cuda/detail/cub/block/block_store.cuh>
#include <thrust/system/cuda/detail/cub/block/block_scan.cuh>


BEGIN_NS_THRUST

namespace cuda_cub {
namespace core {

#if (CUB_PTX_ARCH >= 600)
#  define THRUST_TUNING_ARCH sm60
#elif (CUB_PTX_ARCH >= 520)
#  define THRUST_TUNING_ARCH sm52
#elif (CUB_PTX_ARCH >= 350)
#  define THRUST_TUNING_ARCH sm35
#elif (CUB_PTX_ARCH >= 300)
#  define THRUST_TUNING_ARCH sm30
#else
#  define THRUST_TUNING_ARCH sm20
#endif

  struct sm20  { enum { ver = 200 }; };
  struct sm30  { enum { ver = 300 }; };
  struct sm35  { enum { ver = 350 }; };
  struct sm52  { enum { ver = 520 }; };
  struct sm60  { enum { ver = 600 }; };

  
  // supported SM versions
  // ---------------------
  template<size_t I=(size_t)-1> 
  struct sm_arch { enum {count = 5}; };

  template<> struct sm_arch<4> : sm60 { typedef sm60 type; typedef sm_arch<3> next;};
  template<> struct sm_arch<3> : sm52 { typedef sm52 type; typedef sm_arch<2> next;};
  template<> struct sm_arch<2> : sm35 { typedef sm35 type; typedef sm_arch<1> next;};
  template<> struct sm_arch<1> : sm30 { typedef sm30 type; typedef sm_arch<0> next;};
  template<> struct sm_arch<0> : sm20 { typedef sm20 type; };


  // metafunction to find next viable PtxPlan specialization
  // -------------------------------------------------------
  // find the first sm_arch<K>::ver <= Arch that is available
  // for example if Arch = 520
  // and we don't have PtxPlan<520> but do have PtxPlan<350>
  // the metafunction will return PtxPlan<350>
 
#if 0 
  template <class T>
  class has_tuning
  {
    typedef char one;
    typedef long two;

    template <typename C>
    static one test(typename C::tuning*);    // typeof(&C::helloworld) ) ;
    template <typename C>
    static two test(...);

  public:
    enum
    {
      value = sizeof(test<T>(0)) == sizeof(char)
    };
  };
#else
  __THRUST_DEFINE_HAS_NESTED_TYPE(has_tuning, tuning)
  __THRUST_DEFINE_HAS_NESTED_TYPE(has_type, type)
#endif

  template <size_t, class, class, template <class> class>
  struct specialize_plan_find;


  // Tuning with 1 typename
  //
  template <size_t I,
            class Arch,
            template <class, class> class Tuning,
            class _0,
            template <class> class Plan>
  struct specialize_plan_find<I,
                              Arch,
                              Tuning<typename sm_arch<0>::type, _0>,
                              Plan>
      : detail::conditional<
            ((size_t)sm_arch<I>::type::ver <= (size_t)Arch::ver) &&
                has_type<Tuning<typename sm_arch<I>::type, _0> >::value,
            Plan<typename sm_arch<I>::type>,
            specialize_plan_find<I - 1,
                                 Arch,
                                 Tuning<typename sm_arch<0>::type, _0>,
                                 Plan> >::type
  {
  };

  template <class Arch,
            template <class, class> class Tuning,
            class _0,
            template <class> class Plan>
  struct specialize_plan_find<0,
                              Arch,
                              Tuning<typename sm_arch<0>::type, _0>,
                              Plan>
      : detail::enable_if<(size_t)sm_arch<0>::type::ver <= (size_t)Arch::ver,
                          Plan<typename sm_arch<0>::type> >::type {};
  
  // Tuning with 2 typenames
  //
  template <size_t I,
            class Arch,
            template <class, class, class> class Tuning,
            class _0, class _1,
            template <class> class Plan>
  struct specialize_plan_find<I,
                              Arch,
                              Tuning<typename sm_arch<0>::type, _0, _1>,
                              Plan>
      : detail::conditional<
            ((size_t)sm_arch<I>::type::ver <= (size_t)Arch::ver) &&
                has_type<Tuning<typename sm_arch<I>::type, _0, _1> >::value,
            Plan<typename sm_arch<I>::type>,
            specialize_plan_find<I - 1,
                                 Arch,
                                 Tuning<typename sm_arch<0>::type, _0, _1>,
                                 Plan> >::type
  {
  };

  // Dispatcher
  //
  template <class Arch,
            template <class, class, class> class Tuning,
            class _0, class _1, 
            template <class> class Plan>
  struct specialize_plan_find<0,
                              Arch,
                              Tuning<typename sm_arch<0>::type, _0, _1>,
                              Plan>
      : detail::enable_if<(size_t)sm_arch<0>::type::ver <= (size_t)Arch::ver,
                          Plan<typename sm_arch<0>::type> >::type {};

  template <class Arch, class _, template <class> class Plan>
  struct specialize_plan_impl
      : specialize_plan_find<sm_arch<>::count - 1,
                             Arch,
                             typename _::tuning,
                             Plan>
  {
  };

  template <template <class> class Plan, class Arch = THRUST_TUNING_ARCH>
  struct specialize_plan
      : detail::conditional<
            has_tuning<Plan<typename sm_arch<0>::type > >::value,
            specialize_plan_impl<Arch,
                                 Plan<typename sm_arch<0>::type>,
                                 Plan>,
            Plan<Arch> >::type 
  {
    typedef  typename
      detail::conditional<
            has_tuning<Plan<typename sm_arch<0>::type > >::value,
            specialize_plan_impl<Arch,
                                 Plan<typename sm_arch<0>::type>,
                                 Plan>,
            Plan<Arch> >::type  type;
  };
  template <template <class> class Plan, class Arch = THRUST_TUNING_ARCH>
  struct specialize_plan_msvc13_war
  {
    typedef  typename
      detail::conditional<
            has_tuning<Plan<typename sm_arch<0>::type > >::value,
            specialize_plan_impl<Arch,
                                 Plan<typename sm_arch<0>::type>,
                                 Plan>,
            Plan<Arch> >::type  type;
  };
  template <template <class> class Plan, class Arch = THRUST_TUNING_ARCH>
  struct specialize_plan_msvc10_war
  {
    typedef  
      detail::conditional<
            has_tuning<Plan<typename sm_arch<0>::type > >::value,
            specialize_plan_impl<Arch,
                                 Plan<typename sm_arch<0>::type>,
                                 Plan>,
            Plan<Arch> >  type;
  };


  /////////////////////////
  /////////////////////////
  /////////////////////////

  // retrieve temp storage size from an Agent
  // ------------------------------------
  // metafunction introspects Agent, and if it finds TempStorage type
  // it will return its size
 
  __THRUST_DEFINE_HAS_NESTED_TYPE(has_temp_storage, TempStorage)
  
  template <class Agent, class U>
  struct temp_storage_size_impl;

  template<class Agent>
  struct temp_storage_size_impl<Agent, detail::false_type>
  {
    enum { value = 0 };
  };

  template<class Agent>
  struct temp_storage_size_impl<Agent, detail::true_type>
  {
    enum { value = sizeof(typename Agent::TempStorage) };
  };

  template <class Agent>
  struct temp_storage_size
      : temp_storage_size_impl<Agent, typename has_temp_storage<Agent>::type>
  {};
  
  template<class Agent, size_t MAX_SHMEM>
  struct has_enough_shmem
  {
    enum
    {
      value =
          temp_storage_size<specialize_plan<Agent::template PtxPlan, typename sm_arch<0>::type> >::value <= MAX_SHMEM &&
          temp_storage_size<specialize_plan<Agent::template PtxPlan, typename sm_arch<1>::type> >::value <= MAX_SHMEM &&
          temp_storage_size<specialize_plan<Agent::template PtxPlan, typename sm_arch<2>::type> >::value <= MAX_SHMEM &&
          temp_storage_size<specialize_plan<Agent::template PtxPlan, typename sm_arch<3>::type> >::value <= MAX_SHMEM &&
          temp_storage_size<specialize_plan<Agent::template PtxPlan, typename sm_arch<4>::type> >::value <= MAX_SHMEM
    };
    typedef typename detail::conditional<value,
                                         detail::true_type,
                                         detail::false_type>::type type;
  };
  
  /////////////////////////
  /////////////////////////
  /////////////////////////

  // AgentPlan structure and helpers
  // --------------------------------
   
  struct AgentPlan
  {
    int block_threads;
    int items_per_thread;
    int items_per_tile;
    int shared_memory_size;
    int grid_size;

    THRUST_RUNTIME_FUNCTION
    AgentPlan()  {}

    THRUST_RUNTIME_FUNCTION
    AgentPlan(int block_threads_,
              int items_per_thread_,
              int shared_memory_size_,
              int grid_size_ = 0)
        : block_threads(block_threads_),
          items_per_thread(items_per_thread_),
          items_per_tile(items_per_thread * block_threads),
          shared_memory_size(shared_memory_size_),
          grid_size(grid_size_)
    {
    }

    THRUST_RUNTIME_FUNCTION
    AgentPlan(AgentPlan const& plan)
        : block_threads(plan.block_threads),
          items_per_thread(plan.items_per_thread),
          items_per_tile(plan.items_per_tile),
          shared_memory_size(plan.shared_memory_size),
          grid_size(plan.grid_size) {}

    template <class PtxPlan>
    THRUST_RUNTIME_FUNCTION
    AgentPlan(PtxPlan,
              typename detail::disable_if_convertible<
                  PtxPlan,
                  AgentPlan>::type* = NULL)
        : block_threads(PtxPlan::BLOCK_THREADS),
          items_per_thread(PtxPlan::ITEMS_PER_THREAD),
          items_per_tile(PtxPlan::ITEMS_PER_TILE),
          shared_memory_size(temp_storage_size<PtxPlan>::value),
          grid_size(0) {}
  }; // struct AgentPlan

  
  __THRUST_DEFINE_HAS_NESTED_TYPE(has_Plan, Plan)

  template<class Agent>
  struct return_Plan
  {
    typedef typename Agent::Plan type;
  };

  template<class Agent>
  struct get_plan : detail::conditional<
                    has_Plan<Agent>::value,
                    return_Plan<Agent>,
                    detail::identity_<AgentPlan> > ::type {};
 
  // returns AgentPlan corresponding to a given ptx version
  // ------------------------------------------------------
  
  template <class Agent>
  typename get_plan<Agent>::type THRUST_RUNTIME_FUNCTION
  get_agent_plan(int ptx_version)
  {
    typedef typename get_plan<Agent>::type Plan;
#if (CUB_PTX_ARCH > 0) && defined(__THRUST_HAS_CUDART__)
    // We're on device, use default policy
    return Plan(typename Agent::ptx_plan());
#else
    // order is imporant, check from highet to lowest SM version
    if (ptx_version >= 600)
    {
      return Plan(specialize_plan<Agent::template PtxPlan, sm60>());
    }
    else if (ptx_version >= 520)
    {
      return Plan(specialize_plan<Agent::template PtxPlan, sm52>());
    }
    else if (ptx_version >= 350)
    {
      return Plan(specialize_plan<Agent::template PtxPlan, sm35>());
    }
    else if (ptx_version >= 300)
    {
      return Plan(specialize_plan<Agent::template PtxPlan, sm30>());
    } 
    else
    {
      return Plan(specialize_plan<Agent::template PtxPlan, sm20>());
    }
#endif
  }    // function get_agent_config


  // if we don't know ptx version, we can call kernel
  // to retrieve AgentPlan from device code. Slower, but guaranteed to work
  // -----------------------------------------------------------------------
#if 0 
  template<class Agent>
  void __global__ get_agent_plan_kernel(AgentPlan *plan);

  static __device__ AgentPlan agent_plan_device;

  template<class Agent>
  AgentPlan __device__ get_agent_plan_dev()
  {
    AgentPlan plan;
    plan.block_threads      = Agent::ptx_plan::BLOCK_THREADS;
    plan.items_per_thread   = Agent::ptx_plan::ITEMS_PER_THREAD;
    plan.items_per_tile     = Agent::ptx_plan::ITEMS_PER_TILE;
    plan.shared_memory_size = temp_storage_size<typename Agent::ptx_plan>::value;
    return plan;
  }

  template <class Agent, class F>
  AgentPlan __host__ __device__ __forceinline__
  get_agent_plan_impl(F f, cudaStream_t s, void* d_ptr)
  {
    AgentPlan plan;
#ifdef __CUDA_ARCH__
    plan = get_agent_plan_dev<Agent>();
#else
    static cub::Mutex mutex;
    bool lock = false;
    if (d_ptr == 0)
    {
      lock = true;
      cudaGetSymbolAddress(&d_ptr, agent_plan_device);
    }
    if (lock)
      mutex.Lock();
    f<<<1,1,0,s>>>((AgentPlan*)d_ptr);
    cudaMemcpyAsync((void*)&plan,
                    d_ptr,
                    sizeof(AgentPlan),
                    cudaMemcpyDeviceToHost,
                    s);
    if (lock)
      mutex.Unlock();
    cudaStreamSynchronize(s);
#endif
    return plan;
  }

  template <class Agent>
  AgentPlan THRUST_RUNTIME_FUNCTION
  get_agent_plan(cudaStream_t s = 0, void *ptr = 0)
  {
    return get_agent_plan_impl<Agent>(get_agent_plan_kernel<Agent>,
                                        s,
                                        ptr);
  }

  template<class Agent>
  void __global__ get_agent_plan_kernel(AgentPlan *plan)
  {
    *plan = get_agent_plan_dev<Agent>();
  }
#endif

  /////////////////////////
  /////////////////////////
  /////////////////////////

  inline static cudaError_t CUB_RUNTIME_FUNCTION
  get_occ_device_properties(cudaOccDeviceProp &occ_prop, int dev_id)
  {
    cudaError_t status = cudaSuccess;
#ifdef __CUDA_ARCH__
    {
      cudaOccDeviceProp &o = occ_prop;
      //
      status = cudaDeviceGetAttribute(&o.computeMajor,
                                      cudaDevAttrComputeCapabilityMajor,
                                      dev_id);
      status = cudaDeviceGetAttribute(&o.computeMinor,
                                      cudaDevAttrComputeCapabilityMinor,
                                      dev_id);
      status = cudaDeviceGetAttribute(&o.maxThreadsPerBlock,
                                      cudaDevAttrMaxThreadsPerBlock,
                                      dev_id);
      status = cudaDeviceGetAttribute(&o.maxThreadsPerMultiprocessor,
                                      cudaDevAttrMaxThreadsPerMultiProcessor,
                                      dev_id);
      status = cudaDeviceGetAttribute(&o.regsPerBlock,
                                      cudaDevAttrMaxRegistersPerBlock,
                                      dev_id);
      status = cudaDeviceGetAttribute(&o.regsPerMultiprocessor,
                                      cudaDevAttrMaxRegistersPerMultiprocessor,
                                      dev_id);
      status = cudaDeviceGetAttribute(&o.warpSize,
                                      cudaDevAttrWarpSize,
                                      dev_id);

      int i32value;
      status = cudaDeviceGetAttribute(&i32value,
                                      cudaDevAttrMaxSharedMemoryPerBlock,
                                      dev_id);
      o.sharedMemPerBlock = static_cast<size_t>(i32value);

      status = cudaDeviceGetAttribute(&i32value,
                                      cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                      dev_id);
      o.sharedMemPerMultiprocessor = static_cast<size_t>(i32value);

      status = cudaDeviceGetAttribute(&o.numSms,
                                      cudaDevAttrMultiProcessorCount,
                                      dev_id);
    }
#else
    {
      cudaDeviceProp props;
      status   = cudaGetDeviceProperties(&props, dev_id);
      occ_prop = cudaOccDeviceProp(props);
    }
#endif
    return status;
  }
  
  int CUB_RUNTIME_FUNCTION
  inline get_sm_count()
  {
    int dev_id;
    cuda_cub::throw_on_error(cudaGetDevice(&dev_id),
                             "get_sm_count:"
                             "failed to cudaGetDevice");

    cudaError_t status;
    int         i32value;
    status = cudaDeviceGetAttribute(&i32value,
                                    cudaDevAttrMultiProcessorCount,
                                    dev_id);
    cuda_cub::throw_on_error(status,
                             "get_sm_count:"
                             "failed to sm_count");
    return i32value;
  }

  size_t CUB_RUNTIME_FUNCTION
  inline get_max_shared_memory_per_block()
  {
    int dev_id;
    cuda_cub::throw_on_error(cudaGetDevice(&dev_id),
                             "get_max_shared_memory_per_block :"
                             "failed to cudaGetDevice");

    cudaError_t status;
    int         i32value;
    status = cudaDeviceGetAttribute(&i32value,
                                    cudaDevAttrMaxSharedMemoryPerBlock,
                                    dev_id);
    cuda_cub::throw_on_error(status,
                             "get_max_shared_memory_per_block :"
                             "failed to get max shared memory per block");

    return static_cast<size_t>(i32value);
  }

  size_t CUB_RUNTIME_FUNCTION
  inline virtual_shmem_size(size_t shmem_per_block)
  {
    size_t max_shmem_per_block = core::get_max_shared_memory_per_block();
    if (shmem_per_block > max_shmem_per_block)
      return shmem_per_block;
    else
      return 0;
  }
  
  size_t CUB_RUNTIME_FUNCTION
  inline vshmem_size(size_t shmem_per_block, size_t num_blocks)
  {
    size_t max_shmem_per_block = core::get_max_shared_memory_per_block();
    if (shmem_per_block > max_shmem_per_block)
      return shmem_per_block*num_blocks;
    else
      return 0;
  }

  template <class Kernel>
  int CUB_RUNTIME_FUNCTION 
  get_max_block_size(Kernel k)
  {
    int devId;
    cuda_cub::throw_on_error(cudaGetDevice(&devId),
                   "get_max_block_size :"
                   "failed to cudaGetDevice");

    cudaOccDeviceProp occ_prop;
    cuda_cub::throw_on_error(get_occ_device_properties(occ_prop, devId),
                   "get_max_block_size: "
                   "failed to cudaGetDeviceProperties");


    cudaFuncAttributes attribs;
    cuda_cub::throw_on_error(cudaFuncGetAttributes(&attribs, reinterpret_cast<void *>(k)),
                   "get_max_block_size: "
                   "failed to cudaFuncGetAttributes");
    cudaOccFuncAttributes occ_attrib(attribs);


    cudaFuncCache cacheConfig;
    cuda_cub::throw_on_error(cudaDeviceGetCacheConfig(&cacheConfig),
                   "get_max_block_size: "
                   "failed to cudaDeviceGetCacheConfig");

    cudaOccDeviceState occ_state;
    occ_state.cacheConfig      = (cudaOccCacheConfig)cacheConfig;
    int          block_size    = 0;
    int          min_grid_size = 0;
    cudaOccError occ_status    = cudaOccMaxPotentialOccupancyBlockSize(&min_grid_size,
                                                                    &block_size,
                                                                    &occ_prop,
                                                                    &occ_attrib,
                                                                    &occ_state,
                                                                    0);
    if (CUDA_OCC_SUCCESS != occ_status || block_size <= 0)
      cuda_cub::throw_on_error(cudaErrorInvalidConfiguration,
                     "get_max_block_size: "
                     "failed to cudaOccMaxPotentialOccupancyBlockSize");

    return block_size;
  }
  
  // LoadIterator
  // ------------
  // if trivial iterator is passed, wrap loads into LDG
  //
  template <class PtxPlan, class It>
  struct LoadIterator
  {
    typedef typename iterator_traits<It>::value_type      value_type;
    typedef typename iterator_traits<It>::difference_type size_type;

    typedef typename detail::conditional<
        detail::is_trivial_iterator<It>::value,
        cub::CacheModifiedInputIterator<PtxPlan::LOAD_MODIFIER,
                                        value_type,
                                        size_type>,
        It>::type type;
  };    // struct Iterator

  template <class PtxPlan, class It>
  typename LoadIterator<PtxPlan, It>::type __device__ __forceinline__
  make_load_iterator_impl(It it, detail::true_type /* is_trivial */)
  {
    return raw_pointer_cast(&*it);
  }
  
  template <class PtxPlan, class It>
  typename LoadIterator<PtxPlan, It>::type __device__ __forceinline__
  make_load_iterator_impl(It it, detail::false_type /* is_trivial */)
  {
    return it;
  }

  template <class PtxPlan, class It>
  typename LoadIterator<PtxPlan, It>::type __device__ __forceinline__
  make_load_iterator(PtxPlan const&, It it)
  {
    return make_load_iterator_impl<PtxPlan>(
        it, typename detail::is_trivial_iterator<It>::type());
  }

  template<class>
  struct get_arch;

  template<template<class> class Plan, class Arch>
  struct get_arch<Plan<Arch> > { typedef Arch type; };

  // BlockLoad
  // -----------
  // a helper metaprogram that returns type of a block loader
  template <class PtxPlan,
            class It,
            class T    = typename iterator_traits<It>::value_type>
  struct BlockLoad
  {
    typedef cub::BlockLoadGeneric<T,
                                  It,
                                  PtxPlan::BLOCK_THREADS,
                                  PtxPlan::ITEMS_PER_THREAD,
                                  PtxPlan::LOAD_ALGORITHM,
                                  1,
                                  1,
                                  get_arch<PtxPlan>::type::ver >


        type;
  };
  
  // BlockStore
  // -----------
  // a helper metaprogram that returns type of a block loader
  template <class PtxPlan,
            class It,
            class T = typename iterator_traits<It>::value_type>
  struct BlockStore
  {
    typedef cub::BlockStoreGeneric<T,
                                   It,
                                   PtxPlan::BLOCK_THREADS,
                                   PtxPlan::ITEMS_PER_THREAD,
                                   PtxPlan::STORE_ALGORITHM,
                                   1,
                                   1,
                                   get_arch<PtxPlan>::type::ver>
        type;
  };
  // cuda_otional
  // --------------
  // used for function that return cudaError_t along with the result
  //
  template <class T>
  class cuda_optional
  {
    cudaError_t status_;
    T           value_;

  public:
    __host__ __device__
    cuda_optional() : status_(cudaSuccess) {}

    __host__ __device__
    cuda_optional(T v, cudaError_t status = cudaSuccess) : status_(status), value_(v) {}

    bool __host__ __device__
    isValid() const { return cudaSuccess == status_; }

    cudaError_t __host__ __device__
    status() const { return status_; }

    __host__ __device__ T const &
    value() const { return value_; }

    __host__ __device__ operator T const &() const { return value_; }
  };

  inline cuda_optional<int> CUB_RUNTIME_FUNCTION
  get_ptx_version()
  {
    int ptx_version = 0;
    cudaError_t status = cub::PtxVersion(ptx_version);
    return cuda_optional<int>(ptx_version, status);
  }

  inline cudaError_t CUB_RUNTIME_FUNCTION
  sync_stream(cudaStream_t stream)
  {
    return cub::SyncStream(stream);
  }

  inline void __device__ sync_threadblock()
  {
    __syncthreads();
  }

#define CUDA_CUB_RET_IF_FAIL(e) \
  if (cub::Debug((e), __FILE__, __LINE__)) return e;

  // uninitialized
  // -------
  // stores type in uninitialized form
  //
  template <class T>
  struct uninitialized
  {
    typedef typename cub::UnitWord<T>::DeviceWord DeviceWord;

    enum
    {
      WORDS = sizeof(T) / sizeof(DeviceWord)
    };

    DeviceWord storage[WORDS];

    __host__ __device__ __forceinline__ T& get()
    {
      return reinterpret_cast<T&>(*this);
    }

    __host__ __device__ __forceinline__ operator T&() { return get(); }
  };
  
  // uninitialized_array
  // --------------
  // allocates uninitialized data on stack
  template<class T, size_t N>
  struct array
  {
    typedef T value_type;
    typedef T ref[N];
    enum {SIZE = N};
    private:
      T data_[N];

    public:
      __host__ __device__ T* data() { return data_; }
      __host__ __device__ const T* data() const { return data_; }
      __host__ __device__ T& operator[](unsigned int idx) { return ((T*)data_)[idx]; }
      __host__ __device__ T const& operator[](unsigned int idx) const { return ((T*)data_)[idx]; }
      __host__ __device__ unsigned int size() const { return N; }
      __host__ __device__ operator ref&() { return data_; }
  };


  // uninitialized_array
  // --------------
  // allocates uninitialized data on stack
  template<class T, size_t N>
  struct uninitialized_array
  {
    typedef T value_type;
    typedef T ref[N];
    enum {SIZE = N};
    private:
      char data_[N * sizeof(T)];

    public:
      __host__ __device__ T* data() { return data_; }
      __host__ __device__ const T* data() const { return data_; }
      __host__ __device__ T& operator[](unsigned int idx) { return ((T*)data_)[idx]; }
      __host__ __device__ T const& operator[](unsigned int idx) const { return ((T*)data_)[idx]; }
      __host__ __device__ T& operator[](int idx) { return ((T*)data_)[idx]; }
      __host__ __device__ T const& operator[](int idx) const { return ((T*)data_)[idx]; }
      __host__ __device__ unsigned int size() const { return N; }
      __host__ __device__ operator ref&() { return *reinterpret_cast<ref*>(data_); }
      __host__ __device__ ref& get_ref() { return (ref&)*this; }
  };

  __host__ __device__ __forceinline__ size_t align_to(size_t n, size_t align)
  {
    return ((n+align-1)/align) * align;
  }

  namespace host {
    inline cuda_optional<size_t> get_max_shared_memory_per_block()
    {
      cudaError_t status = cudaSuccess;
      int         dev_id = 0;
      status             = cudaGetDevice(&dev_id);
      if (status != cudaSuccess) return cuda_optional<size_t>(0, status);

      int max_shmem = 0;
      status        = cudaDeviceGetAttribute(&max_shmem,
                                      cudaDevAttrMaxSharedMemoryPerBlock,
                                      dev_id);
      if (status != cudaSuccess) return cuda_optional<size_t>(0, status);
      return cuda_optional<size_t>(max_shmem, status);
    }
  }

  template <int           ALLOCATIONS>
  THRUST_RUNTIME_FUNCTION cudaError_t
  alias_storage(void*   storage_ptr,
                size_t& storage_size,
                void* (&allocations)[ALLOCATIONS],
                size_t (&allocation_sizes)[ALLOCATIONS])
  {
    return cub::AliasTemporaries(storage_ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
  }


}    // namespace core
using core::sm60;
using core::sm52;
using core::sm35;
using core::sm30;
using core::sm20;
} // namespace cuda_ 

END_NS_THRUST

