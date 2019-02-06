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


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#include <thrust/system/cuda/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/functional.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>

#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/cub/device/device_scan.cuh>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>

THRUST_BEGIN_NS
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename AssociativeOperator>
__host__ __device__ OutputIterator
inclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               InputIterator                                               first,
               InputIterator                                               last,
               OutputIterator                                              result,
               AssociativeOperator                                         binary_op);

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename T,
          typename AssociativeOperator>
__host__ __device__ OutputIterator
exclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               InputIterator                                               first,
               InputIterator                                               last,
               OutputIterator                                              result,
               T                                                           init,
               AssociativeOperator                                         binary_op);
THRUST_END_NS

THRUST_BEGIN_NS
namespace cuda_cub {

namespace __scan {

  namespace mpl = thrust::detail::mpl::math;

  template<class>
  struct WarpSize { enum { value = 32 }; };

  template <int                      _BLOCK_THREADS,
            int                      _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm  _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier   _LOAD_MODIFIER    = cub::LOAD_DEFAULT,
            cub::BlockStoreAlgorithm _STORE_ALGORITHM  = cub::BLOCK_STORE_DIRECT,
            cub::BlockScanAlgorithm  _SCAN_ALGORITHM   = cub::BLOCK_SCAN_WARP_SCANS,
            int                      _MIN_BLOCKS       = 1>
  struct PtxPolicy
  {
    enum
    {
      BLOCK_THREADS    = _BLOCK_THREADS,
      ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
      ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD,
      MIN_BLOCKS       = _MIN_BLOCKS
    };

    static const cub::BlockLoadAlgorithm  LOAD_ALGORITHM  = _LOAD_ALGORITHM;
    static const cub::CacheLoadModifier   LOAD_MODIFIER   = _LOAD_MODIFIER;
    static const cub::BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
    static const cub::BlockScanAlgorithm  SCAN_ALGORITHM  = _SCAN_ALGORITHM;
  };    // struct PtxPolicy


  // Scale the number of warps to keep same amount of "tile" storage
  // as the nominal configuration for 4B data.  Minimum of two warps.
  //
  template<class Arch, int NOMINAL_4B_BLOCK_THREADS, class T>
  struct THRUST_BLOCK_THREADS
  {
    enum
    {
      value = mpl::min<int,
                       NOMINAL_4B_BLOCK_THREADS,
                       mpl::max<int,
                                3,
                                ((NOMINAL_4B_BLOCK_THREADS /
                                  WarpSize<Arch>::value) *
                                 4) /
                                    sizeof(T)>::value *
                           WarpSize<Arch>::value>::value
    };
  }; // struct THRUST_BLOCK_THREADS

  // If necessary, scale down number of items per thread to keep
  // the same amount of "tile" storage as the nominal configuration for 4B data.
  // Minimum 1 item per thread
  //
  template <class Arch,
            int NOMINAL_4B_ITEMS_PER_THREAD,
            int NOMINAL_4B_BLOCK_THREADS,
            class T>
  struct THRUST_ITEMS_PER_THREAD
  {
    enum
    {
      value = mpl::min<
          int,
          NOMINAL_4B_ITEMS_PER_THREAD,
          mpl::max<
              int,
              1,
              (NOMINAL_4B_ITEMS_PER_THREAD *
               NOMINAL_4B_BLOCK_THREADS * 4 / sizeof(T)) /
                  THRUST_BLOCK_THREADS<Arch,
                                       NOMINAL_4B_BLOCK_THREADS,
                                       T>::value>::value>::value
    };
  };


  template <class Arch, class T, class U>
  struct Tuning;
  
  template<class T, class U>
  struct Tuning<sm30,T,U>
  {
    typedef sm30 Arch;
    enum
    {
      NOMINAL_4B_BLOCK_THREADS    = 256,
      NOMINAL_4B_ITEMS_PER_THREAD = 9,
    };

    typedef PtxPolicy<THRUST_BLOCK_THREADS<Arch,
                                           NOMINAL_4B_BLOCK_THREADS,
                                           T>::value,
                      THRUST_ITEMS_PER_THREAD<Arch,
                                              NOMINAL_4B_ITEMS_PER_THREAD,
                                              NOMINAL_4B_BLOCK_THREADS,
                                              T>::value,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
                      cub::LOAD_DEFAULT,
                      cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
                      cub::BLOCK_SCAN_RAKING_MEMOIZE>
        type;
  };    // struct Tuning for sm30
  
  template<class T, class U>
  struct Tuning<sm35,T,U>
  {
    typedef sm35 Arch;
    enum
    {
      NOMINAL_4B_BLOCK_THREADS    = 128,
      NOMINAL_4B_ITEMS_PER_THREAD = 12,
    };

    typedef PtxPolicy<THRUST_BLOCK_THREADS<Arch,
                                           NOMINAL_4B_BLOCK_THREADS,
                                           T>::value,
                      THRUST_ITEMS_PER_THREAD<Arch,
                                              NOMINAL_4B_ITEMS_PER_THREAD,
                                              NOMINAL_4B_BLOCK_THREADS,
                                              T>::value,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
                      cub::LOAD_LDG,
                      cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
                      cub::BLOCK_SCAN_RAKING>
        type;
  };    // struct Tuning for sm35
  
  template<class T, class U>
  struct Tuning<sm52,T,U>
  {
    typedef sm52 Arch;
    enum
    {
      NOMINAL_4B_BLOCK_THREADS    = 128,
      NOMINAL_4B_ITEMS_PER_THREAD = 12,
    };

    typedef PtxPolicy<THRUST_BLOCK_THREADS<Arch,
                                           NOMINAL_4B_BLOCK_THREADS,
                                           T>::value,
                      THRUST_ITEMS_PER_THREAD<Arch,
                                              NOMINAL_4B_ITEMS_PER_THREAD,
                                              NOMINAL_4B_BLOCK_THREADS,
                                              T>::value,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
                      cub::LOAD_LDG,
                      cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
                      cub::BLOCK_SCAN_RAKING>
        type;
  };    // struct Tuning for sm52

  template <class InputIt,
            class OutputIt,
            class ScanOp,
            class Size,
            class T,
            class Inclusive>
  struct ScanAgent
  {
    typedef cub::ScanTileState<T> ScanTileState;
    typedef cub::BlockScanRunningPrefixOp<T, ScanOp> RunningPrefixCallback;

    template<class Arch>
    struct PtxPlan : Tuning<Arch,T,T>::type
    {
      typedef Tuning<Arch, T, T> tuning;


      typedef typename core::LoadIterator<PtxPlan, InputIt>::type LoadIt;
      typedef typename core::BlockLoad<PtxPlan, LoadIt, T>::type    BlockLoad;
      typedef typename core::BlockStore<PtxPlan, OutputIt, T>::type BlockStore;

      typedef cub::TilePrefixCallbackOp<T, ScanOp, ScanTileState, Arch::ver>
          TilePrefixCallback;
      typedef cub::BlockScan<T,
                             PtxPlan::BLOCK_THREADS,
                             PtxPlan::SCAN_ALGORITHM,
                             1,
                             1,
                             Arch::ver>
          BlockScan;

      union TempStorage
      {
        typename BlockLoad::TempStorage  load;
        typename BlockStore::TempStorage store;

        struct
        {
          typename TilePrefixCallback::TempStorage prefix;
          typename BlockScan::TempStorage          scan;
        };
      };    // struct TempStorage
    };    // struct PtxPlan
    typedef typename core::specialize_plan_msvc10_war<PtxPlan>::type::type ptx_plan;

    typedef typename ptx_plan::LoadIt             LoadIt;
    typedef typename ptx_plan::BlockLoad          BlockLoad;
    typedef typename ptx_plan::BlockStore         BlockStore;
    typedef typename ptx_plan::TilePrefixCallback TilePrefixCallback;
    typedef typename ptx_plan::BlockScan          BlockScan;
    typedef typename ptx_plan::TempStorage        TempStorage;

    enum
    {
      INCLUSIVE        = Inclusive::value,
      BLOCK_THREADS    = ptx_plan::BLOCK_THREADS,
      ITEMS_PER_THREAD = ptx_plan::ITEMS_PER_THREAD,
      ITEMS_PER_TILE   = ptx_plan::ITEMS_PER_TILE,

      SYNC_AFTER_LOAD = (ptx_plan::LOAD_ALGORITHM != cub::BLOCK_LOAD_DIRECT),
    };

    struct impl
    {
      //---------------------------------------------------------------------
      // Per thread data
      //---------------------------------------------------------------------

      TempStorage &storage;
      ScanTileState &tile_state;
      LoadIt load_it;
      OutputIt output_it;
      ScanOp scan_op;

      //---------------------------------------------------------------------
      // Block scan utility methods (first tile)
      //---------------------------------------------------------------------

      // Exclusive scan specialization
      //
      template <class _ScanOp>
      void THRUST_DEVICE_FUNCTION scan_tile(T (&items)[ITEMS_PER_THREAD],
                                            _ScanOp scan_op,
                                            T &     block_aggregate,
                                            thrust::detail::false_type /* is_inclusive */)
      {
        BlockScan(storage.scan).ExclusiveScan(items, items, scan_op, block_aggregate);
      }

      // Exclusive sum specialization
      //
      void THRUST_DEVICE_FUNCTION scan_tile(T (&items)[ITEMS_PER_THREAD],
                                            plus<T> /*scan_op*/,
                                            T &     block_aggregate,
                                            thrust::detail::false_type /* is_inclusive */)
      {
        BlockScan(storage.scan).ExclusiveSum(items, items, block_aggregate);
      }

      // Inclusive scan specialization
      //
      template <typename _ScanOp>
      void THRUST_DEVICE_FUNCTION scan_tile(T (&items)[ITEMS_PER_THREAD],
                                            _ScanOp scan_op,
                                            T &     block_aggregate,
                                            thrust::detail::true_type /* is_inclusive */)
      {
        BlockScan(storage.scan).InclusiveScan(items, items, scan_op, block_aggregate);
      }


      // Inclusive sum specialization
      //
      void THRUST_DEVICE_FUNCTION scan_tile(T (&items)[ITEMS_PER_THREAD],
                                            plus<T> /*scan_op*/,
                                            T &     block_aggregate,
                                            thrust::detail::true_type /* is_inclusive */)
      {
        BlockScan(storage.scan).InclusiveSum(items, items, block_aggregate);
      }

      //---------------------------------------------------------------------
      // Block scan utility methods (subsequent tiles)
      //---------------------------------------------------------------------

      // Exclusive scan specialization (with prefix from predecessors)
      //
      template <class _ScanOp, class PrefixCallback>
      void THRUST_DEVICE_FUNCTION scan_tile(T (&items)[ITEMS_PER_THREAD],
                                            _ScanOp         scan_op,
                                            T &             block_aggregate,
                                            PrefixCallback &prefix_op,
                                            thrust::detail::false_type /* is_inclusive */)
      {
        BlockScan(storage.scan).ExclusiveScan(items, items, scan_op, prefix_op);
        block_aggregate = prefix_op.GetBlockAggregate();
      }
  
      // Exclusive sum specialization (with prefix from predecessors)
      //
      template <class PrefixCallback>
      THRUST_DEVICE_FUNCTION void scan_tile(T (&items)[ITEMS_PER_THREAD],
                                            plus<T>         /*scan_op*/,
                                            T &             block_aggregate,
                                            PrefixCallback &prefix_op,
                                            thrust::detail::false_type /* is_inclusive */)
      {
        BlockScan(storage.scan).ExclusiveSum(items, items, prefix_op);
        block_aggregate = prefix_op.GetBlockAggregate();
      }

      // Inclusive scan specialization (with prefix from predecessors)
      //
      template <class _ScanOp, class PrefixCallback>
      THRUST_DEVICE_FUNCTION void scan_tile(T (&items)[ITEMS_PER_THREAD],
                                            _ScanOp         scan_op,
                                            T &             block_aggregate,
                                            PrefixCallback &prefix_op,
                                            thrust::detail::true_type /* is_inclusive */)
      {
        BlockScan(storage.scan).InclusiveScan(items, items, scan_op, prefix_op);
        block_aggregate = prefix_op.GetBlockAggregate();
      }

      // Inclusive sum specialization (with prefix from predecessors)
      //
      template <class U, class PrefixCallback>
      THRUST_DEVICE_FUNCTION void scan_tile(T (&items)[ITEMS_PER_THREAD],
                                            plus<T>         /*scan_op*/,
                                            T &             block_aggregate,
                                            PrefixCallback &prefix_op,
                                            thrust::detail::true_type /* is_inclusive */)
      {
        BlockScan(storage.scan).InclusiveSum(items, items, prefix_op);
        block_aggregate = prefix_op.GetBlockAggregate();
      }

      //---------------------------------------------------------------------
      // Cooperatively scan a device-wide sequence of tiles with other CTAs
      //---------------------------------------------------------------------

      // Process a tile of input (dynamic chained scan)
      //
      template <bool IS_FULL_TILE, class AddInitToExclusive>
      THRUST_DEVICE_FUNCTION void
      consume_tile(Size               /*num_items*/,
                   Size               num_remaining,
                   int                tile_idx,
                   Size               tile_base,
                   AddInitToExclusive add_init_to_exclusive_scan)
      {
        using core::sync_threadblock;

        // Load items
        T items[ITEMS_PER_THREAD];

        if (IS_FULL_TILE)
        {
          BlockLoad(storage.load).Load(load_it + tile_base, items);
        }
        else
        {
          // Fill last element with the first element
          // because collectives are not suffix guarded
          BlockLoad(storage.load)
              .Load(load_it + tile_base,
                    items,
                    num_remaining,
                    *(load_it + tile_base));
        }

        if (SYNC_AFTER_LOAD)
          sync_threadblock();

        // Perform tile scan
        if (tile_idx == 0)
        {
          // Scan first tile
          T block_aggregate;
          scan_tile(items, scan_op, block_aggregate, Inclusive());

          // Update tile status if there may be successor tiles (i.e., this tile is full)
          if (IS_FULL_TILE && (threadIdx.x == 0))
            tile_state.SetInclusive(0, block_aggregate);
        }
        else
        {
          // Scan non-first tile
          T                  block_aggregate;
          TilePrefixCallback prefix_op(tile_state, storage.prefix, scan_op, tile_idx);
          scan_tile(items, scan_op, block_aggregate, prefix_op, Inclusive());
        }

        sync_threadblock();

        add_init_to_exclusive_scan(items, tile_idx);

        // Store items
        if (IS_FULL_TILE)
        {
          BlockStore(storage.store).Store(output_it + tile_base, items);
        }
        else
        {
          BlockStore(storage.store).Store(output_it + tile_base, items, num_remaining);
        }
      }
      

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------
      
      // Dequeue and scan tiles of items as part of a dynamic chained scan
      // with Init
      template <class AddInitToExclusiveScan>
      THRUST_DEVICE_FUNCTION
      impl(TempStorage &          storage_,
           ScanTileState &        tile_state_,
           InputIt                input_it,
           OutputIt               output_it_,
           ScanOp                 scan_op_,
           Size                   num_items,
           AddInitToExclusiveScan add_init_to_exclusive_scan)
          : storage(storage_),
            tile_state(tile_state_),
            load_it(core::make_load_iterator(ptx_plan(), input_it)),
            output_it(output_it_),
            scan_op(scan_op_)
      {
        int  tile_idx      = blockIdx.x;
        Size tile_base     = ITEMS_PER_TILE * tile_idx;
        Size num_remaining = num_items - tile_base;

        if (num_remaining > ITEMS_PER_TILE)
        {
          // Full tile
          consume_tile<true>(num_items,
                             num_remaining,
                             tile_idx,
                             tile_base,
                             add_init_to_exclusive_scan);
        }
        else if (num_remaining > 0)
        {
          // Partially-full tile
          consume_tile<false>(num_items,
                              num_remaining,
                              tile_idx,
                              tile_base,
                              add_init_to_exclusive_scan);
        }
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    template <class AddInitToExclusiveScan>
    THRUST_AGENT_ENTRY(InputIt                input_it,
                       OutputIt               output_it,
                       ScanOp                 scan_op,
                       Size                   num_items,
                       ScanTileState          tile_state,
                       AddInitToExclusiveScan add_init_to_exclusive_scan,
                       char *                 shmem)
    {
      TempStorage &storage = *reinterpret_cast<TempStorage *>(shmem);
      impl(storage,
           tile_state,
           input_it,
           output_it,
           scan_op,
           num_items,
           add_init_to_exclusive_scan);
    }
  };    // struct ScanAgent

  template <class ScanTileState,
            class Size>
  struct InitAgent
  {
    template <class Arch>
    struct PtxPlan : PtxPolicy<128> {};
   
    typedef core::specialize_plan<PtxPlan> ptx_plan;

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    THRUST_AGENT_ENTRY(ScanTileState tile_state,
                       Size          num_tiles,
                       char *        /*shmem*/)
    {
      tile_state.InitializeStatus(num_tiles);
    }

  }; // struct InitAgent

  template<class T>
  struct DoNothing
  {
    typedef T     type;
    template <int ITEMS_PER_THREAD>
    THRUST_DEVICE_FUNCTION void
    operator()(T (&items)[ITEMS_PER_THREAD], int /*tile_idx*/)
    {
      THRUST_UNUSED_VAR(items);
    }
  };    // struct DoNothing

  template<class T, class ScanOp>
  struct AddInitToExclusiveScan
  {
    typedef T type;
    T         init;
    ScanOp    scan_op;

    THRUST_RUNTIME_FUNCTION
    AddInitToExclusiveScan(T init_, ScanOp scan_op_)
        : init(init_), scan_op(scan_op_) {}

    template <int ITEMS_PER_THREAD>
    THRUST_DEVICE_FUNCTION void
    operator()(T (&items)[ITEMS_PER_THREAD], int tile_idx)
    {
      if (tile_idx == 0 && threadIdx.x == 0)
      {
        items[0] = init;
        for (int i = 1; i < ITEMS_PER_THREAD; ++i)
          items[i] = scan_op(init, items[i]);
      }
      else
      {
        for (int i = 0; i < ITEMS_PER_THREAD; ++i)
          items[i] = scan_op(init, items[i]);
      }
    }
  };    // struct AddInitToExclusiveScan

  template <class Inclusive,
            class InputIt,
            class OutputIt,
            class ScanOp,
            class Size,
            class AddInitToExclusiveScan>
  static cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *                 d_temp_storage,
            size_t &               temp_storage_bytes,
            InputIt                input_it,
            Size                   num_items,
            AddInitToExclusiveScan add_init_to_exclusive_scan,
            OutputIt               output_it,
            ScanOp                 scan_op,
            cudaStream_t           stream,
            bool                   debug_sync)
  {
    using core::AgentPlan;
    using core::AgentLauncher;

    cudaError_t status = cudaSuccess;
    if (num_items == 0)
      return cudaErrorNotSupported;

    typedef typename AddInitToExclusiveScan::type T;

    typedef AgentLauncher<
        ScanAgent<InputIt, OutputIt, ScanOp, Size, T, Inclusive> >
        scan_agent;

    typedef typename scan_agent::ScanTileState ScanTileState;

    typedef AgentLauncher<InitAgent<ScanTileState, Size> > init_agent;

    AgentPlan scan_plan = scan_agent::get_plan(stream);
    AgentPlan init_plan = init_agent::get_plan();

    int tile_size = scan_plan.items_per_tile;
    Size num_tiles = static_cast<Size>((num_items + tile_size - 1) / tile_size);

    size_t vshmem_size = core::vshmem_size(scan_plan.shared_memory_size,
                                           num_tiles);

    size_t allocation_sizes[2] = {0, vshmem_size};
    status = ScanTileState::AllocationSize(static_cast<int>(num_tiles), allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    void* allocations[2] = {NULL, NULL};

    status = core::alias_storage(d_temp_storage,
                                 temp_storage_bytes,
                                 allocations,
                                 allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    if (d_temp_storage == NULL)
    {
      return status;
    }
    
    ScanTileState tile_state;
    status = tile_state.Init(static_cast<int>(num_tiles), allocations[0], allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    char *vshmem_ptr = vshmem_size > 0 ? (char*)allocations[1] : NULL;
    
    init_agent ia(init_plan, num_tiles, stream, "scan::init_agent", debug_sync);
    ia.launch(tile_state, num_tiles);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    scan_agent sa(scan_plan, num_items, stream, vshmem_ptr, "scan::scan_agent", debug_sync);
    sa.launch(input_it,
              output_it,
              scan_op,
              num_items,
              tile_state,
              add_init_to_exclusive_scan);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
    return status;
  }    // func doit_step

  template <typename Inclusive,
            typename Derived,
            typename InputIt,
            typename OutputIt,
            typename Size,
            typename ScanOp,
            typename AddInitToExclusiveScan>
  THRUST_RUNTIME_FUNCTION
  OutputIt scan(execution_policy<Derived>& policy,
                InputIt                    input_it,
                OutputIt                   output_it,
                Size                       num_items,
                ScanOp                     scan_op,
                AddInitToExclusiveScan     add_init_to_exclusive_scan)
  {
    if (num_items == 0)
      return output_it;

    size_t       storage_size = 0;
    cudaStream_t stream       = cuda_cub::stream(policy);
    bool         debug_sync   = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    status = doit_step<Inclusive>(NULL,
                                  storage_size,
                                  input_it,
                                  num_items,
                                  add_init_to_exclusive_scan,
                                  output_it,
                                  scan_op,
                                  stream,
                                  debug_sync);
    cuda_cub::throw_on_error(status, "scan failed on 1st step");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = doit_step<Inclusive>(ptr,
                                  storage_size,
                                  input_it,
                                  num_items,
                                  add_init_to_exclusive_scan,
                                  output_it,
                                  scan_op,
                                  stream,
                                  debug_sync);
    cuda_cub::throw_on_error(status, "scan failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "scan failed to synchronize");

    return output_it + num_items;
  }    // func scan

}    // namespace __scan

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class Size,
          class OutputIt,
          class ScanOp>
OutputIt __host__ __device__
inclusive_scan_n(execution_policy<Derived> &policy,
                 InputIt                    first,
                 Size                       num_items,
                 OutputIt                   result,
                 ScanOp                     scan_op)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    typedef typename iterator_traits<InputIt>::value_type T;
    ret = __scan::scan<thrust::detail::true_type>(policy,
                                                  first,
                                                  result,
                                                  num_items,
                                                  scan_op,
                                                  __scan::DoNothing<T>());
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::inclusive_scan(cvt_to_seq(derived_cast(policy)),
                                 first,
                                 first + num_items,
                                 result,
                                 scan_op);
#endif
  }
  return ret;
}


template <class Derived,
          class InputIt,
          class OutputIt,
          class ScanOp>
OutputIt __host__ __device__
inclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               InputIt                    last,
               OutputIt                   result,
               ScanOp                     scan_op)
{
  int num_items = static_cast<int>(thrust::distance(first, last));
  return cuda_cub::inclusive_scan_n(policy, first, num_items, result, scan_op);
}


template <class Derived,
          class InputIt,
          class OutputIt>
OutputIt __host__ __device__
inclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               OutputIt                   last,
               OutputIt                   result)
{

  typedef typename thrust::detail::eval_if<
      thrust::detail::is_output_iterator<OutputIt>::value,
      thrust::iterator_value<InputIt>,
      thrust::iterator_value<OutputIt> >::type result_type;
  return cuda_cub::inclusive_scan(policy, first, last, result, plus<result_type>());
};

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class Size,
          class OutputIt,
          class T,
          class ScanOp>
OutputIt __host__ __device__
exclusive_scan_n(execution_policy<Derived> &policy,
                 InputIt                    first,
                 Size                       num_items,
                 OutputIt                   result,
                 T                          init,
                 ScanOp                     scan_op)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __scan::scan<thrust::detail::false_type>(
        policy,
        first,
        result,
        num_items,
        scan_op,
        __scan::AddInitToExclusiveScan<T, ScanOp>(init, scan_op));
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::exclusive_scan(cvt_to_seq(derived_cast(policy)),
                                 first,
                                 first + num_items,
                                 result,
                                 init,
                                 scan_op);
#endif
  }
  return ret;
}

template <class Derived,
          class InputIt,
          class OutputIt,
          class T,
          class ScanOp>
OutputIt __host__ __device__
exclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               InputIt                    last,
               OutputIt                   result,
               T                          init,
               ScanOp                   scan_op)
{
  int num_items = static_cast<int>(thrust::distance(first, last));
  return cuda_cub::exclusive_scan_n(policy, first, num_items, result, init, scan_op);
}

template <class Derived,
          class InputIt,
          class OutputIt,
          class T>
OutputIt __host__ __device__
exclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               OutputIt                   last,
               OutputIt                   result,
               T                          init)
{
  return cuda_cub::exclusive_scan(policy, first, last, result, init, plus<T>());
}

template <class Derived,
          class InputIt,
          class OutputIt>
OutputIt __host__ __device__
exclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               OutputIt                   last,
               OutputIt                   result)
{
  typedef typename thrust::detail::eval_if<
      thrust::detail::is_output_iterator<OutputIt>::value,
      thrust::iterator_value<InputIt>,
      thrust::iterator_value<OutputIt>
  >::type result_type;
  return cuda_cub::exclusive_scan(policy, first, last, result, result_type(0));
};

} // namespace cuda_cub
THRUST_END_NS

#include <thrust/scan.h>

#endif
