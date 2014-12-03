
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceReduceByKey provides device-wide, parallel operations for reducing segments of values residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "device_scan_dispatch.cuh"
#include "../../block_sweep/block_reduce_by_key_sweep.cuh"
#include "../../thread/thread_operators.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../util_device.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Multi-block reduce-by-key sweep kernel entry point
 */
template <
    typename            BlockReduceSweepByKeyPolicy,            ///< Parameterized BlockReduceSweepByKeyPolicy tuning policy type
    typename            KeysInputIterator,                      ///< Random-access input iterator type for keys
    typename            UniqueOutputIterator,                   ///< Random-access output iterator type for keys
    typename            ValuesInputIterator,                    ///< Random-access input iterator type for values
    typename            AggregatesOutputIterator,               ///< Random-access output iterator type for values
    typename            NumRunsOutputIterator,                  ///< Output iterator type for recording number of segments encountered
    typename            ScanTileState,                          ///< Tile status interface type
    typename            EqualityOp,                             ///< Key equality operator type
    typename            ReductionOp,                            ///< Value reduction operator type
    typename            Offset>                                 ///< Signed integer type for global offsets
__launch_bounds__ (int(BlockReduceSweepByKeyPolicy::BLOCK_THREADS))
__global__ void DeviceReduceByKeySweepKernel(
    KeysInputIterator           d_keys_in,                      ///< [in] Pointer to the input sequence of keys
    UniqueOutputIterator        d_unique_out,                   ///< [out] Pointer to the output sequence of unique keys (one key per run)
    ValuesInputIterator         d_values_in,                    ///< [in] Pointer to the input sequence of corresponding values
    AggregatesOutputIterator    d_aggregates_out,               ///< [out] Pointer to the output sequence of value aggregates (one aggregate per run)
    NumRunsOutputIterator       d_num_runs_out,                     ///< [out] Pointer to total number of runs encountered (i.e., the length of d_unique_out)
    ScanTileState               tile_status,                    ///< [in] Tile status interface
    EqualityOp                  equality_op,                    ///< [in] Key equality operator
    ReductionOp                 reduction_op,                   ///< [in] Value reduction operator
    Offset                      num_items,                      ///< [in] Total number of items to select from
    int                         num_tiles,                      ///< [in] Total number of tiles for the entire problem
    GridQueue<int>              queue)                          ///< [in] Drain queue descriptor for dynamically mapping tile data onto thread blocks
{
    // Thread block type for reducing tiles of value segments
    typedef BlockReduceSweepByKey<
        BlockReduceSweepByKeyPolicy,
        KeysInputIterator,
        UniqueOutputIterator,
        ValuesInputIterator,
        AggregatesOutputIterator,
        EqualityOp,
        ReductionOp,
        Offset> BlockReduceSweepByKeyT;

    // Shared memory for BlockReduceSweepByKey
    __shared__ typename BlockReduceSweepByKeyT::TempStorage temp_storage;

    // Process tiles
    BlockReduceSweepByKeyT(temp_storage, d_keys_in, d_unique_out, d_values_in, d_aggregates_out, equality_op, reduction_op, num_items).ConsumeRange(
        num_tiles,
        queue,
        tile_status,
        d_num_runs_out);
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceReduceByKey
 */
template <
    typename    KeysInputIterator,               ///< Random-access input iterator type for keys
    typename    UniqueOutputIterator,              ///< Random-access output iterator type for keys
    typename    ValuesInputIterator,             ///< Random-access input iterator type for values
    typename    AggregatesOutputIterator,            ///< Random-access output iterator type for values
    typename    NumRunsOutputIterator,            ///< Output iterator type for recording number of segments encountered
    typename    EqualityOp,                     ///< Key equality operator type
    typename    ReductionOp,                    ///< Value reduction operator type
    typename    Offset>                         ///< Signed integer type for global offsets
struct DeviceReduceByKeyDispatch
{
    /******************************************************************************
     * Types and constants
     ******************************************************************************/

    // Data type of key input iterator
    typedef typename std::iterator_traits<KeysInputIterator>::value_type Key;

    // Data type of value input iterator
    typedef typename std::iterator_traits<ValuesInputIterator>::value_type Value;

    enum
    {
        INIT_KERNEL_THREADS     = 128,
        MAX_INPUT_BYTES         = CUB_MAX(sizeof(Key), sizeof(Value)),
        COMBINED_INPUT_BYTES    = sizeof(Key) + sizeof(Value),
    };

    // Tile status descriptor interface type
    typedef ReduceByKeyScanTileState<Value, Offset> ScanTileState;


    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 8,
            ITEMS_PER_THREAD            = (MAX_INPUT_BYTES <= 8) ? 8 : CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, ((NOMINAL_4B_ITEMS_PER_THREAD * 8) + COMBINED_INPUT_BYTES - 1) / COMBINED_INPUT_BYTES)),
        };

        typedef BlockReduceSweepByKeyPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                true,
                BLOCK_SCAN_WARP_SCANS>
            ReduceByKeyPolicy;
    };

    /// SM30
    struct Policy300
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 6,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, ((NOMINAL_4B_ITEMS_PER_THREAD * 8) + COMBINED_INPUT_BYTES - 1) / COMBINED_INPUT_BYTES)),
        };

        typedef BlockReduceSweepByKeyPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                true,
                BLOCK_SCAN_WARP_SCANS>
            ReduceByKeyPolicy;
    };

    /// SM20
    struct Policy200
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 13,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, ((NOMINAL_4B_ITEMS_PER_THREAD * 8) + COMBINED_INPUT_BYTES - 1) / COMBINED_INPUT_BYTES)),
        };

        typedef BlockReduceSweepByKeyPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                true,
                BLOCK_SCAN_WARP_SCANS>
            ReduceByKeyPolicy;
    };

    /// SM13
    struct Policy130
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 7,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, ((NOMINAL_4B_ITEMS_PER_THREAD * 8) + COMBINED_INPUT_BYTES - 1) / COMBINED_INPUT_BYTES)),
        };

        typedef BlockReduceSweepByKeyPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                true,
                BLOCK_SCAN_WARP_SCANS>
            ReduceByKeyPolicy;
    };

    /// SM10
    struct Policy100
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 5,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 8) / COMBINED_INPUT_BYTES)),
        };

        typedef BlockReduceSweepByKeyPolicy<
                64,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                true,
                BLOCK_SCAN_RAKING>
            ReduceByKeyPolicy;
    };


    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

#if (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_ARCH >= 130)
    typedef Policy130 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxReduceByKeyPolicy : PtxPolicy::ReduceByKeyPolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &device_reduce_by_key_sweep_config)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        device_reduce_by_key_sweep_config.template Init<PtxReduceByKeyPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 350)
        {
            device_reduce_by_key_sweep_config.template Init<typename Policy350::ReduceByKeyPolicy>();
        }
        else if (ptx_version >= 300)
        {
            device_reduce_by_key_sweep_config.template Init<typename Policy300::ReduceByKeyPolicy>();
        }
        else if (ptx_version >= 200)
        {
            device_reduce_by_key_sweep_config.template Init<typename Policy200::ReduceByKeyPolicy>();
        }
        else if (ptx_version >= 130)
        {
            device_reduce_by_key_sweep_config.template Init<typename Policy130::ReduceByKeyPolicy>();
        }
        else
        {
            device_reduce_by_key_sweep_config.template Init<typename Policy100::ReduceByKeyPolicy>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration.  Mirrors the constants within BlockReduceSweepByKeyPolicy.
     */
    struct KernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        BlockLoadAlgorithm      load_policy;
        bool                    two_phase_scatter;
        BlockScanAlgorithm      scan_algorithm;
        cudaSharedMemConfig     smem_config;

        template <typename BlockReduceSweepByKeyPolicy>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads               = BlockReduceSweepByKeyPolicy::BLOCK_THREADS;
            items_per_thread            = BlockReduceSweepByKeyPolicy::ITEMS_PER_THREAD;
            load_policy                 = BlockReduceSweepByKeyPolicy::LOAD_ALGORITHM;
            two_phase_scatter           = BlockReduceSweepByKeyPolicy::TWO_PHASE_SCATTER;
            scan_algorithm              = BlockReduceSweepByKeyPolicy::SCAN_ALGORITHM;
            smem_config                 = cudaSharedMemBankSizeEightByte;
        }

        CUB_RUNTIME_FUNCTION __forceinline__
        void Print()
        {
            printf("%d, %d, %d, %d, %d, %d",
                block_threads,
                items_per_thread,
                load_policy,
                two_phase_scatter,
                scan_algorithm,
                smem_config);
        }
    };


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide reduce-by-key using the
     * specified kernel functions.
     */
    template <
        typename                    DeviceScanInitKernelPtr,                ///< Function type of cub::DeviceScanInitKernel
        typename                    DeviceReduceByKeySweepKernelPtr>        ///< Function type of cub::DeviceReduceByKeySweepKernelPtr
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void                            *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIterator               d_keys_in,                          ///< [in] Pointer to the input sequence of keys
        UniqueOutputIterator            d_unique_out,                       ///< [out] Pointer to the output sequence of unique keys (one key per run)
        ValuesInputIterator             d_values_in,                        ///< [in] Pointer to the input sequence of corresponding values
        AggregatesOutputIterator        d_aggregates_out,                   ///< [out] Pointer to the output sequence of value aggregates (one aggregate per run)
        NumRunsOutputIterator           d_num_runs_out,                         ///< [out] Pointer to total number of runs encountered (i.e., the length of d_unique_out)
        EqualityOp                      equality_op,                        ///< [in] Key equality operator
        ReductionOp                     reduction_op,                       ///< [in] Value reduction operator
        Offset                          num_items,                          ///< [in] Total number of items to select from
        cudaStream_t                    stream,                             ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                            debug_synchronous,                  ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        int                             ptx_version,                        ///< [in] PTX version of dispatch kernels
        DeviceScanInitKernelPtr         device_scan_init_kernel,            ///< [in] Kernel function pointer to parameterization of cub::DeviceScanInitKernel
        DeviceReduceByKeySweepKernelPtr range_reduce_by_key_kernel,         ///< [in] Kernel function pointer to parameterization of cub::DeviceReduceByKeySweepKernel
        KernelConfig                    device_reduce_by_key_sweep_config)  ///< [in] Dispatch parameters that match the policy that \p range_reduce_by_key_kernel was compiled for
    {

#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

#else

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get device SM version
            int sm_version;
            if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Number of input tiles
            int tile_size = device_reduce_by_key_sweep_config.block_threads * device_reduce_by_key_sweep_config.items_per_thread;
            int num_tiles = (num_items + tile_size - 1) / tile_size;

            // Specify temporary storage allocation requirements
            size_t  allocation_sizes[2];
            if (CubDebug(error = ScanTileState::AllocationSize(num_tiles, allocation_sizes[0]))) break;    // bytes needed for tile status descriptors
            allocation_sizes[1] = GridQueue<int>::AllocationSize();                                             // bytes needed for grid queue descriptor

            // Compute allocation pointers into the single storage blob (or set the necessary size of the blob)
            void* allocations[2];
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Construct the tile status interface
            ScanTileState tile_status;
            if (CubDebug(error = tile_status.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;

            // Construct the grid queue descriptor
            GridQueue<int> queue(allocations[1]);

            // Log device_scan_init_kernel configuration
            int init_grid_size = (num_tiles + INIT_KERNEL_THREADS - 1) / INIT_KERNEL_THREADS;
            if (debug_synchronous) CubLog("Invoking device_scan_init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);

            // Invoke device_scan_init_kernel to initialize tile descriptors and queue descriptors
            device_scan_init_kernel<<<init_grid_size, INIT_KERNEL_THREADS, 0, stream>>>(
                queue,
                tile_status,
                num_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Get SM occupancy for range_reduce_by_key_kernel
            int range_reduce_by_key_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                range_reduce_by_key_sm_occupancy,            // out
                sm_version,
                range_reduce_by_key_kernel,
                device_reduce_by_key_sweep_config.block_threads))) break;

            // Get grid size for scanning tiles
            dim3 reduce_by_key_grid_size;
            if (ptx_version <= 130)
            {
                // Blocks are launched in order, so just assign one block per tile
                int max_dim_x = 32 * 1024;
                reduce_by_key_grid_size.z = 1;
                reduce_by_key_grid_size.y = (num_tiles + max_dim_x - 1) / max_dim_x;
                reduce_by_key_grid_size.x = CUB_MIN(num_tiles, max_dim_x);
            }
            else
            {
                // Blocks may not be launched in order, so use atomics
                int range_reduce_by_key_occupancy = range_reduce_by_key_sm_occupancy * sm_count;      // Whole-device occupancy for range_reduce_by_key_kernel
                reduce_by_key_grid_size.z = 1;
                reduce_by_key_grid_size.y = 1;
                reduce_by_key_grid_size.x = (num_tiles < range_reduce_by_key_occupancy) ?
                    num_tiles :                             // Not enough to fill the device with threadblocks
                    range_reduce_by_key_occupancy;         // Fill the device with threadblocks
            }

#if (CUB_PTX_ARCH == 0)
            // Get current smem bank configuration
            cudaSharedMemConfig original_smem_config;
            if (CubDebug(error = cudaDeviceGetSharedMemConfig(&original_smem_config))) break;
            cudaSharedMemConfig current_smem_config = original_smem_config;

            // Update smem config if necessary
            if (current_smem_config != device_reduce_by_key_sweep_config.smem_config)
            {
                if (CubDebug(error = cudaDeviceSetSharedMemConfig(device_reduce_by_key_sweep_config.smem_config))) break;
                current_smem_config = device_reduce_by_key_sweep_config.smem_config;
            }
#endif

            // Log range_reduce_by_key_kernel configuration
            if (debug_synchronous) CubLog("Invoking range_reduce_by_key_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                reduce_by_key_grid_size.x, reduce_by_key_grid_size.y, reduce_by_key_grid_size.z, device_reduce_by_key_sweep_config.block_threads, (long long) stream, device_reduce_by_key_sweep_config.items_per_thread, range_reduce_by_key_sm_occupancy);

            // Invoke range_reduce_by_key_kernel
            range_reduce_by_key_kernel<<<reduce_by_key_grid_size, device_reduce_by_key_sweep_config.block_threads, 0, stream>>>(
                d_keys_in,
                d_unique_out,
                d_values_in,
                d_aggregates_out,
                d_num_runs_out,
                tile_status,
                equality_op,
                reduction_op,
                num_items,
                num_tiles,
                queue);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

#if (CUB_PTX_ARCH == 0)
            // Reset smem config if necessary
            if (current_smem_config != original_smem_config)
            {
                if (CubDebug(error = cudaDeviceSetSharedMemConfig(original_smem_config))) break;
            }
#endif

        }
        while (0);

        return error;

#endif  // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void                        *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIterator           d_keys_in,                      ///< [in] Pointer to the input sequence of keys
        UniqueOutputIterator        d_unique_out,                   ///< [out] Pointer to the output sequence of unique keys (one key per run)
        ValuesInputIterator         d_values_in,                    ///< [in] Pointer to the input sequence of corresponding values
        AggregatesOutputIterator    d_aggregates_out,               ///< [out] Pointer to the output sequence of value aggregates (one aggregate per run)
        NumRunsOutputIterator       d_num_runs_out,                     ///< [out] Pointer to total number of runs encountered (i.e., the length of d_unique_out)
        EqualityOp                  equality_op,                    ///< [in] Key equality operator
        ReductionOp                 reduction_op,                   ///< [in] Value reduction operator
        Offset                      num_items,                      ///< [in] Total number of items to select from
        cudaStream_t                stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous)              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #if (CUB_PTX_ARCH == 0)
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_ARCH;
    #endif

            // Get kernel kernel dispatch configurations
            KernelConfig device_reduce_by_key_sweep_config;
            InitConfigs(ptx_version, device_reduce_by_key_sweep_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_keys_in,
                d_unique_out,
                d_values_in,
                d_aggregates_out,
                d_num_runs_out,
                equality_op,
                reduction_op,
                num_items,
                stream,
                debug_synchronous,
                ptx_version,
                DeviceScanInitKernel<Offset, ScanTileState>,
                DeviceReduceByKeySweepKernel<PtxReduceByKeyPolicy, KeysInputIterator, UniqueOutputIterator, ValuesInputIterator, AggregatesOutputIterator, NumRunsOutputIterator, ScanTileState, EqualityOp, ReductionOp, Offset>,
                device_reduce_by_key_sweep_config))) break;
        }
        while (0);

        return error;
    }
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


