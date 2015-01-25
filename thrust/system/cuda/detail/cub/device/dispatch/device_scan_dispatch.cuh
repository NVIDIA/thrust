
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
 * cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "../../block_sweep/block_scan_sweep.cuh"
#include "../../thread/thread_operators.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../util_debug.cuh"
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
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename            Offset,                 ///< Signed integer type for global offsets
    typename            ScanTileState>          ///< Tile status interface type
__global__ void DeviceScanInitKernel(
    GridQueue<Offset>   grid_queue,             ///< [in] Descriptor for performing dynamic mapping of input tiles to thread blocks
    ScanTileState       tile_status,            ///< [in] Tile status interface
    int                 num_tiles)              ///< [in] Number of tiles
{
    // Reset queue descriptor
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
        grid_queue.FillAndResetDrain(num_tiles);

    // Initialize tile status
    tile_status.InitializeStatus(num_tiles);
}


/**
 * Scan kernel entry point (multi-block)
 */
template <
    typename            BlockScanSweepPolicy,       ///< Parameterized BlockScanSweepPolicy tuning policy type
    typename            InputIterator,              ///< Random-access input iterator type for reading scan inputs \iterator
    typename            OutputIterator,             ///< Random-access output iterator type for writing scan outputs \iterator
    typename            ScanTileState,              ///< Tile status interface type
    typename            ScanOp,                     ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename            Identity,                   ///< Identity value type (cub::NullType for inclusive scans)
    typename            Offset>                     ///< Signed integer type for global offsets
__launch_bounds__ (int(BlockScanSweepPolicy::BLOCK_THREADS))
__global__ void DeviceScanSweepKernel(
    InputIterator       d_in,                       ///< Input data
    OutputIterator      d_out,                      ///< Output data
    ScanTileState       tile_status,                ///< [in] Tile status interface
    ScanOp              scan_op,                    ///< Binary scan functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
    Identity            identity,                   ///< Identity element
    Offset              num_items,                  ///< Total number of scan items for the entire problem
    GridQueue<int>      queue)                      ///< Drain queue descriptor for dynamically mapping tile data onto thread blocks
{
    // Thread block type for scanning input tiles
    typedef BlockScanSweep<
        BlockScanSweepPolicy,
        InputIterator,
        OutputIterator,
        ScanOp,
        Identity,
        Offset> BlockScanSweepT;

    // Shared memory for BlockScanSweep
    __shared__ typename BlockScanSweepT::TempStorage temp_storage;

    // Process tiles
    BlockScanSweepT(temp_storage, d_in, d_out, scan_op, identity).ConsumeRange(
        num_items,
        queue,
        tile_status);
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceScan
 */
template <
    typename InputIterator,      ///< Random-access input iterator type for reading scan inputs \iterator
    typename OutputIterator,     ///< Random-access output iterator type for writing scan outputs \iterator
    typename ScanOp,             ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename Identity,           ///< Identity value type (cub::NullType for inclusive scans)
    typename Offset>             ///< Signed integer type for global offsets
struct DeviceScanDispatch
{
    enum
    {
        INIT_KERNEL_THREADS     = 128
    };

    // Data type
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Tile status descriptor interface type
    typedef ScanTileState<T> ScanTileState;


    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 12,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // GTX Titan: 29.5B items/s (232.4 GB/s) @ 48M 32-bit T
        typedef BlockScanSweepPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_DIRECT,
                false,
                LOAD_LDG,
                BLOCK_STORE_WARP_TRANSPOSE,
                true,
                BLOCK_SCAN_RAKING_MEMOIZE>
            RangeScanPolicy;
    };

    /// SM30
    struct Policy300
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 9,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        typedef BlockScanSweepPolicy<
                256,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                false,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                false,
                BLOCK_SCAN_RAKING_MEMOIZE>
            RangeScanPolicy;
    };

    /// SM20
    struct Policy200
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 15,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // GTX 580: 20.3B items/s (162.3 GB/s) @ 48M 32-bit T
        typedef BlockScanSweepPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                false,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                false,
                BLOCK_SCAN_RAKING_MEMOIZE>
            RangeScanPolicy;
    };

    /// SM13
    struct Policy130
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 21,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        typedef BlockScanSweepPolicy<
                96,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                false,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                false,
                BLOCK_SCAN_RAKING_MEMOIZE>
            RangeScanPolicy;
    };

    /// SM10
    struct Policy100
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 9,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        typedef BlockScanSweepPolicy<
                64,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                true,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                true,
                BLOCK_SCAN_WARP_SCANS>
            RangeScanPolicy;
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
    struct PtxRangeScanPolicy : PtxPolicy::RangeScanPolicy {};


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
        KernelConfig    &device_scan_sweep_config)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        device_scan_sweep_config.template Init<PtxRangeScanPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 350)
        {
            device_scan_sweep_config.template Init<typename Policy350::RangeScanPolicy>();
        }
        else if (ptx_version >= 300)
        {
            device_scan_sweep_config.template Init<typename Policy300::RangeScanPolicy>();
        }
        else if (ptx_version >= 200)
        {
            device_scan_sweep_config.template Init<typename Policy200::RangeScanPolicy>();
        }
        else if (ptx_version >= 130)
        {
            device_scan_sweep_config.template Init<typename Policy130::RangeScanPolicy>();
        }
        else
        {
            device_scan_sweep_config.template Init<typename Policy100::RangeScanPolicy>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration.  Mirrors the constants within BlockScanSweepPolicy.
     */
    struct KernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        BlockLoadAlgorithm      load_policy;
        BlockStoreAlgorithm     store_policy;
        BlockScanAlgorithm      scan_algorithm;

        template <typename BlockScanSweepPolicy>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads               = BlockScanSweepPolicy::BLOCK_THREADS;
            items_per_thread            = BlockScanSweepPolicy::ITEMS_PER_THREAD;
            load_policy                 = BlockScanSweepPolicy::LOAD_ALGORITHM;
            store_policy                = BlockScanSweepPolicy::STORE_ALGORITHM;
            scan_algorithm              = BlockScanSweepPolicy::SCAN_ALGORITHM;
        }

        CUB_RUNTIME_FUNCTION __forceinline__
        void Print()
        {
            printf("%d, %d, %d, %d, %d",
                block_threads,
                items_per_thread,
                load_policy,
                store_policy,
                scan_algorithm);
        }
    };


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide prefix scan using the
     * specified kernel functions.
     */
    template <
        typename                    DeviceScanInitKernelPtr,        ///< Function type of cub::DeviceScanInitKernel
        typename                    DeviceScanSweepKernelPtr>       ///< Function type of cub::DeviceScanSweepKernelPtr
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void                        *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                           ///< [in] Pointer to the input sequence of data items
        OutputIterator              d_out,                          ///< [out] Pointer to the output sequence of data items
        ScanOp                      scan_op,                        ///< [in] Binary scan functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
        Identity                    identity,                       ///< [in] Identity element
        Offset                      num_items,                      ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t                stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous,              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        int                         ptx_version,                    ///< [in] PTX version of dispatch kernels
        DeviceScanInitKernelPtr     device_scan_init_kernel,        ///< [in] Kernel function pointer to parameterization of cub::DeviceScanInitKernel
        DeviceScanSweepKernelPtr    device_scan_sweep_kernel,       ///< [in] Kernel function pointer to parameterization of cub::DeviceScanSweepKernel
        KernelConfig                device_scan_sweep_config)       ///< [in] Dispatch parameters that match the policy that \p device_scan_sweep_kernel was compiled for
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
            int tile_size = device_scan_sweep_config.block_threads * device_scan_sweep_config.items_per_thread;
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

            // Get SM occupancy for device_scan_sweep_kernel
            int range_scan_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                range_scan_sm_occupancy,            // out
                sm_version,
                device_scan_sweep_kernel,
                device_scan_sweep_config.block_threads))) break;

            // Get grid size for scanning tiles
            dim3 scan_grid_size;
            if (ptx_version <= 130)
            {
                // Blocks are launched in order, so just assign one block per tile
                int max_dim_x = 32 * 1024;
                scan_grid_size.z = 1;
                scan_grid_size.y = (num_tiles + max_dim_x - 1) / max_dim_x;
                scan_grid_size.x = CUB_MIN(num_tiles, max_dim_x);
            }
            else
            {
                // Blocks may not be launched in order, so use atomics
                int range_scan_occupancy = range_scan_sm_occupancy * sm_count;        // Whole-device occupancy for device_scan_sweep_kernel
                scan_grid_size.z = 1;
                scan_grid_size.y = 1;
                scan_grid_size.x = (num_tiles < range_scan_occupancy) ?
                    num_tiles :                     // Not enough to fill the device with threadblocks
                    range_scan_occupancy;          // Fill the device with threadblocks
            }

            // Log device_scan_sweep_kernel configuration
            if (debug_synchronous) CubLog("Invoking device_scan_sweep_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                scan_grid_size.x, scan_grid_size.y, scan_grid_size.z, device_scan_sweep_config.block_threads, (long long) stream, device_scan_sweep_config.items_per_thread, range_scan_sm_occupancy);

            // Invoke device_scan_sweep_kernel
            device_scan_sweep_kernel<<<scan_grid_size, device_scan_sweep_config.block_threads, 0, stream>>>(
                d_in,
                d_out,
                tile_status,
                scan_op,
                identity,
                num_items,
                queue);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
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
        void            *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator   d_in,                           ///< [in] Pointer to the input sequence of data items
        OutputIterator  d_out,                          ///< [out] Pointer to the output sequence of data items
        ScanOp          scan_op,                        ///< [in] Binary scan functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
        Identity        identity,                       ///< [in] Identity element
        Offset          num_items,                      ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream,                         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous)              ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
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
            KernelConfig device_scan_sweep_config;
            InitConfigs(ptx_version, device_scan_sweep_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                scan_op,
                identity,
                num_items,
                stream,
                debug_synchronous,
                ptx_version,
                DeviceScanInitKernel<Offset, ScanTileState>,
                DeviceScanSweepKernel<PtxRangeScanPolicy, InputIterator, OutputIterator, ScanTileState, ScanOp, Identity, Offset>,
                device_scan_sweep_config))) break;
        }
        while (0);

        return error;
    }
};



}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


