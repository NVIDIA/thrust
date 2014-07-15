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
 * cub::BlockScanSweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan across a range of tiles.
 */

#pragma once

#include <iterator>

#include "block_scan_prefix_operators.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_scan.cuh"
#include "../grid/grid_queue.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for BlockScanSweep
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    bool                        _LOAD_WARP_TIME_SLICING,        ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any load-related data transpositions (versus each warp having its own storage)
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    BlockStoreAlgorithm         _STORE_ALGORITHM,               ///< The BlockStore algorithm to use
    bool                        _STORE_WARP_TIME_SLICING,       ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any store-related data transpositions (versus each warp having its own storage)
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct BlockScanSweepPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
        LOAD_WARP_TIME_SLICING  = _LOAD_WARP_TIME_SLICING,      ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any load-related data transpositions (versus each warp having its own storage)
        STORE_WARP_TIME_SLICING = _STORE_WARP_TIME_SLICING,     ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any store-related data transpositions (versus each warp having its own storage)
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;          ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;           ///< Cache load modifier for reading input elements
    static const BlockStoreAlgorithm    STORE_ALGORITHM         = _STORE_ALGORITHM;         ///< The BlockStore algorithm to use
    static const BlockScanAlgorithm     SCAN_ALGORITHM    = _SCAN_ALGORITHM;    ///< The BlockScan algorithm to use
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockScanSweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan across a range of tiles.
 */
template <
    typename BlockScanSweepPolicy,      ///< Parameterized BlockScanSweepPolicy tuning policy type
    typename InputIterator,             ///< Random-access input iterator type
    typename OutputIterator,            ///< Random-access output iterator type
    typename ScanOp,                    ///< Scan functor type
    typename Identity,                  ///< Identity element type (cub::NullType for inclusive scan)
    typename Offset>                    ///< Signed integer type for global offsets
struct BlockScanSweep
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Tile status descriptor interface type
    typedef ScanTileState<T> ScanTileState;

    // Input iterator wrapper type
    typedef typename If<IsPointer<InputIterator>::VALUE,
            CacheModifiedInputIterator<BlockScanSweepPolicy::LOAD_MODIFIER, T, Offset>,    // Wrap the native input pointer with CacheModifiedInputIterator
            InputIterator>::Type                                                            // Directly use the supplied input iterator type
        WrappedInputIterator;

    // Constants
    enum
    {
        INCLUSIVE           = Equals<Identity, NullType>::VALUE,            // Inclusive scan if no identity type is provided
        BLOCK_THREADS       = BlockScanSweepPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockScanSweepPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,

        // Whether or not to sync after loading data
        SYNC_AFTER_LOAD     = (BlockScanSweepPolicy::LOAD_ALGORITHM != BLOCK_LOAD_DIRECT),

    };

    // Parameterized BlockLoad type
    typedef BlockLoad<
            WrappedInputIterator,
            BlockScanSweepPolicy::BLOCK_THREADS,
            BlockScanSweepPolicy::ITEMS_PER_THREAD,
            BlockScanSweepPolicy::LOAD_ALGORITHM,
            BlockScanSweepPolicy::LOAD_WARP_TIME_SLICING>
        BlockLoadT;

    // Parameterized BlockStore type
    typedef BlockStore<
            OutputIterator,
            BlockScanSweepPolicy::BLOCK_THREADS,
            BlockScanSweepPolicy::ITEMS_PER_THREAD,
            BlockScanSweepPolicy::STORE_ALGORITHM,
            BlockScanSweepPolicy::STORE_WARP_TIME_SLICING>
        BlockStoreT;

    // Parameterized BlockScan type
    typedef BlockScan<
            T,
            BlockScanSweepPolicy::BLOCK_THREADS,
            BlockScanSweepPolicy::SCAN_ALGORITHM>
        BlockScanT;

    // Callback type for obtaining tile prefix during block scan
    typedef BlockScanLookbackPrefixOp<
            T,
            ScanOp,
            ScanTileState>
        LookbackPrefixCallbackOp;

    // Stateful BlockScan prefix callback type for managing a running total while scanning consecutive tiles
    typedef BlockScanRunningPrefixOp<
            T,
            ScanOp>
        RunningPrefixCallbackOp;

    // Shared memory type for this threadblock
    struct _TempStorage
    {
        union
        {
            typename BlockLoadT::TempStorage    load;       // Smem needed for tile loading
            typename BlockStoreT::TempStorage   store;      // Smem needed for tile storing
            struct
            {
                typename LookbackPrefixCallbackOp::TempStorage  prefix;     // Smem needed for cooperative prefix callback
                typename BlockScanT::TempStorage                scan;       // Smem needed for tile scanning
            };
        };

        Offset tile_idx;   // Shared tile index
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage                &temp_storage;      ///< Reference to temp_storage
    WrappedInputIterator        d_in;               ///< Input data
    OutputIterator              d_out;              ///< Output data
    ScanOp                      scan_op;            ///< Binary scan operator
    Identity                    identity;           ///< Identity element



    //---------------------------------------------------------------------
    // Block scan utility methods (first tile)
    //---------------------------------------------------------------------

    /**
     * Exclusive scan specialization
     */
    template <typename _ScanOp, typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, _Identity identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate);
    }

    /**
     * Exclusive sum specialization
     */
    template <typename _Identity>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum scan_op, _Identity identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).ExclusiveSum(items, items, block_aggregate);
    }

    /**
     * Inclusive scan specialization
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, NullType identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate);
    }

    /**
     * Inclusive sum specialization
     */
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum scan_op, NullType identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).InclusiveSum(items, items, block_aggregate);
    }

    //---------------------------------------------------------------------
    // Block scan utility methods (subsequent tiles)
    //---------------------------------------------------------------------

    /**
     * Exclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp, typename _Identity, typename PrefixCallback>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, _Identity identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Exclusive sum specialization (with prefix from predecessors)
     */
    template <typename _Identity, typename PrefixCallback>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum scan_op, _Identity identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).ExclusiveSum(items, items, block_aggregate, prefix_op);
    }

    /**
     * Inclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp, typename PrefixCallback>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, NullType identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Inclusive sum specialization (with prefix from predecessors)
     */
    template <typename PrefixCallback>
    __device__ __forceinline__
    void ScanBlock(T (&items)[ITEMS_PER_THREAD], Sum scan_op, NullType identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).InclusiveSum(items, items, block_aggregate, prefix_op);
    }


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockScanSweep(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIterator               d_in,               ///< Input data
        OutputIterator              d_out,              ///< Output data
        ScanOp                      scan_op,            ///< Binary scan operator
        Identity                    identity)           ///< Identity element
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_out(d_out),
        scan_op(scan_op),
        identity(identity)
    {}


    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------

    /**
     * Process a tile of input (dynamic chained scan)
     */
    template <bool LAST_TILE>
    __device__ __forceinline__ void ConsumeTile(
        Offset              num_items,          ///< Total number of input items
        Offset              num_remaining,      ///< Total number of items remaining to be processed (including this tile)
        int                 tile_idx,           ///< Tile index
        Offset              block_offset,       ///< Tile offset
        ScanTileState       &tile_status)       ///< Global list of tile status
    {
        // Load items
        T items[ITEMS_PER_THREAD];

        if (LAST_TILE)
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items, num_remaining);
        else
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);

        if (SYNC_AFTER_LOAD)
            __syncthreads();

        // Perform tile scan
        if (tile_idx == 0)
        {
            // Scan first tile
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate);

            // Update tile status if there may be successor tiles (i.e., this tile is full)
            if (!LAST_TILE && (threadIdx.x == 0))
                tile_status.SetInclusive(0, block_aggregate);
        }
        else
        {
            // Scan non-first tile
            T block_aggregate;
            LookbackPrefixCallbackOp prefix_op(tile_status, temp_storage.prefix, scan_op, tile_idx);
            ScanBlock(items, scan_op, identity, block_aggregate, prefix_op);
        }

        __syncthreads();

        // Store items
        if (LAST_TILE)
            BlockStoreT(temp_storage.store).Store(d_out + block_offset, items, num_remaining);
        else
            BlockStoreT(temp_storage.store).Store(d_out + block_offset, items);
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic chained scan
     */
    __device__ __forceinline__ void ConsumeRange(
        int                 num_items,          ///< Total number of input items
        GridQueue<int>      queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        ScanTileState       &tile_status)       ///< Global list of tile status
    {
#if (CUB_PTX_ARCH <= 130)
        // Blocks are launched in increasing order, so just assign one tile per block

        int     tile_idx        = (blockIdx.y * gridDim.x) + blockIdx.x;    // Current tile index
        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;            // Global offset for the current tile
        Offset  num_remaining   = num_items - block_offset;                 // Remaining items (including this tile)

        if (num_remaining > TILE_ITEMS)
            ConsumeTile<false>(num_items, num_remaining, tile_idx, block_offset, tile_status);
        else if (num_remaining > 0)
            ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

#else
        // Blocks may not be launched in increasing order, so work-steal tiles

        // Get first tile index
        if (threadIdx.x == 0)
            temp_storage.tile_idx = queue.Drain(1);

        __syncthreads();

        int     tile_idx        = temp_storage.tile_idx;
        Offset  block_offset    = TILE_ITEMS * tile_idx;
        Offset  num_remaining   = num_items - block_offset;

        while (num_remaining >= TILE_ITEMS)
        {
            // Consume full tile
            ConsumeTile<false>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            // Get next tile
            if (threadIdx.x == 0)
                temp_storage.tile_idx = queue.Drain(1);

            __syncthreads();

            tile_idx        = temp_storage.tile_idx;
            block_offset    = TILE_ITEMS * tile_idx;
            num_remaining   = num_items - block_offset;
        }

        // Consume the last (and potentially partially-full) tile
        if (num_remaining > 0)
        {
            ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);
        }

#endif

    }


    //---------------------------------------------------------------------
    // Scan an sequence of consecutive tiles (independent of other thread blocks)
    //---------------------------------------------------------------------

    /**
     * Process a tile of input
     */
    template <
        bool                FULL_TILE,
        bool                FIRST_TILE>
    __device__ __forceinline__ void ConsumeTile(
        Offset                      block_offset,               ///< Tile offset
        RunningPrefixCallbackOp     &prefix_op,                 ///< Running prefix operator
        int                         valid_items = TILE_ITEMS)   ///< Number of valid items in the tile
    {
        // Load items
        T items[ITEMS_PER_THREAD];

        if (FULL_TILE)
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
        else
            BlockLoadT(temp_storage.load).Load(d_in + block_offset, items, valid_items);

        __syncthreads();

        // Block scan
        if (FIRST_TILE)
        {
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate);
            prefix_op.running_total = block_aggregate;
        }
        else
        {
            T block_aggregate;
            ScanBlock(items, scan_op, identity, block_aggregate, prefix_op);
        }

        __syncthreads();

        // Store items
        if (FULL_TILE)
            BlockStoreT(temp_storage.store).Store(d_out + block_offset, items);
        else
            BlockStoreT(temp_storage.store).Store(d_out + block_offset, items, valid_items);
    }


    /**
     * Scan a consecutive share of input tiles
     */
    __device__ __forceinline__ void ConsumeRange(
        Offset   block_offset,      ///< [in] Threadblock begin offset (inclusive)
        Offset   block_end)         ///< [in] Threadblock end offset (exclusive)
    {
        BlockScanRunningPrefixOp<T, ScanOp> prefix_op(scan_op);

        if (block_offset + TILE_ITEMS <= block_end)
        {
            // Consume first tile of input (full)
            ConsumeTile<true, true>(block_offset, prefix_op);
            block_offset += TILE_ITEMS;

            // Consume subsequent full tiles of input
            while (block_offset + TILE_ITEMS <= block_end)
            {
                ConsumeTile<true, false>(block_offset, prefix_op);
                block_offset += TILE_ITEMS;
            }

            // Consume a partially-full tile
            if (block_offset < block_end)
            {
                int valid_items = block_end - block_offset;
                ConsumeTile<false, false>(block_offset, prefix_op, valid_items);
            }
        }
        else
        {
            // Consume the first tile of input (partially-full)
            int valid_items = block_end - block_offset;
            ConsumeTile<false, true>(block_offset, prefix_op, valid_items);
        }
    }


    /**
     * Scan a consecutive share of input tiles, seeded with the specified prefix value
     */
    __device__ __forceinline__ void ConsumeRange(
        Offset  block_offset,                       ///< [in] Threadblock begin offset (inclusive)
        Offset  block_end,                          ///< [in] Threadblock end offset (exclusive)
        T       prefix)                             ///< [in] The prefix to apply to the scan segment
    {
        BlockScanRunningPrefixOp<T, ScanOp> prefix_op(prefix, scan_op);

        // Consume full tiles of input
        while (block_offset + TILE_ITEMS <= block_end)
        {
            ConsumeTile<true, false>(block_offset, prefix_op);
            block_offset += TILE_ITEMS;
        }

        // Consume a partially-full tile
        if (block_offset < block_end)
        {
            int valid_items = block_end - block_offset;
            ConsumeTile<false, false>(block_offset, prefix_op, valid_items);
        }
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

