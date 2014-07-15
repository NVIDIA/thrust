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
 * cub::BlockSelectSweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide select.
 */

#pragma once

#include <iterator>

#include "block_scan_prefix_operators.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_scan.cuh"
#include "../block/block_exchange.cuh"
#include "../block/block_discontinuity.cuh"
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
 * Parameterizable tuning policy type for BlockSelectSweep
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    bool                        _STORE_WARP_TIME_SLICING,       ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any store-related data transpositions (versus each warp having its own storage)
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct BlockSelectSweepPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
        STORE_WARP_TIME_SLICING = _STORE_WARP_TIME_SLICING,     ///< Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any store-related data transpositions (versus each warp having its own storage)
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;      ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;      ///< The BlockScan algorithm to use
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockSelectSweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide selection across a range of tiles
 *
 * Performs functor-based selection if SelectOp functor type != NullType
 * Otherwise performs flag-based selection if FlagsInputIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 */
template <
    typename    BlockSelectSweepPolicy,         ///< Parameterized BlockSelectSweepPolicy tuning policy type
    typename    InputIterator,                  ///< Random-access input iterator type for selection items
    typename    FlagsInputIterator,                   ///< Random-access input iterator type for selections (NullType* if a selection functor or discontinuity flagging is to be used for selection)
    typename    SelectedOutputIterator,                 ///< Random-access input iterator type for selected items
    typename    SelectOp,                       ///< Selection operator type (NullType if selections or discontinuity flagging is to be used for selection)
    typename    EqualityOp,                     ///< Equality operator type (NullType if selection functor or selections is to be used for selection)
    typename    Offset,                         ///< Signed integer type for global offsets
    bool        KEEP_REJECTS>                   ///< Whether or not we push rejected items to the back of the output
struct BlockSelectSweep
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Data type of flag iterator
    typedef typename std::iterator_traits<FlagsInputIterator>::value_type Flag;

    // Tile status descriptor interface type
    typedef ScanTileState<Offset> ScanTileState;

    // Constants
    enum
    {
        USE_SELECT_OP,
        USE_SELECT_FLAGS,
        USE_DISCONTINUITY,

        BLOCK_THREADS           = BlockSelectSweepPolicy::BLOCK_THREADS,

        /// Number of warp threads
        WARP_THREADS            = CUB_WARP_THREADS(PTX_ARCH),

        /// Number of active warps
        WARPS                   = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        ITEMS_PER_THREAD        = BlockSelectSweepPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,

        /// Whether or not to sync after loading data
        SYNC_AFTER_LOAD         = (BlockSelectSweepPolicy::LOAD_ALGORITHM != BLOCK_LOAD_DIRECT),

        /// Whether or not only one warp's worth of shared memory should be allocated and time-sliced among block-warps during any store-related data transpositions (versus each warp having its own storage)
        STORE_WARP_TIME_SLICING = BlockSelectSweepPolicy::STORE_WARP_TIME_SLICING,
        ACTIVE_EXCHANGE_WARPS   = (STORE_WARP_TIME_SLICING) ? 1 : WARPS,

        SELECT_METHOD           = (!Equals<SelectOp, NullType>::VALUE) ?
                                    USE_SELECT_OP :
                                    (!Equals<Flag, NullType>::VALUE) ?
                                        USE_SELECT_FLAGS :
                                        USE_DISCONTINUITY
    };

    // Input iterator wrapper type
    typedef typename If<IsPointer<InputIterator>::VALUE,
            CacheModifiedInputIterator<BlockSelectSweepPolicy::LOAD_MODIFIER, T, Offset>,      // Wrap the native input pointer with CacheModifiedInputIterator
            InputIterator>::Type                                                                // Directly use the supplied input iterator type
        WrappedInputIterator;

    // Flag iterator wrapper type
    typedef typename If<IsPointer<FlagsInputIterator>::VALUE,
            CacheModifiedInputIterator<BlockSelectSweepPolicy::LOAD_MODIFIER, Flag, Offset>,   // Wrap the native input pointer with CacheModifiedInputIterator
            FlagsInputIterator>::Type                                                                 // Directly use the supplied input iterator type
        WrappedFlagsInputIterator;

    // Parameterized BlockLoad type for input items
    typedef BlockLoad<
            WrappedInputIterator,
            BlockSelectSweepPolicy::BLOCK_THREADS,
            BlockSelectSweepPolicy::ITEMS_PER_THREAD,
            BlockSelectSweepPolicy::LOAD_ALGORITHM>
        BlockLoadT;

    // Parameterized BlockLoad type for flags
    typedef BlockLoad<
            WrappedFlagsInputIterator,
            BlockSelectSweepPolicy::BLOCK_THREADS,
            BlockSelectSweepPolicy::ITEMS_PER_THREAD,
            BlockSelectSweepPolicy::LOAD_ALGORITHM>
        BlockLoadFlags;

    // Parameterized BlockDiscontinuity type for input items
    typedef BlockDiscontinuity<T, BLOCK_THREADS> BlockDiscontinuityT;

    // Parameterized WarpScan
    typedef WarpScan<Offset> WarpScanAllocations;

    // Callback type for obtaining tile prefix during block scan
    typedef BlockScanLookbackPrefixOp<
            Offset,
            Sum,
            ScanTileState>
        LookbackPrefixCallbackOp;

    // Warp exchange type
    typedef WarpExchange<T, ITEMS_PER_THREAD> WarpExchangeT;

    // Shared memory type for this threadblock
    struct _TempStorage
    {
        union
        {
            struct
            {
                typename BlockDiscontinuityT::TempStorage       discontinuity;              // Smem needed for discontinuity detection
                typename WarpScanAllocations::TempStorage       warp_scan[WARPS];           // Smem needed for warp-synchronous scans
                Offset                                          warp_aggregates[WARPS];     // Smem needed for sharing warp-wide aggregates
                typename LookbackPrefixCallbackOp::TempStorage  prefix;                     // Smem needed for cooperative prefix callback
            };

            // Smem needed for input loading
            typename BlockLoadT::TempStorage        load_items;

            // Smem needed for flag loading
            typename BlockLoadFlags::TempStorage    load_flags;

            // Smem needed for two-phase scatter
            union
            {
                unsigned long long                  align;
                typename WarpExchangeT::TempStorage exchange[ACTIVE_EXCHANGE_WARPS];
            };
        };

        Offset      tile_idx;                   // Shared tile index
        Offset      tile_inclusive;             // Inclusive tile prefix
        Offset      tile_exclusive;             // Exclusive tile prefix
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage                    &temp_storage;      ///< Reference to temp_storage
    WrappedInputIterator            d_in;               ///< Input data
    WrappedFlagsInputIterator       d_flags;            ///< Input flags
    SelectedOutputIterator          d_selected_out;     ///< Output data
    SelectOp                        select_op;          ///< Selection operator
    InequalityWrapper<EqualityOp>   inequality_op;      ///< Inequality operator
    Offset                          num_items;          ///< Total number of input items


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockSelectSweep(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIterator               d_in,               ///< Input data
        FlagsInputIterator          d_flags,            ///< Input flags
        SelectedOutputIterator      d_selected_out,     ///< Output data
        SelectOp                    select_op,          ///< Selection operator
        EqualityOp                  equality_op,        ///< Equality operator
        Offset                      num_items)          ///< Total number of input items
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_flags(d_flags),
        d_selected_out(d_selected_out),
        select_op(select_op),
        inequality_op(equality_op),
        num_items(num_items)
    {}


    //---------------------------------------------------------------------
    // Utility methods for initializing the selections
    //---------------------------------------------------------------------

    /**
     * Template unrolled selection via selection operator
     */
    template <bool FIRST_TILE, bool LAST_TILE, int ITERATION>
    __device__ __forceinline__ void ApplySelectionOp(
        Offset                      block_offset,
        Offset                      num_remaining,
        T                           (&items)[ITEMS_PER_THREAD],
        Offset                      (&selected)[ITEMS_PER_THREAD],
        Int2Type<ITERATION>         iteration)
    {
        selected[ITERATION] = 0;
        if (!LAST_TILE || (Offset(threadIdx.x * ITEMS_PER_THREAD) + ITERATION < num_remaining))
            selected[ITERATION] = select_op(items[ITERATION]);

        ApplySelectionOp<FIRST_TILE, LAST_TILE>(block_offset, num_remaining, items, selected, Int2Type<ITERATION + 1>());
    }

    /**
     * Template unrolled selection via selection operator
     */
    template <bool FIRST_TILE, bool LAST_TILE>
    __device__ __forceinline__ void ApplySelectionOp(
        Offset                      block_offset,
        Offset                      num_remaining,
        T                           (&items)[ITEMS_PER_THREAD],
        Offset                      (&selected)[ITEMS_PER_THREAD],
        Int2Type<ITEMS_PER_THREAD>  iteration)
    {}

    /**
     * Initialize selections (specialized for selection operator)
     */
    template <bool FIRST_TILE, bool LAST_TILE>
    __device__ __forceinline__ void InitializeSelections(
        Offset                      block_offset,
        Offset                      num_remaining,
        T                           (&items)[ITEMS_PER_THREAD],
        Offset                      (&selected)[ITEMS_PER_THREAD],
        Int2Type<USE_SELECT_OP>     select_method)
    {
        __syncthreads();

        ApplySelectionOp<FIRST_TILE, LAST_TILE>(block_offset, num_remaining, items, selected, Int2Type<0>());
    }


    /**
     * Initialize selections (specialized for valid flags)
     */
    template <bool FIRST_TILE, bool LAST_TILE>
    __device__ __forceinline__ void InitializeSelections(
        Offset                      block_offset,
        Offset                      num_remaining,
        T                           (&items)[ITEMS_PER_THREAD],
        Offset                      (&selected)[ITEMS_PER_THREAD],
        Int2Type<USE_SELECT_FLAGS>  select_method)
    {
        Flag flags[ITEMS_PER_THREAD];

        if (LAST_TILE)
            BlockLoadFlags(temp_storage.load_flags).Load(d_flags + block_offset, flags, num_remaining, 0);
        else
            BlockLoadFlags(temp_storage.load_flags).Load(d_flags + block_offset, flags);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            selected[ITEM] = flags[ITEM];
        }

        if (SYNC_AFTER_LOAD)
            __syncthreads();
    }


    /**
     * Initialize selections (specialized for discontinuity detection)
     */
    template <bool FIRST_TILE, bool LAST_TILE>
    __device__ __forceinline__ void InitializeSelections(
        Offset                      block_offset,
        Offset                      num_remaining,
        T                           (&items)[ITEMS_PER_THREAD],
        Offset                      (&selected)[ITEMS_PER_THREAD],
        Int2Type<USE_DISCONTINUITY> select_method)
    {
        if (FIRST_TILE)
        {
            // First tile always flags the first item
            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeads(selected, items, inequality_op);
        }
        else
        {
            // Subsequent tiles require the last item from the previous tile
            T tile_predecessor_item;
            if (threadIdx.x == 0)
                tile_predecessor_item = d_in[block_offset - 1];

            BlockDiscontinuityT(temp_storage.discontinuity).FlagHeads(selected, items, inequality_op, tile_predecessor_item);
        }
    }


    //---------------------------------------------------------------------
    // Utility methods for scan
    //---------------------------------------------------------------------

    /**
     * Scan of allocations
     */
    __device__ __forceinline__ void ScanAllocations(
        Offset  &tile_aggregate,
        int     &warp_aggregate,
        int     &warp_exclusive,
        int     (&selected)[ITEMS_PER_THREAD],
        int     (&thread_exclusives)[ITEMS_PER_THREAD])
    {
        // Perform warpscans
        int warp_id = ((WARPS == 1) ? 0 : threadIdx.x / WARP_THREADS);
        int lane_id = LaneId();

        int thread_aggregate = ThreadReduce(selected, cub::Sum());
        int inclusive_partial, exclusive_partial;
        WarpScanAllocations(temp_storage.warp_scan[warp_id]).Sum(thread_aggregate, inclusive_partial, exclusive_partial);
        ThreadScanExclusive(selected, thread_exclusives, cub::Sum(), exclusive_partial);

        // Last lane in each warp shares its warp-aggregate
        if (lane_id == WARP_THREADS - 1)
            temp_storage.warp_aggregates[warp_id] = inclusive_partial;

        __syncthreads();

        // Accumulate total selected and the warp-wide prefix
        warp_exclusive   = 0;
        warp_aggregate   = temp_storage.warp_aggregates[warp_id];
        tile_aggregate   = temp_storage.warp_aggregates[0];

        #pragma unroll
        for (int WARP = 1; WARP < WARPS; ++WARP)
        {
            if (warp_id == WARP)
                warp_exclusive = tile_aggregate;

            tile_aggregate += temp_storage.warp_aggregates[WARP];
        }

        // Push unselected items into the local exchange's guard band
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (!selected[ITEM])
                thread_exclusives[ITEM] = WARP_THREADS * ITEMS_PER_THREAD;
        }
    }

    //---------------------------------------------------------------------
    // Utility methods for scattering selections
    //---------------------------------------------------------------------

    /**
     * Two-phase scatter, specialized for warp time-slicing
     */
    __device__ __forceinline__ void ScatterTwoPhase(
        Offset          tile_exclusive,
        int             warp_aggregate,
        int             warp_exclusive,
        int             (&thread_exclusives)[ITEMS_PER_THREAD],
        T               (&items)[ITEMS_PER_THREAD],
        Int2Type<true>  is_warp_time_slice)
    {
        int warp_id = ((WARPS == 1) ? 0 : threadIdx.x / WARP_THREADS);
        int lane_id = LaneId();

        // Locally compact items within the warp (first warp)
        if (warp_id == 0)
        {
            WarpExchangeT(temp_storage.exchange[0]).ScatterToStriped(items, thread_exclusives);
        }

        // Locally compact items within the warp (remaining warps)
        #pragma unroll
        for (int SLICE = 1; SLICE < WARPS; ++SLICE)
        {
            __syncthreads();

            if (warp_id == SLICE)
            {
                WarpExchangeT(temp_storage.exchange[0]).ScatterToStriped(items, thread_exclusives);
            }
        }

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if ((ITEM * WARP_THREADS) < warp_aggregate - lane_id)
            {
                d_selected_out[tile_exclusive + warp_exclusive + (ITEM * WARP_THREADS) + lane_id] = items[ITEM];
            }
        }
    }



    /**
     * Two-phase scatter
     */
    __device__ __forceinline__ void ScatterTwoPhase(
        Offset          tile_exclusive,
        int             warp_aggregate,
        int             warp_exclusive,
        int             (&thread_exclusives)[ITEMS_PER_THREAD],
        T               (&items)[ITEMS_PER_THREAD],
        Int2Type<false> is_warp_time_slice)
    {
        int warp_id = ((WARPS == 1) ? 0 : threadIdx.x / WARP_THREADS);
        int lane_id = LaneId();

        WarpExchangeT(temp_storage.exchange[warp_id]).ScatterToStriped(items, thread_exclusives);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if ((ITEM * WARP_THREADS) < warp_aggregate - lane_id)
            {
                d_selected_out[tile_exclusive + warp_exclusive + (ITEM * WARP_THREADS) + lane_id] = items[ITEM];
            }
        }
    }



    /**
     * Scatter
     */
    __device__ __forceinline__ void Scatter(
        Offset  tile_aggregate,
        Offset  tile_exclusive,
        int     warp_aggregate,
        int     warp_exclusive,
        int     (&thread_exclusives)[ITEMS_PER_THREAD],
        T       (&items)[ITEMS_PER_THREAD])
    {
        if ((ITEMS_PER_THREAD == 1) || (tile_aggregate < BLOCK_THREADS))
        {
            // Direct scatter if the warp has any items
            if (warp_aggregate)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
                {
                    if (thread_exclusives[ITEM] < warp_aggregate)
                        d_selected_out[tile_exclusive + warp_exclusive + thread_exclusives[ITEM]] = items[ITEM];
                }
            }
        }
        else
        {
            ScatterTwoPhase(
                tile_exclusive,
                warp_aggregate,
                warp_exclusive,
                thread_exclusives,
                items,
                Int2Type<STORE_WARP_TIME_SLICING>());
        }
    }





    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------

    /**
     * Process a tile of input (dynamic chained scan)
     */
    template <bool LAST_TILE>
    __device__ __forceinline__ Offset ConsumeTile(
        Offset              num_items,          ///< Total number of input items
        Offset              num_remaining,      ///< Total number of items remaining to be processed (including this tile)
        int                 tile_idx,           ///< Tile index
        Offset              block_offset,       ///< Tile offset
        ScanTileState       &tile_status)       ///< Global list of tile status
    {
        if (tile_idx == 0)
        {
            // First tile

            // Load items
            T items[ITEMS_PER_THREAD];
            if (LAST_TILE)
            {
                T oob_item = (SELECT_METHOD == USE_DISCONTINUITY) ?
                    d_in[num_items - 1] : // Repeat last item
                    ZeroInitialize<T>();

                BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items, num_remaining, oob_item);
            }
            else
            {
                BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items);
            }

            if (SYNC_AFTER_LOAD)
                __syncthreads();

            // Initialize selected/rejected output flags for first tile
            int selected[ITEMS_PER_THREAD];             // Selection flags
            InitializeSelections<true, LAST_TILE>(block_offset, num_remaining, items, selected, Int2Type<SELECT_METHOD>());

            // Scan the selected flags
            Offset tile_aggregate;
            int warp_aggregate, warp_exclusive;
            int thread_exclusives[ITEMS_PER_THREAD];    // Thread exclusive scatter prefixes
            ScanAllocations(tile_aggregate, warp_aggregate, warp_exclusive, selected, thread_exclusives);

            // Update tile status if there may be successor tiles
            if (!LAST_TILE && (threadIdx.x == 0))
                tile_status.SetInclusive(0, tile_aggregate);

            Offset tile_exclusive = 0;

            // Scatter
            Scatter(tile_aggregate, tile_exclusive, warp_aggregate, warp_exclusive, thread_exclusives, items);

            // Return total number of items selected (inclusive of this tile)
            return tile_aggregate;
        }
        else
        {
            // Not first tile

            // Load items
            T items[ITEMS_PER_THREAD];
            if (LAST_TILE)
            {
                T oob_item = (SELECT_METHOD == USE_DISCONTINUITY) ?
                    d_in[num_items - 1] : // Repeat last item
                    ZeroInitialize<T>();

                BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items, num_remaining, oob_item);
            }
            else
            {
                BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items);
            }

            if (SYNC_AFTER_LOAD)
                __syncthreads();

            // Initialize selected/rejected output flags for non-first tile
            int selected[ITEMS_PER_THREAD];              // Selection flags
            InitializeSelections<false, LAST_TILE>(block_offset, num_remaining, items, selected, Int2Type<SELECT_METHOD>());

            // Scan the selected flags
            Offset tile_aggregate;
            int warp_aggregate, warp_exclusive;
            int thread_exclusives[ITEMS_PER_THREAD];       // Scatter offsets
            ScanAllocations(tile_aggregate, warp_aggregate, warp_exclusive, selected, thread_exclusives);

            // First warp computes tile prefix in lane 0
            LookbackPrefixCallbackOp prefix_op(tile_status, temp_storage.prefix, Sum(), tile_idx);
            int warp_id = ((WARPS == 1) ? 0 : threadIdx.x / WARP_THREADS);
            if (warp_id == 0)
            {
                prefix_op(tile_aggregate);
                if (threadIdx.x == 0)
                    temp_storage.tile_exclusive = prefix_op.exclusive_prefix;
            }

            __syncthreads();

            Offset tile_exclusive = temp_storage.tile_exclusive;

            // Scatter
            Scatter(tile_aggregate, tile_exclusive, warp_aggregate, warp_exclusive, thread_exclusives, items);

            // Return total number of items selected (inclusive of this tile)
            return prefix_op.inclusive_prefix;
        }
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic chained scan
     */
    template <typename NumSelectedIterator>         ///< Output iterator type for recording number of items selected
    __device__ __forceinline__ void ConsumeRange(
        int                     num_tiles,          ///< Total number of input tiles
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        ScanTileState           &tile_status,       ///< Global list of tile status
        NumSelectedIterator     d_num_selected_out)     ///< Output total number selected
    {

#if __CUDA_ARCH__ > 130

        // Blocks may not be launched in increasing order, so work-steal tiles
        if (threadIdx.x == 0)
            temp_storage.tile_idx = queue.Drain(1);

        __syncthreads();

        int tile_idx = temp_storage.tile_idx;

#else

        // Blocks are launched in increasing order, so just assign one tile per block
        int tile_idx = (blockIdx.y * gridDim.x) + blockIdx.x;

#endif

        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;            // Global offset for the current tile
        Offset  num_remaining   = num_items - block_offset;                 // Remaining items (including this tile)

        if (num_remaining > 0)
        {
            if (num_remaining > TILE_ITEMS)
            {
                // Full tile
                ConsumeTile<false>(num_items, num_remaining, tile_idx, block_offset, tile_status);
            }
            else
            {
                // Last tile
                Offset total_selected = ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

                // Output the total number of items selected
                if (threadIdx.x == 0)
                {
                    *d_num_selected_out = total_selected;
                }
            }
        }
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

