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
 * cub::BlockRangeSelect implements a stateful abstraction of CUDA thread blocks for participating in device-wide select.
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
 * Parameterizable tuning policy type for BlockRangeSelect
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    bool                        _TWO_PHASE_SCATTER,             ///< Whether or not to coalesce output values in shared memory before scattering them to global
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct BlockRangeSelectPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
        TWO_PHASE_SCATTER       = _TWO_PHASE_SCATTER,           ///< Whether or not to coalesce output values in shared memory before scattering them to global
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;      ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;      ///< The BlockScan algorithm to use
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockRangeSelect implements a stateful abstraction of CUDA thread blocks for participating in device-wide selection across a range of tiles
 *
 * Performs functor-based selection if SelectOp functor type != NullType
 * Otherwise performs flag-based selection if FlagIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 */
template <
    typename    BlockRangeSelectPolicy,         ///< Parameterized BlockRangeSelectPolicy tuning policy type
    typename    InputIterator,                  ///< Random-access input iterator type for selection items
    typename    FlagIterator,                   ///< Random-access input iterator type for selections (NullType* if a selection functor or discontinuity flagging is to be used for selection)
    typename    OutputIterator,                 ///< Random-access input iterator type for selected items
    typename    SelectOp,                       ///< Selection operator type (NullType if selections or discontinuity flagging is to be used for selection)
    typename    EqualityOp,                     ///< Equality operator type (NullType if selection functor or selections is to be used for selection)
    typename    Offset,                         ///< Signed integer type for global offsets
    bool        KEEP_REJECTS>                   ///< Whether or not we push rejected items to the back of the output
struct BlockRangeSelect
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Data type of flag iterator
    typedef typename std::iterator_traits<FlagIterator>::value_type Flag;

    // Tile status descriptor interface type
    typedef ScanTileState<Offset> ScanTileState;

    // Constants
    enum
    {
        USE_SELECT_OP,
        USE_SELECT_FLAGS,
        USE_DISCONTINUITY,

        BLOCK_THREADS       = BlockRangeSelectPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockRangeSelectPolicy::ITEMS_PER_THREAD,
        TWO_PHASE_SCATTER   = (BlockRangeSelectPolicy::TWO_PHASE_SCATTER) && (ITEMS_PER_THREAD > 1),
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,

        // Whether or not to sync after loading data
        SYNC_AFTER_LOAD     = (BlockRangeSelectPolicy::LOAD_ALGORITHM != BLOCK_LOAD_DIRECT),

        SELECT_METHOD       = (!Equals<SelectOp, NullType>::VALUE) ?
                                USE_SELECT_OP :
                                (!Equals<Flag, NullType>::VALUE) ?
                                    USE_SELECT_FLAGS :
                                    USE_DISCONTINUITY
    };

    // Input iterator wrapper type
    typedef typename If<IsPointer<InputIterator>::VALUE,
            CacheModifiedInputIterator<BlockRangeSelectPolicy::LOAD_MODIFIER, T, Offset>,      // Wrap the native input pointer with CacheModifiedInputIterator
            InputIterator>::Type                                                                // Directly use the supplied input iterator type
        WrappedInputIterator;

    // Flag iterator wrapper type
    typedef typename If<IsPointer<FlagIterator>::VALUE,
            CacheModifiedInputIterator<BlockRangeSelectPolicy::LOAD_MODIFIER, Flag, Offset>,   // Wrap the native input pointer with CacheModifiedInputIterator
            FlagIterator>::Type                                                                 // Directly use the supplied input iterator type
        WrappedFlagIterator;

    // Parameterized BlockLoad type for input items
    typedef BlockLoad<
            WrappedInputIterator,
            BlockRangeSelectPolicy::BLOCK_THREADS,
            BlockRangeSelectPolicy::ITEMS_PER_THREAD,
            BlockRangeSelectPolicy::LOAD_ALGORITHM>
        BlockLoadT;

    // Parameterized BlockLoad type for flags
    typedef BlockLoad<
            WrappedFlagIterator,
            BlockRangeSelectPolicy::BLOCK_THREADS,
            BlockRangeSelectPolicy::ITEMS_PER_THREAD,
            BlockRangeSelectPolicy::LOAD_ALGORITHM>
        BlockLoadFlags;

    // Parameterized BlockExchange type for input items
    typedef BlockExchange<
            T,
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
        BlockExchangeT;

    // Parameterized BlockDiscontinuity type for input items
    typedef BlockDiscontinuity<T, BLOCK_THREADS> BlockDiscontinuityT;

    // Parameterized BlockScan type
    typedef BlockScan<
            Offset,
            BlockRangeSelectPolicy::BLOCK_THREADS,
            BlockRangeSelectPolicy::SCAN_ALGORITHM>
        BlockScanAllocations;

    // Callback type for obtaining tile prefix during block scan
    typedef BlockScanLookbackPrefixOp<
            Offset,
            Sum,
            ScanTileState>
        LookbackPrefixCallbackOp;

    // Shared memory type for this threadblock
    struct _TempStorage
    {
        union
        {
            struct
            {
                typename LookbackPrefixCallbackOp::TempStorage  prefix;         // Smem needed for cooperative prefix callback
                typename BlockScanAllocations::TempStorage      scan;           // Smem needed for tile scanning
                typename BlockDiscontinuityT::TempStorage       discontinuity;  // Smem needed for discontinuity detection
            };

            // Smem needed for input loading
            typename BlockLoadT::TempStorage load_items;

            // Smem needed for flag loading
            typename BlockLoadFlags::TempStorage load_flags;

            // Smem needed for two-phase scatter
            typename If<TWO_PHASE_SCATTER, typename BlockExchangeT::TempStorage, NullType>::Type exchange;
        };

        Offset      tile_idx;                   // Shared tile index
        Offset      tile_num_selected_prefix;   // Exclusive tile prefix
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage                    &temp_storage;      ///< Reference to temp_storage
    WrappedInputIterator            d_in;               ///< Input data
    WrappedFlagIterator             d_flags;            ///< Input flags
    OutputIterator                  d_out;              ///< Output data
    SelectOp                        select_op;          ///< Selection operator
    InequalityWrapper<EqualityOp>   inequality_op;      ///< Inequality operator
    Offset                          num_items;          ///< Total number of input items


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockRangeSelect(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        InputIterator               d_in,               ///< Input data
        FlagIterator                d_flags,            ///< Input flags
        OutputIterator              d_out,              ///< Output data
        SelectOp                    select_op,          ///< Selection operator
        EqualityOp                  equality_op,        ///< Equality operator
        Offset                      num_items)          ///< Total number of input items
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_flags(d_flags),
        d_out(d_out),
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
    // Utility methods for scattering selections
    //---------------------------------------------------------------------

    /**
     * Scatter data items to select offsets (specialized for direct scattering and for discarding rejected items)
     */
    template <bool LAST_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          block_offset,
        T               (&items)[ITEMS_PER_THREAD],
        Offset          selected[ITEMS_PER_THREAD],
        Offset          scatter_offsets[ITEMS_PER_THREAD],
        Offset          tile_num_selected_prefix,
        Offset          tile_num_selected,
        Offset          num_remaining,
        Int2Type<false> keep_rejects,
        Int2Type<false> two_phase_scatter)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (selected[ITEM])
            {
                // Selected items are placed front-to-back
                d_out[scatter_offsets[ITEM]] = items[ITEM];
            }
        }
    }


    /**
     * Scatter data items to select offsets (specialized for direct scattering and for partitioning rejected items after selected items)
     */
    template <bool LAST_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          block_offset,
        T               (&items)[ITEMS_PER_THREAD],
        Offset          selected[ITEMS_PER_THREAD],
        Offset          scatter_offsets[ITEMS_PER_THREAD],
        Offset          tile_num_selected_prefix,
        Offset          tile_num_selected,
        Offset          num_remaining,
        Int2Type<true>  keep_rejects,
        Int2Type<false> two_phase_scatter)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (selected[ITEM])
            {
                // Selected items are placed front-to-back
                d_out[scatter_offsets[ITEM]] = items[ITEM];
            }
            else if (!LAST_TILE || (Offset(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_remaining))
            {
                Offset global_idx = block_offset + (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
                Offset reject_idx = global_idx - scatter_offsets[ITEM];

                // Rejected items are placed back-to-front
                d_out[num_items - reject_idx - 1] = items[ITEM];
            }
        }
    }


    /**
     * Scatter data items to select offsets (specialized for two-phase scattering and for discarding rejected items)
     */
    template <bool LAST_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          block_offset,
        T               (&items)[ITEMS_PER_THREAD],
        Offset          selected[ITEMS_PER_THREAD],
        Offset          scatter_offsets[ITEMS_PER_THREAD],
        Offset          tile_num_selected_prefix,
        Offset          tile_num_selected,
        Offset          num_remaining,
        Int2Type<false> keep_rejects,
        Int2Type<true>  two_phase_scatter)
    {
        if ((tile_num_selected >> Log2<BLOCK_THREADS>::VALUE) == 0)
        {
            // Average number of selected items per thread is less than one, so just do a one-phase scatter
            Scatter<LAST_TILE>(
                block_offset,
                items,
                selected,
                scatter_offsets,
                tile_num_selected_prefix,
                tile_num_selected,
                num_remaining,
                keep_rejects,
                Int2Type<false>());
        }
        else
        {
            // Share exclusive tile prefix
            if (threadIdx.x == 0)
            {
                temp_storage.tile_num_selected_prefix = tile_num_selected_prefix;
            }

            __syncthreads();

            // Load exclusive tile prefix in all threads
            tile_num_selected_prefix = temp_storage.tile_num_selected_prefix;

            int local_ranks[ITEMS_PER_THREAD];

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                local_ranks[ITEM] = scatter_offsets[ITEM] - tile_num_selected_prefix;
            }

            BlockExchangeT(temp_storage.exchange).ScatterToStriped(items, local_ranks, selected);

            // Selected items are placed front-to-back
            StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + tile_num_selected_prefix, items, tile_num_selected);
        }
    }


    /**
     * Scatter data items to select offsets (specialized for two-phase scattering and for partitioning rejected items after selected items)
     */
    template <bool LAST_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          block_offset,
        T               (&items)[ITEMS_PER_THREAD],
        Offset          selected[ITEMS_PER_THREAD],
        Offset          scatter_offsets[ITEMS_PER_THREAD],
        Offset          tile_num_selected_prefix,
        Offset          tile_num_selected,
        Offset          num_remaining,
        Int2Type<true>  keep_rejects,
        Int2Type<true>  two_phase_scatter)
    {
        // Share exclusive tile prefix
        if (threadIdx.x == 0)
        {
            temp_storage.tile_num_selected_prefix = tile_num_selected_prefix;
        }

        __syncthreads();

        // Load the exclusive tile prefix in all threads
        tile_num_selected_prefix = temp_storage.tile_num_selected_prefix;

        // Determine the exclusive prefix for rejects
        Offset tile_rejected_exclusive_prefix = block_offset - tile_num_selected_prefix;

        // Determine local scatter offsets
        int local_ranks[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            local_ranks[ITEM]   = -1;
            Offset global_idx   = block_offset + (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
            Offset reject_idx   = global_idx - scatter_offsets[ITEM];

            if (selected[ITEM])
            {
                // Selected items
                local_ranks[ITEM] = scatter_offsets[ITEM] - tile_num_selected_prefix;
            }
            else if (!LAST_TILE || (Offset(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_remaining))
            {
                // Rejected items
                local_ranks[ITEM] = (reject_idx - tile_rejected_exclusive_prefix) + tile_num_selected;
            }
        }

        // Coalesce selected and rejected items in shared memory, gathering in striped arrangements
        if (LAST_TILE)
            BlockExchangeT(temp_storage.exchange).ScatterToStripedGuarded(items, local_ranks);
        else
            BlockExchangeT(temp_storage.exchange).ScatterToStriped(items, local_ranks);

        // Store in striped order
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            Offset local_idx = (ITEM * BLOCK_THREADS) + threadIdx.x;
            Offset scatter_offset = tile_num_selected_prefix + local_idx;
            if (local_idx >= tile_num_selected)
                scatter_offset = num_items - (tile_rejected_exclusive_prefix + (local_idx - tile_num_selected)) - 1;

            if (!LAST_TILE || (local_idx < num_remaining))
            {
                d_out[scatter_offset] = items[ITEM];
            }
        }
    }


    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------

    /**
     * Process a tile of input (dynamic domino scan)
     */
    template <bool LAST_TILE>
    __device__ __forceinline__ Offset ConsumeTile(
        Offset              num_items,          ///< Total number of input items
        Offset              num_remaining,      ///< Total number of items remaining to be processed (including this tile)
        int                 tile_idx,           ///< Tile index
        Offset              block_offset,       ///< Tile offset
        ScanTileState  &tile_status)       ///< Global list of tile status
    {
        T items[ITEMS_PER_THREAD];
        Offset selected[ITEMS_PER_THREAD];              // Selection flags
        Offset scatter_offsets[ITEMS_PER_THREAD];       // Scatter offsets
        Offset tile_num_selected_prefix;                // Total number of selected items prior to this tile
        Offset tile_num_selected;                       // Total number of selected items within this tile
        Offset num_selected;                            //

        // Load items
        if (LAST_TILE)
            BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items, num_remaining, d_in[num_items - 1]);     // Repeat last item
        else
            BlockLoadT(temp_storage.load_items).Load(d_in + block_offset, items);

        if (SYNC_AFTER_LOAD)
            __syncthreads();

        if (tile_idx == 0)
        {
            // Initialize selected/rejected output flags for first tile
            InitializeSelections<true, LAST_TILE>(
                block_offset,
                num_remaining,
                items,
                selected,
                Int2Type<SELECT_METHOD>());

            // Compute scatter offsets by scanning the flags
            BlockScanAllocations(temp_storage.scan).ExclusiveSum(selected, scatter_offsets, tile_num_selected);

            // Update tile status if there may be successor tiles
            if (!LAST_TILE && (threadIdx.x == 0))
                tile_status.SetInclusive(0, tile_num_selected);

            tile_num_selected_prefix = 0;
            num_selected = tile_num_selected;
        }
        else
        {
            // Initialize selected/rejected output flags for non-first tile
            InitializeSelections<false, LAST_TILE>(
                block_offset,
                num_remaining,
                items,
                selected,
                Int2Type<SELECT_METHOD>());

            // Compute scatter offsets by scanning the flags
            LookbackPrefixCallbackOp prefix_op(tile_status, temp_storage.prefix, Sum(), tile_idx);
            BlockScanAllocations(temp_storage.scan).ExclusiveSum(selected, scatter_offsets, tile_num_selected, prefix_op);

            tile_num_selected_prefix = prefix_op.exclusive_prefix;
            num_selected = prefix_op.inclusive_prefix;
        }

        // Store selected items
        Scatter<LAST_TILE>(
            block_offset,
            items,
            selected,
            scatter_offsets,
            tile_num_selected_prefix,
            tile_num_selected,
            num_remaining,
            Int2Type<KEEP_REJECTS>(),
            Int2Type<TWO_PHASE_SCATTER>());

        // Return total number of items selected (inclusive of this tile)
        return num_selected;
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic domino scan
     */
    template <typename NumSelectedIterator>         ///< Output iterator type for recording number of items selected
    __device__ __forceinline__ void ConsumeRange(
        int                     num_tiles,          ///< Total number of input tiles
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        ScanTileState      &tile_status,       ///< Global list of tile status
        NumSelectedIterator     d_num_selected)     ///< Output total number selected
    {
#if (CUB_PTX_ARCH <= 130)
        // Blocks are launched in increasing order, so just assign one tile per block

        int     tile_idx        = (blockIdx.y * 32 * 1024) + blockIdx.x;    // Current tile index
        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;            // Global offset for the current tile
        Offset  num_remaining   = num_items - block_offset;                 // Remaining items (including this tile)

        if (num_remaining > TILE_ITEMS)
        {
            ConsumeTile<false>(num_items, num_remaining, tile_idx, block_offset, tile_status);
        }
        else if (num_remaining > 0)
        {
            Offset total_selected = ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            // Output the total number of items selected
            if (threadIdx.x == 0)
            {
                *d_num_selected = total_selected;
            }
        }

#else
        // Blocks may not be launched in increasing order, so work-steal tiles

        // Get first tile index
        if (threadIdx.x == 0)
            temp_storage.tile_idx = queue.Drain(1);

        __syncthreads();

        int     tile_idx        = temp_storage.tile_idx;
        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;
        Offset  num_remaining   = num_items - block_offset;

        while (num_remaining > TILE_ITEMS)
        {
            // Consume full tile
            ConsumeTile<false>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            // Get next tile
            if (threadIdx.x == 0)
                temp_storage.tile_idx = queue.Drain(1);

            __syncthreads();

            tile_idx        = temp_storage.tile_idx;
            block_offset    = Offset(TILE_ITEMS) * tile_idx;
            num_remaining   = num_items - block_offset;
        }

        // Consume the last (and potentially partially-full) tile
        if (num_remaining > 0)
        {
            Offset total_selected = ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            // Output the total number of items selected
            if (threadIdx.x == 0)
            {
                *d_num_selected = total_selected;
            }
        }

#endif

    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

