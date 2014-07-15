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
 * cub::BlockReduceSweepByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key.
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
#include "../iterator/constant_input_iterator.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for BlockReduceSweepByKey
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    bool                        _TWO_PHASE_SCATTER,             ///< Whether or not to coalesce output values in shared memory before scattering them to global
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct BlockReduceSweepByKeyPolicy
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
 * \brief BlockReduceSweepByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key across a range of tiles
 */
template <
    typename    BlockReduceSweepByKeyPolicy,    ///< Parameterized BlockReduceSweepByKeyPolicy tuning policy type
    typename    KeysInputIterator,               ///< Random-access input iterator type for keys
    typename    UniqueOutputIterator,              ///< Random-access output iterator type for keys
    typename    ValuesInputIterator,             ///< Random-access input iterator type for values
    typename    AggregatesOutputIterator,            ///< Random-access output iterator type for values
    typename    EqualityOp,                     ///< Key equality operator type
    typename    ReductionOp,                    ///< Value reduction operator type
    typename    Offset>                         ///< Signed integer type for global offsets
struct BlockReduceSweepByKey
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of key iterator
    typedef typename std::iterator_traits<KeysInputIterator>::value_type Key;

    // Data type of value iterator
    typedef typename std::iterator_traits<ValuesInputIterator>::value_type Value;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef ItemOffsetPair<Value, Offset> ReductionOffsetPair;

    // Tile status descriptor interface type
    typedef ReduceByKeyScanTileState<Value, Offset> ScanTileState;

    // Constants
    enum
    {
        BLOCK_THREADS       = BlockReduceSweepByKeyPolicy::BLOCK_THREADS,
        WARPS               = BLOCK_THREADS / CUB_PTX_WARP_THREADS,
        ITEMS_PER_THREAD    = BlockReduceSweepByKeyPolicy::ITEMS_PER_THREAD,
        TWO_PHASE_SCATTER   = (BlockReduceSweepByKeyPolicy::TWO_PHASE_SCATTER) && (ITEMS_PER_THREAD > 1),
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,

        // Whether or not the scan operation has a zero-valued identity value (true if we're performing addition on a primitive type)
        HAS_IDENTITY_ZERO       = (Equals<ReductionOp, cub::Sum>::VALUE) && (Traits<Value>::PRIMITIVE),

        // Whether or not to sync after loading data
        SYNC_AFTER_LOAD         = (BlockReduceSweepByKeyPolicy::LOAD_ALGORITHM != BLOCK_LOAD_DIRECT),

        // Whether or not this is run-length-encoding with a constant iterator as values
        IS_RUN_LENGTH_ENCODE    = (Equals<ValuesInputIterator, ConstantInputIterator<Value, size_t> >::VALUE) || (Equals<ValuesInputIterator, ConstantInputIterator<Value, int> >::VALUE) || (Equals<ValuesInputIterator, ConstantInputIterator<Value, unsigned int> >::VALUE),

    };

    // Cache-modified input iterator wrapper type for keys
    typedef typename If<IsPointer<KeysInputIterator>::VALUE,
            CacheModifiedInputIterator<BlockReduceSweepByKeyPolicy::LOAD_MODIFIER, Key, Offset>,   // Wrap the native input pointer with CacheModifiedValuesInputIterator
            KeysInputIterator>::Type                                                                 // Directly use the supplied input iterator type
        WrappedKeysInputIterator;

    // Cache-modified input iterator wrapper type for values
    typedef typename If<IsPointer<ValuesInputIterator>::VALUE,
            CacheModifiedInputIterator<BlockReduceSweepByKeyPolicy::LOAD_MODIFIER, Value, Offset>,  // Wrap the native input pointer with CacheModifiedValuesInputIterator
            ValuesInputIterator>::Type                                                                // Directly use the supplied input iterator type
        WrappedValuesInputIterator;

    // Reduce-value-by-segment scan operator
    typedef ReduceBySegmentOp<ReductionOp, ReductionOffsetPair> ReduceBySegmentOp;

    // Parameterized BlockLoad type for keys
    typedef BlockLoad<
            WrappedKeysInputIterator,
            BlockReduceSweepByKeyPolicy::BLOCK_THREADS,
            BlockReduceSweepByKeyPolicy::ITEMS_PER_THREAD,
            BlockReduceSweepByKeyPolicy::LOAD_ALGORITHM>
        BlockLoadKeys;

    // Parameterized BlockLoad type for values
    typedef BlockLoad<
            WrappedValuesInputIterator,
            BlockReduceSweepByKeyPolicy::BLOCK_THREADS,
            BlockReduceSweepByKeyPolicy::ITEMS_PER_THREAD,
            (IS_RUN_LENGTH_ENCODE) ?
                BLOCK_LOAD_DIRECT :
                (BlockLoadAlgorithm) BlockReduceSweepByKeyPolicy::LOAD_ALGORITHM>
        BlockLoadValues;

    // Parameterized BlockExchange type for locally compacting items as part of a two-phase scatter
    typedef BlockExchange<
            Key,
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
        BlockExchangeKeys;

    // Parameterized BlockExchange type for locally compacting items as part of a two-phase scatter
    typedef BlockExchange<
            Value,
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
        BlockExchangeValues;

    // Parameterized BlockDiscontinuity type for keys
    typedef BlockDiscontinuity<Key, BLOCK_THREADS> BlockDiscontinuityKeys;

    // Parameterized BlockScan type
    typedef BlockScan<
            ReductionOffsetPair,
            BlockReduceSweepByKeyPolicy::BLOCK_THREADS,
            BlockReduceSweepByKeyPolicy::SCAN_ALGORITHM>
        BlockScanAllocations;

    // Callback type for obtaining tile prefix during block scan
    typedef BlockScanLookbackPrefixOp<
            ReductionOffsetPair,
            ReduceBySegmentOp,
            ScanTileState>
        LookbackPrefixCallbackOp;

    // Shared memory type for this threadblock
    struct _TempStorage
    {

        union
        {
            struct
            {
                typename BlockScanAllocations::TempStorage      scan;           // Smem needed for tile scanning
                typename LookbackPrefixCallbackOp::TempStorage  prefix;         // Smem needed for cooperative prefix callback
                typename BlockDiscontinuityKeys::TempStorage    discontinuity;  // Smem needed for discontinuity detection
                typename BlockLoadKeys::TempStorage             load_keys;      // Smem needed for loading keys

                Offset      tile_idx;               // Shared tile index
                Offset      tile_num_flags_prefix;  // Exclusive tile prefix
            };

            // Smem needed for loading values
            typename BlockLoadValues::TempStorage load_values;

            // Smem needed for compacting values
            typename BlockExchangeValues::TempStorage exchange_values;

            // Smem needed for compacting keys
            typename BlockExchangeKeys::TempStorage exchange_keys;
        };

    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage                    &temp_storage;      ///< Reference to temp_storage

    WrappedKeysInputIterator        d_keys_in;          ///< Input keys
    UniqueOutputIterator            d_unique_out;       ///< Unique output keys

    WrappedValuesInputIterator      d_values_in;        ///< Input values
    AggregatesOutputIterator        d_aggregates_out;   ///< Output value aggregates

    InequalityWrapper<EqualityOp>   inequality_op;      ///< Key inequality operator
    ReduceBySegmentOp               scan_op;            ///< Reduce-value-by-flag scan operator
    Offset                          num_items;          ///< Total number of input items


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockReduceSweepByKey(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        KeysInputIterator           d_keys_in,          ///< Input keys
        UniqueOutputIterator        d_unique_out,       ///< Unique output keys
        ValuesInputIterator         d_values_in,        ///< Input values
        AggregatesOutputIterator    d_aggregates_out,   ///< Output value aggregates
        EqualityOp                  equality_op,        ///< Key equality operator
        ReductionOp                 reduction_op,       ///< Value reduction operator
        Offset                      num_items)          ///< Total number of input items
    :
        temp_storage(temp_storage.Alias()),
        d_keys_in(d_keys_in),
        d_unique_out(d_unique_out),
        d_values_in(d_values_in),
        d_aggregates_out(d_aggregates_out),
        inequality_op(equality_op),
        scan_op(reduction_op),
        num_items(num_items)
    {}


    //---------------------------------------------------------------------
    // Block scan utility methods
    //---------------------------------------------------------------------

    /**
     * Scan with identity (first tile)
     */
    __device__ __forceinline__
    void ScanBlock(
        ReductionOffsetPair     (&values_and_segments)[ITEMS_PER_THREAD],
        ReductionOffsetPair     &block_aggregate,
        Int2Type<true>      has_identity)
    {
        ReductionOffsetPair identity;
        identity.value = 0;
        identity.offset = 0;
        BlockScanAllocations(temp_storage.scan).ExclusiveScan(values_and_segments, values_and_segments, identity, scan_op, block_aggregate);
    }

    /**
     * Scan without identity (first tile).  Without an identity, the first output item is undefined.
     *
     */
    __device__ __forceinline__
    void ScanBlock(
        ReductionOffsetPair     (&values_and_segments)[ITEMS_PER_THREAD],
        ReductionOffsetPair     &block_aggregate,
        Int2Type<false>     has_identity)
    {
        BlockScanAllocations(temp_storage.scan).ExclusiveScan(values_and_segments, values_and_segments, scan_op, block_aggregate);
    }

    /**
     * Scan with identity (subsequent tile)
     */
    __device__ __forceinline__
    void ScanBlock(
        ReductionOffsetPair             (&values_and_segments)[ITEMS_PER_THREAD],
        ReductionOffsetPair             &block_aggregate,
        LookbackPrefixCallbackOp    &prefix_op,
        Int2Type<true>              has_identity)
    {
        ReductionOffsetPair identity;
        identity.value = 0;
        identity.offset = 0;
        BlockScanAllocations(temp_storage.scan).ExclusiveScan(values_and_segments, values_and_segments, identity, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Scan without identity (subsequent tile).  Without an identity, the first output item is undefined.
     */
    __device__ __forceinline__
    void ScanBlock(
        ReductionOffsetPair             (&values_and_segments)[ITEMS_PER_THREAD],
        ReductionOffsetPair             &block_aggregate,
        LookbackPrefixCallbackOp    &prefix_op,
        Int2Type<false>             has_identity)
    {
        BlockScanAllocations(temp_storage.scan).ExclusiveScan(values_and_segments, values_and_segments, scan_op, block_aggregate, prefix_op);
    }


    //---------------------------------------------------------------------
    // Zip utility methods
    //---------------------------------------------------------------------

    template <bool LAST_TILE>
    __device__ __forceinline__ void ZipValuesAndFlags(
        Offset          num_remaining,
        Value           (&values)[ITEMS_PER_THREAD],
        Offset          (&flags)[ITEMS_PER_THREAD],
        ReductionOffsetPair (&values_and_segments)[ITEMS_PER_THREAD])
    {
        // Zip values and flags
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Unset flags for out-of-bounds keys
            if ((LAST_TILE) && (Offset(threadIdx.x * ITEMS_PER_THREAD) + ITEM >= num_remaining))
                flags[ITEM] = 0;

            values_and_segments[ITEM].value      = values[ITEM];
            values_and_segments[ITEM].offset     = flags[ITEM];
        }
    }

    //---------------------------------------------------------------------
    // Scatter utility methods
    //---------------------------------------------------------------------



    /**
     * Scatter flagged items to output offsets (specialized for direct scattering)
     *
     * The exclusive scan causes each head flag to be paired with the previous
     * value aggregate. As such:
     * - The scatter offsets must be decremented for value value aggregates
     * - The first tile does not scatter the first flagged value (it is undefined from the exclusive scan)
     * - If the tile is partially-full, we need to scatter the first out-of-bounds value (which aggregates all valid values in the last segment)
     *
     */
    template <bool LAST_TILE, bool FIRST_TILE, int ITEM>
    __device__ __forceinline__ void ScatterDirect(
        Offset              num_remaining,
        Key                 (&keys)[ITEMS_PER_THREAD],
        ReductionOffsetPair     (&values_and_segments)[ITEMS_PER_THREAD],
        Offset              (&flags)[ITEMS_PER_THREAD],
        Offset              tile_num_flags,
        Int2Type<ITEM>      iteration)
    {
        // Scatter key
        if (flags[ITEM])
        {
            d_unique_out[values_and_segments[ITEM].offset] = keys[ITEM];
        }

        bool is_first_flag     = FIRST_TILE && (ITEM == 0) && (threadIdx.x == 0);
        bool is_oob_value      = (LAST_TILE) && (Offset(threadIdx.x * ITEMS_PER_THREAD) + ITEM == num_remaining);

        // Scatter value reduction
        if (((flags[ITEM] || is_oob_value)) && (!is_first_flag))
        {
            d_aggregates_out[values_and_segments[ITEM].offset - 1] = values_and_segments[ITEM].value;
        }

        ScatterDirect<LAST_TILE, FIRST_TILE>(num_remaining, keys, values_and_segments, flags, tile_num_flags, Int2Type<ITEM + 1>());
    }

    template <bool LAST_TILE, bool FIRST_TILE>
    __device__ __forceinline__ void ScatterDirect(
        Offset                      num_remaining,
        Key                         (&keys)[ITEMS_PER_THREAD],
        ReductionOffsetPair             (&values_and_segments)[ITEMS_PER_THREAD],
        Offset                      (&flags)[ITEMS_PER_THREAD],
        Offset                      tile_num_flags,
        Int2Type<ITEMS_PER_THREAD>  iteration)
    {}

    /**
     * Scatter flagged items to output offsets (specialized for two-phase scattering)
     *
     * The exclusive scan causes each head flag to be paired with the previous
     * value aggregate. As such:
     * - The scatter offsets must be decremented for value value aggregates
     * - The first tile does not scatter the first flagged value (it is undefined from the exclusive scan)
     * - If the tile is partially-full, we need to scatter the first out-of-bounds value (which aggregates all valid values in the last segment)
     *
     */
    template <bool LAST_TILE, bool FIRST_TILE>
    __device__ __forceinline__ void ScatterTwoPhase(
        Offset          num_remaining,
        Key             (&keys)[ITEMS_PER_THREAD],
        ReductionOffsetPair (&values_and_segments)[ITEMS_PER_THREAD],
        Offset          (&flags)[ITEMS_PER_THREAD],
        Offset          tile_num_flags,
        Offset          tile_num_flags_prefix)
    {
        int     local_ranks[ITEMS_PER_THREAD];
        Value   values[ITEMS_PER_THREAD];

        // Share exclusive tile prefix
        if (threadIdx.x == 0)
        {
            temp_storage.tile_num_flags_prefix = tile_num_flags_prefix;
        }

        __syncthreads();

        // Load exclusive tile prefix in all threads
        tile_num_flags_prefix = temp_storage.tile_num_flags_prefix;

        __syncthreads();

        // Compute local scatter ranks
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            local_ranks[ITEM] = values_and_segments[ITEM].offset - tile_num_flags_prefix;
        }

        // Compact keys in shared memory
        BlockExchangeKeys(temp_storage.exchange_keys).ScatterToStriped(keys, local_ranks, flags);

        // Scatter keys
        StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_unique_out + tile_num_flags_prefix, keys, tile_num_flags);

        // Unzip values and set flag for first oob item in last tile
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            values[ITEM] = values_and_segments[ITEM].value;

            if (FIRST_TILE)
                local_ranks[ITEM]--;

            if (LAST_TILE && (Offset(threadIdx.x * ITEMS_PER_THREAD) + ITEM == num_remaining))
                flags[ITEM] = 1;
        }

        // Unset first flag in first tile
        if (FIRST_TILE && (threadIdx.x == 0))
            flags[0] = 0;

        __syncthreads();

        // Compact values in shared memory
        BlockExchangeValues(temp_storage.exchange_values).ScatterToStriped(values, local_ranks, flags);

        // Number to output
        Offset exchange_count = tile_num_flags;

        if (LAST_TILE && (num_remaining < TILE_ITEMS))
            exchange_count++;

        if (FIRST_TILE)
        {
            exchange_count--;
        }
        else
        {
            tile_num_flags_prefix--;
        }

        // Scatter values
        StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_aggregates_out + tile_num_flags_prefix, values, exchange_count);

        __syncthreads();
    }


    /**
     * Scatter flagged items
     */
    template <bool LAST_TILE, bool FIRST_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          num_remaining,
        Key             (&keys)[ITEMS_PER_THREAD],
        ReductionOffsetPair (&values_and_segments)[ITEMS_PER_THREAD],
        Offset          (&flags)[ITEMS_PER_THREAD],
        Offset          tile_num_flags,
        Offset          tile_num_flags_prefix)
    {
        // Do a one-phase scatter if (a) two-phase is disabled or (b) the average number of selected items per thread is less than one
        if (TWO_PHASE_SCATTER && (tile_num_flags > BLOCK_THREADS))
        {
            ScatterTwoPhase<LAST_TILE, FIRST_TILE>(
                num_remaining,
                keys,
                values_and_segments,
                flags,
                tile_num_flags,
                tile_num_flags_prefix);
        }
        else
        {
            ScatterDirect<LAST_TILE, FIRST_TILE>(
                num_remaining,
                keys,
                values_and_segments,
                flags,
                tile_num_flags,
                Int2Type<0>());
        }
    }


    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------

    /**
     * Process a tile of input (dynamic chained scan)
     */
    template <
        bool                LAST_TILE>
    __device__ __forceinline__ ReductionOffsetPair ConsumeTile(
        Offset              num_items,          ///< Total number of global input items
        Offset              num_remaining,      ///< Number of global input items remaining (including this tile)
        int                 tile_idx,           ///< Tile index
        Offset              block_offset,       ///< Tile offset
        ScanTileState       &tile_status)       ///< Global list of tile status
    {
        Key                 keys[ITEMS_PER_THREAD];                         // Tile keys
        Value               values[ITEMS_PER_THREAD];                       // Tile values
        Offset              flags[ITEMS_PER_THREAD];                        // Segment head flags
        ReductionOffsetPair values_and_segments[ITEMS_PER_THREAD];          // Zipped values and segment flags|indices
        ReductionOffsetPair running_total;                                  // Running count of segments and current value aggregate (including this tile)

        // Load keys
        if (LAST_TILE)
            BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + block_offset, keys, num_remaining);
        else
            BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + block_offset, keys);

        if (tile_idx == 0)
        {
            // First tile
            __syncthreads();

            // Load values
            if (LAST_TILE)
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values, num_remaining);
            else
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values);

            __syncthreads();

            // Set head flags.  First tile sets the first flag for the first item
            BlockDiscontinuityKeys(temp_storage.discontinuity).FlagHeads(flags, keys, inequality_op);

            // Zip values and flags
            ZipValuesAndFlags<LAST_TILE>(num_remaining, values, flags, values_and_segments);

            // Exclusive scan of values and flags
            ReductionOffsetPair block_aggregate;
            ScanBlock(values_and_segments, block_aggregate, Int2Type<HAS_IDENTITY_ZERO>());

            // Update tile status if this is not the last tile
            if (!LAST_TILE && (threadIdx.x == 0))
                tile_status.SetInclusive(0, block_aggregate);

            // Set offset for first scan output
            if (!HAS_IDENTITY_ZERO && (threadIdx.x == 0))
                values_and_segments[0].offset = 0;

            running_total = block_aggregate;

            // Scatter flagged items
            Scatter<LAST_TILE, true>(num_remaining, keys, values_and_segments, flags, block_aggregate.offset, 0);
        }
        else
        {
            // Not first tile

            Key tile_predecessor_key = (threadIdx.x == 0) ?
                d_keys_in[block_offset - 1] :
                ZeroInitialize<Key>();

            __syncthreads();

            // Load values
            if (LAST_TILE)
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values, num_remaining);
            else
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values);

            __syncthreads();

            // Set head flags
            BlockDiscontinuityKeys(temp_storage.discontinuity).FlagHeads(flags, keys, inequality_op, tile_predecessor_key);

            // Zip values and flags
            ZipValuesAndFlags<LAST_TILE>(num_remaining, values, flags, values_and_segments);

            // Exclusive scan of values and flags
            ReductionOffsetPair block_aggregate;
            LookbackPrefixCallbackOp prefix_op(tile_status, temp_storage.prefix, scan_op, tile_idx);

            ScanBlock(values_and_segments, block_aggregate, prefix_op, Int2Type<HAS_IDENTITY_ZERO>());
            running_total = prefix_op.inclusive_prefix;

            // Scatter flagged items
            Scatter<LAST_TILE, false>(num_remaining, keys, values_and_segments, flags, block_aggregate.offset, prefix_op.exclusive_prefix.offset);
        }

        return running_total;
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic chained scan
     */
    template <typename NumRunsIterator>         ///< Output iterator type for recording number of items selected
    __device__ __forceinline__ void ConsumeRange(
        int                     num_tiles,          ///< Total number of input tiles
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        ScanTileState           &tile_status,       ///< Global list of tile status
        NumRunsIterator     d_num_runs_out)     ///< Output pointer for total number of segments identified
    {
#if (CUB_PTX_ARCH <= 130)
        // Blocks are launched in increasing order, so just assign one tile per block

        int     tile_idx        = (blockIdx.y * 32 * 1024) + blockIdx.x;    // Current tile index
        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;            // Global offset for the current tile
        Offset  num_remaining   = num_items - block_offset;                 // Remaining items (including this tile)

        if (num_remaining > TILE_ITEMS)
        {
            // Full tile
            ConsumeTile<false>(num_items, num_remaining, tile_idx, block_offset, tile_status);
        }
        else if (num_remaining > 0)
        {
            // Last tile
            ReductionOffsetPair running_total = ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            // Output the total number of items selected
            if (threadIdx.x == 0)
            {
                *d_num_runs_out = running_total.offset;

                // If the last tile is a whole tile, the inclusive prefix contains accumulated value reduction for the last segment
                if (num_remaining == TILE_ITEMS)
                {
                    d_aggregates_out[running_total.offset - 1] = running_total.value;
                }
            }
        }
#else
        // Blocks may not be launched in increasing order, so work-steal tiles

        // Get first tile index
        if (threadIdx.x == 0)
            temp_storage.tile_idx = queue.Drain(1);

        __syncthreads();

        int     tile_idx        = temp_storage.tile_idx;
        Offset  block_offset    = Offset(TILE_ITEMS) * tile_idx;    // Global offset for the current tile
        Offset  num_remaining   = num_items - block_offset;         // Remaining items (including this tile)

        while (num_remaining > TILE_ITEMS)
        {
            // Consume full tile
            ConsumeTile<false>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            // Get tile index
            if (threadIdx.x == 0)
                temp_storage.tile_idx = queue.Drain(1);

            __syncthreads();

            tile_idx        = temp_storage.tile_idx;
            block_offset    = Offset(TILE_ITEMS) * tile_idx;
            num_remaining   = num_items - block_offset;
        }

        if (num_remaining > 0)
        {
            // Consume last tile (treat as partially-full)
            ReductionOffsetPair running_total = ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            if ((threadIdx.x == 0))
            {
                // Output the total number of items selected
                *d_num_runs_out = running_total.offset;

                // If the last tile is a whole tile, the inclusive prefix contains accumulated value reduction for the last segment
                if (num_remaining == TILE_ITEMS)
                {
                    d_aggregates_out[running_total.offset - 1] = running_total.value;
                }
            }
        }
#endif
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

