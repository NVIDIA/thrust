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
 * cub::BlockRangeReduceByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key.
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
 * Parameterizable tuning policy type for BlockRangeReduceByKey
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    bool                        _TWO_PHASE_SCATTER,             ///< Whether or not to coalesce output values in shared memory before scattering them to global
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct BlockRangeReduceByKeyPolicy
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
 * Tile status interface types
 ******************************************************************************/

/**
 * Tile status interface for reduction by key.
 *
 */
template <
    typename    Value,
    typename    Offset,
    bool        SINGLE_WORD = (Traits<Value>::PRIMITIVE) && (sizeof(Value) + sizeof(Offset) < 16)>
struct ReduceByKeyScanTileState;


/**
 * Tile status interface for reduction by key, specialized for scan status and value types that
 * cannot be combined into one machine word.
 */
template <
    typename    Value,
    typename    Offset>
struct ReduceByKeyScanTileState<Value, Offset, false> :
    ScanTileState<ItemOffsetPair<Value, Offset> >
{
    typedef ScanTileState<ItemOffsetPair<Value, Offset> > SuperClass;

    /// Constructor
    __host__ __device__ __forceinline__
    ReduceByKeyScanTileState() : SuperClass() {}
};


/**
 * Tile status interface for reduction by key, specialized for scan status and value types that
 * can be combined into one machine word that can be read/written coherently in a single access.
 */
template <
    typename Value,
    typename Offset>
struct ReduceByKeyScanTileState<Value, Offset, true>
{
    typedef ItemOffsetPair<Value, Offset> ItemOffsetPair;

    // Constants
    enum
    {
        PAIR_SIZE           = sizeof(Value) + sizeof(Offset),
        TXN_WORD_SIZE       = 1 << Log2<PAIR_SIZE + 1>::VALUE,
        STATUS_WORD_SIZE    = TXN_WORD_SIZE - PAIR_SIZE,

        TILE_STATUS_PADDING = CUB_PTX_WARP_THREADS,
    };

    // Status word type
    typedef typename If<(STATUS_WORD_SIZE == 8),
        long long,
        typename If<(STATUS_WORD_SIZE == 4),
            int,
            typename If<(STATUS_WORD_SIZE == 2),
                short,
                char>::Type>::Type>::Type StatusWord;

    // Status word type
    typedef typename If<(TXN_WORD_SIZE == 16),
        longlong2,
        typename If<(TXN_WORD_SIZE == 8),
            long long,
            int>::Type>::Type TxnWord;

    // Device word type (for when sizeof(Value) == sizeof(Offset))
    struct TileDescriptorBigStatus
    {
        Offset      offset;
        Value       value;
        StatusWord  status;
    };

    // Device word type (for when sizeof(Value) != sizeof(Offset))
    struct TileDescriptorLittleStatus
    {
        Value       value;
        StatusWord  status;
        Offset      offset;
    };

    // Device word type
    typedef typename If<
            (sizeof(Value) == sizeof(Offset)),
            TileDescriptorBigStatus,
            TileDescriptorLittleStatus>::Type
        TileDescriptor;


    // Device storage
    TileDescriptor *d_tile_status;


    /// Constructor
    __host__ __device__ __forceinline__
    ReduceByKeyScanTileState()
    :
        d_tile_status(NULL)
    {}


    /// Initializer
    __host__ __device__ __forceinline__
    cudaError_t Init(
        int     num_tiles,                          ///< [in] Number of tiles
        void    *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t  temp_storage_bytes)                 ///< [in] Size in bytes of \t d_temp_storage allocation
    {
        d_tile_status = reinterpret_cast<TileDescriptor*>(d_temp_storage);
        return cudaSuccess;
    }


    /**
     * Compute device memory needed for tile status
     */
    __host__ __device__ __forceinline__
    static cudaError_t AllocationSize(
        int     num_tiles,                          ///< [in] Number of tiles
        size_t  &temp_storage_bytes)                ///< [out] Size in bytes of \t d_temp_storage allocation
    {
        temp_storage_bytes = (num_tiles + TILE_STATUS_PADDING) * sizeof(TileDescriptor);       // bytes needed for tile status descriptors
        return cudaSuccess;
    }


    /**
     * Initialize (from device)
     */
    __device__ __forceinline__ void InitializeStatus(int num_tiles)
    {
        int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tile_idx < num_tiles)
        {
            // Not-yet-set
            d_tile_status[TILE_STATUS_PADDING + tile_idx].status = StatusWord(SCAN_TILE_INVALID);
        }

        if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
        {
            // Padding
            d_tile_status[threadIdx.x].status = StatusWord(SCAN_TILE_OOB);
        }
    }


    /**
     * Update the specified tile's inclusive value and corresponding status
     */
    __device__ __forceinline__ void SetInclusive(int tile_idx, ItemOffsetPair tile_inclusive)
    {
        TileDescriptor tile_descriptor;
        tile_descriptor.status = SCAN_TILE_INCLUSIVE;
        tile_descriptor.value = tile_inclusive.value;
        tile_descriptor.offset = tile_inclusive.offset;

        TxnWord alias;
        *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;
        ThreadStore<STORE_CG>(reinterpret_cast<TxnWord*>(d_tile_status + TILE_STATUS_PADDING + tile_idx), alias);
    }


    /**
     * Update the specified tile's partial value and corresponding status
     */
    __device__ __forceinline__ void SetPartial(int tile_idx, ItemOffsetPair tile_partial)
    {
        TileDescriptor tile_descriptor;
        tile_descriptor.status = SCAN_TILE_PARTIAL;
        tile_descriptor.value = tile_partial.value;
        tile_descriptor.offset = tile_partial.offset;

        TxnWord alias;
        *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;
        ThreadStore<STORE_CG>(reinterpret_cast<TxnWord*>(d_tile_status + TILE_STATUS_PADDING + tile_idx), alias);
    }

    /**
     * Wait for the corresponding tile to become non-invalid
     */
    __device__ __forceinline__ void WaitForValid(
        int             tile_idx,
        StatusWord      &status,
        ItemOffsetPair  &value)
    {
        // Use warp-any to determine when all threads have valid status
        TxnWord alias = ThreadLoad<LOAD_CG>(reinterpret_cast<TxnWord*>(d_tile_status + TILE_STATUS_PADDING + tile_idx));
        TileDescriptor tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);

        while ((tile_descriptor.status == SCAN_TILE_INVALID))
        {
            alias = ThreadLoad<LOAD_CG>(reinterpret_cast<TxnWord*>(d_tile_status + TILE_STATUS_PADDING + tile_idx));
            tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);
        }

        status = tile_descriptor.status;
        value.value = tile_descriptor.value;
        value.offset = tile_descriptor.offset;
    }

};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockRangeReduceByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key across a range of tiles
 */
template <
    typename    BlockRangeReduceByKeyPolicy,    ///< Parameterized BlockRangeReduceByKeyPolicy tuning policy type
    typename    KeyInputIterator,               ///< Random-access input iterator type for keys
    typename    KeyOutputIterator,              ///< Random-access output iterator type for keys
    typename    ValueInputIterator,             ///< Random-access input iterator type for values
    typename    ValueOutputIterator,            ///< Random-access output iterator type for values
    typename    EqualityOp,                     ///< Key equality operator type
    typename    ReductionOp,                    ///< Value reduction operator type
    typename    Offset>                         ///< Signed integer type for global offsets
struct BlockRangeReduceByKey
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of key iterator
    typedef typename std::iterator_traits<KeyInputIterator>::value_type Key;

    // Data type of value iterator
    typedef typename std::iterator_traits<ValueInputIterator>::value_type Value;

    // Tile status descriptor interface type
    typedef ReduceByKeyScanTileState<Value, Offset> ScanTileState;

    // Constants
    enum
    {
        BLOCK_THREADS       = BlockRangeReduceByKeyPolicy::BLOCK_THREADS,
        WARPS               = BLOCK_THREADS / CUB_PTX_WARP_THREADS,
        ITEMS_PER_THREAD    = BlockRangeReduceByKeyPolicy::ITEMS_PER_THREAD,
        TWO_PHASE_SCATTER   = (BlockRangeReduceByKeyPolicy::TWO_PHASE_SCATTER) && (ITEMS_PER_THREAD > 1),
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,

        // Whether or not the scan operation has a zero-valued identity value (true if we're performing addition on a primitive type)
        HAS_IDENTITY_ZERO       = (Equals<ReductionOp, cub::Sum>::VALUE) && (Traits<Value>::PRIMITIVE),

        // Whether or not to sync after loading data
        SYNC_AFTER_LOAD         = (BlockRangeReduceByKeyPolicy::LOAD_ALGORITHM != BLOCK_LOAD_DIRECT),

        // Whether or not this is run-length-encoding with a constant iterator as values
        IS_RUN_LENGTH_ENCODE    = (Equals<ValueInputIterator, ConstantInputIterator<Value, size_t> >::VALUE) || (Equals<ValueInputIterator, ConstantInputIterator<Value, int> >::VALUE) || (Equals<ValueInputIterator, ConstantInputIterator<Value, unsigned int> >::VALUE),

    };

    // Cache-modified input iterator wrapper type for keys
    typedef typename If<IsPointer<KeyInputIterator>::VALUE,
            CacheModifiedInputIterator<BlockRangeReduceByKeyPolicy::LOAD_MODIFIER, Key, Offset>,   // Wrap the native input pointer with CacheModifiedValueInputIterator
            KeyInputIterator>::Type                                                                 // Directly use the supplied input iterator type
        WrappedKeyInputIterator;

    // Cache-modified input iterator wrapper type for values
    typedef typename If<IsPointer<ValueInputIterator>::VALUE,
            CacheModifiedInputIterator<BlockRangeReduceByKeyPolicy::LOAD_MODIFIER, Value, Offset>,  // Wrap the native input pointer with CacheModifiedValueInputIterator
            ValueInputIterator>::Type                                                                // Directly use the supplied input iterator type
        WrappedValueInputIterator;

    // Value-offset tuple type for scanning (maps accumulated values to segment index)
    typedef ItemOffsetPair<Value, Offset> ValueOffsetPair;

    // Reduce-value-by-segment scan operator
    struct ReduceByKeyOp
    {
        ReductionOp op;                 ///< Wrapped reduction operator

        /// Constructor
        __device__ __forceinline__ ReduceByKeyOp(ReductionOp op) : op(op) {}

        /// Scan operator (specialized for sum on primitive types)
        __device__ __forceinline__ ValueOffsetPair operator()(
            const ValueOffsetPair   &first,             ///< First partial reduction
            const ValueOffsetPair   &second,            ///< Second partial reduction
            Int2Type<true>          has_identity_zero)  ///< Whether the operation has a zero-valued identity
        {
            Value select = (second.offset) ? 0 : first.value;

            ValueOffsetPair retval;
            retval.offset = first.offset + second.offset;
            retval.value = op(select, second.value);
            return retval;
        }

        /// Scan operator (specialized for reductions without zero-valued identity)
        __device__ __forceinline__ ValueOffsetPair operator()(
            const ValueOffsetPair   &first,             ///< First partial reduction
            const ValueOffsetPair   &second,            ///< Second partial reduction
            Int2Type<false>         has_identity_zero)  ///< Whether the operation has a zero-valued identity
        {
#if (__CUDA_ARCH__ > 130)
            // This expression uses less registers and is faster when compiled with nvvm
            ValueOffsetPair retval;
            retval.offset = first.offset + second.offset;
            if (second.offset)
            {
                retval.value = second.value;
                return retval;
            }
            else
            {
                retval.value = op(first.value, second.value);
                return retval;
            }
#else
            // This expression uses less registers and is faster when compiled with Open64
            ValueOffsetPair retval;
            retval.offset = first.offset + second.offset;
            retval.value = (second.offset) ?
                    second.value :                          // The second partial reduction spans a segment reset, so it's value aggregate becomes the running aggregate
                    op(first.value, second.value);          // The second partial reduction does not span a reset, so accumulate both into the running aggregate
            return retval;
#endif
        }

        /// Scan operator
        __device__ __forceinline__ ValueOffsetPair operator()(
            const ValueOffsetPair &first,       ///< First partial reduction
            const ValueOffsetPair &second)      ///< Second partial reduction
        {
            return (*this)(first, second, Int2Type<HAS_IDENTITY_ZERO>());
        }
    };

    // Parameterized BlockLoad type for keys
    typedef BlockLoad<
            WrappedKeyInputIterator,
            BlockRangeReduceByKeyPolicy::BLOCK_THREADS,
            BlockRangeReduceByKeyPolicy::ITEMS_PER_THREAD,
            BlockRangeReduceByKeyPolicy::LOAD_ALGORITHM>
        BlockLoadKeys;

    // Parameterized BlockLoad type for values
    typedef BlockLoad<
            WrappedValueInputIterator,
            BlockRangeReduceByKeyPolicy::BLOCK_THREADS,
            BlockRangeReduceByKeyPolicy::ITEMS_PER_THREAD,
            (IS_RUN_LENGTH_ENCODE) ?
                BLOCK_LOAD_DIRECT :
                (BlockLoadAlgorithm) BlockRangeReduceByKeyPolicy::LOAD_ALGORITHM>
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
            ValueOffsetPair,
            BlockRangeReduceByKeyPolicy::BLOCK_THREADS,
            BlockRangeReduceByKeyPolicy::SCAN_ALGORITHM>
        BlockScanAllocations;

    // Callback type for obtaining tile prefix during block scan
    typedef BlockScanLookbackPrefixOp<
            ValueOffsetPair,
            ReduceByKeyOp,
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

    WrappedKeyInputIterator         d_keys_in;          ///< Input keys
    KeyOutputIterator               d_keys_out;         ///< Output keys

    WrappedValueInputIterator       d_values_in;        ///< Input values
    ValueOutputIterator             d_values_out;       ///< Output values

    InequalityWrapper<EqualityOp>   inequality_op;      ///< Key inequality operator
    ReduceByKeyOp                   scan_op;            ///< Reduce-value-by flag scan operator
    Offset                          num_items;          ///< Total number of input items


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    BlockRangeReduceByKey(
        TempStorage                 &temp_storage,      ///< Reference to temp_storage
        KeyInputIterator            d_keys_in,          ///< Input keys
        KeyOutputIterator           d_keys_out,         ///< Output keys
        ValueInputIterator          d_values_in,        ///< Input values
        ValueOutputIterator         d_values_out,       ///< Output values
        EqualityOp                  equality_op,        ///< Key equality operator
        ReductionOp                 reduction_op,       ///< Value reduction operator
        Offset                      num_items)          ///< Total number of input items
    :
        temp_storage(temp_storage.Alias()),
        d_keys_in(d_keys_in),
        d_keys_out(d_keys_out),
        d_values_in(d_values_in),
        d_values_out(d_values_out),
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
        ValueOffsetPair     (&values_and_segments)[ITEMS_PER_THREAD],
        ValueOffsetPair     &block_aggregate,
        Int2Type<true>      has_identity)
    {
        ValueOffsetPair identity;
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
        ValueOffsetPair     (&values_and_segments)[ITEMS_PER_THREAD],
        ValueOffsetPair     &block_aggregate,
        Int2Type<false>     has_identity)
    {
        BlockScanAllocations(temp_storage.scan).ExclusiveScan(values_and_segments, values_and_segments, scan_op, block_aggregate);
    }

    /**
     * Scan with identity (subsequent tile)
     */
    __device__ __forceinline__
    void ScanBlock(
        ValueOffsetPair             (&values_and_segments)[ITEMS_PER_THREAD],
        ValueOffsetPair             &block_aggregate,
        LookbackPrefixCallbackOp    &prefix_op,
        Int2Type<true>              has_identity)
    {
        ValueOffsetPair identity;
        identity.value = 0;
        identity.offset = 0;
        BlockScanAllocations(temp_storage.scan).ExclusiveScan(values_and_segments, values_and_segments, identity, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Scan without identity (subsequent tile).  Without an identity, the first output item is undefined.
     */
    __device__ __forceinline__
    void ScanBlock(
        ValueOffsetPair             (&values_and_segments)[ITEMS_PER_THREAD],
        ValueOffsetPair             &block_aggregate,
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
        ValueOffsetPair (&values_and_segments)[ITEMS_PER_THREAD])
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
        ValueOffsetPair     (&values_and_segments)[ITEMS_PER_THREAD],
        Offset              (&flags)[ITEMS_PER_THREAD],
        Offset              tile_num_flags,
        Int2Type<ITEM>      iteration)
    {
        // Scatter key
        if (flags[ITEM])
        {
            d_keys_out[values_and_segments[ITEM].offset] = keys[ITEM];
        }

        bool is_first_flag     = FIRST_TILE && (ITEM == 0) && (threadIdx.x == 0);
        bool is_oob_value      = (LAST_TILE) && (Offset(threadIdx.x * ITEMS_PER_THREAD) + ITEM == num_remaining);

        // Scatter value reduction
        if (((flags[ITEM] || is_oob_value)) && (!is_first_flag))
        {
            d_values_out[values_and_segments[ITEM].offset - 1] = values_and_segments[ITEM].value;
        }

        ScatterDirect<LAST_TILE, FIRST_TILE>(num_remaining, keys, values_and_segments, flags, tile_num_flags, Int2Type<ITEM + 1>());
    }

    template <bool LAST_TILE, bool FIRST_TILE>
    __device__ __forceinline__ void ScatterDirect(
        Offset                      num_remaining,
        Key                         (&keys)[ITEMS_PER_THREAD],
        ValueOffsetPair             (&values_and_segments)[ITEMS_PER_THREAD],
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
        ValueOffsetPair (&values_and_segments)[ITEMS_PER_THREAD],
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
        StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out + tile_num_flags_prefix, keys, tile_num_flags);

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
        StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_out + tile_num_flags_prefix, values, exchange_count);

        __syncthreads();
    }


    /**
     * Scatter flagged items
     */
    template <bool LAST_TILE, bool FIRST_TILE>
    __device__ __forceinline__ void Scatter(
        Offset          num_remaining,
        Key             (&keys)[ITEMS_PER_THREAD],
        ValueOffsetPair (&values_and_segments)[ITEMS_PER_THREAD],
        Offset          (&flags)[ITEMS_PER_THREAD],
        Offset          tile_num_flags,
        Offset          tile_num_flags_prefix)
    {
        // Do a one-phase scatter if (a) two-phase is disabled or (b) the average number of selected items per thread is less than one
        if ((TWO_PHASE_SCATTER) && ((tile_num_flags >> Log2<BLOCK_THREADS>::VALUE) > 0))
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
     * Process a tile of input (dynamic domino scan)
     */
    template <
        bool                LAST_TILE>
    __device__ __forceinline__ ValueOffsetPair ConsumeTile(
        Offset              num_items,          ///< Total number of global input items
        Offset              num_remaining,      ///< Number of global input items remaining (including this tile)
        int                 tile_idx,           ///< Tile index
        Offset              block_offset,       ///< Tile offset
        ScanTileState  &tile_status)       ///< Global list of tile status
    {
            Key                 keys[ITEMS_PER_THREAD];                         // Tile keys
            Value               values[ITEMS_PER_THREAD];                       // Tile values
            Offset              flags[ITEMS_PER_THREAD];                        // Segment head flags
            ValueOffsetPair     values_and_segments[ITEMS_PER_THREAD];          // Zipped values and segment flags|indices

        ValueOffsetPair     running_total;                                  // Running count of segments and current value aggregate (including this tile)

        if (tile_idx == 0)
        {
            // First tile

            // Load keys and values
            if (LAST_TILE)
            {
                BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + block_offset, keys, num_remaining);
            }
            else
            {
                BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + block_offset, keys);
            }

            if (SYNC_AFTER_LOAD)
                __syncthreads();

            // Load values
            if (LAST_TILE)
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values, num_remaining);
            else
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values);

            if (SYNC_AFTER_LOAD)
                __syncthreads();

            // Set head flags.  First tile sets the first flag for the first item
            BlockDiscontinuityKeys(temp_storage.discontinuity).FlagHeads(flags, keys, inequality_op);

            // Zip values and flags
            ZipValuesAndFlags<LAST_TILE>(num_remaining, values, flags, values_and_segments);

            // Exclusive scan of values and flags
            ValueOffsetPair block_aggregate;
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

            // Load keys and values
            if (LAST_TILE)
            {
                BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + block_offset, keys, num_remaining);
            }
            else
            {
                BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + block_offset, keys);
            }

            if (SYNC_AFTER_LOAD)
                __syncthreads();

            // Load values
            if (LAST_TILE)
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values, num_remaining);
            else
                BlockLoadValues(temp_storage.load_values).Load(d_values_in + block_offset, values);

            if (SYNC_AFTER_LOAD)
                __syncthreads();

            // Obtain the last key in the previous tile to compare with
            Key tile_predecessor_key = (threadIdx.x == 0) ?
                d_keys_in[block_offset - 1] :
                ZeroInitialize<Key>();

            // Set head flags
            BlockDiscontinuityKeys(temp_storage.discontinuity).FlagHeads(flags, keys, inequality_op, tile_predecessor_key);

            // Zip values and flags
            ZipValuesAndFlags<LAST_TILE>(num_remaining, values, flags, values_and_segments);

            // Exclusive scan of values and flags
            ValueOffsetPair block_aggregate;
            LookbackPrefixCallbackOp prefix_op(tile_status, temp_storage.prefix, scan_op, tile_idx);

            ScanBlock(values_and_segments, block_aggregate, prefix_op, Int2Type<HAS_IDENTITY_ZERO>());
            running_total = prefix_op.inclusive_prefix;

            // Scatter flagged items
            Scatter<LAST_TILE, false>(num_remaining, keys, values_and_segments, flags, block_aggregate.offset, prefix_op.exclusive_prefix.offset);
        }

        return running_total;
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic domino scan
     */
    template <typename NumSegmentsIterator>         ///< Output iterator type for recording number of items selected
    __device__ __forceinline__ void ConsumeRange(
        int                     num_tiles,          ///< Total number of input tiles
        GridQueue<int>          queue,              ///< Queue descriptor for assigning tiles of work to thread blocks
        ScanTileState      &tile_status,       ///< Global list of tile status
        NumSegmentsIterator     d_num_segments)     ///< Output pointer for total number of segments identified
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
            ValueOffsetPair running_total = ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            // Output the total number of items selected
            if (threadIdx.x == 0)
            {
                *d_num_segments = running_total.offset;

                // If the last tile is a whole tile, the inclusive prefix contains accumulated value reduction for the last segment
                if (num_remaining == TILE_ITEMS)
                {
                    d_values_out[running_total.offset - 1] = running_total.value;
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
            if (SYNC_AFTER_LOAD)
                __syncthreads();

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
            ValueOffsetPair running_total = ConsumeTile<true>(num_items, num_remaining, tile_idx, block_offset, tile_status);

            if ((threadIdx.x == 0))
            {
                // Output the total number of items selected
                *d_num_segments = running_total.offset;

                // If the last tile is a whole tile, the inclusive prefix contains accumulated value reduction for the last segment
                if (num_remaining == TILE_ITEMS)
                {
                    d_values_out[running_total.offset - 1] = running_total.value;
                }
            }
        }
#endif
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

