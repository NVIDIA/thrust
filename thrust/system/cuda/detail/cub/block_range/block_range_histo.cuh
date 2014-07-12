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
 * cub::BlockRangeHistogram implements a stateful abstraction of CUDA thread blocks for participating in device-wide selection across a range of tiles.
 */

#pragma once

#include <iterator>

#include "specializations/block_range_histo_gatomic.cuh"
#include "specializations/block_range_histo_satomic.cuh"
#include "specializations/block_range_histo_sort.cuh"
#include "../util_type.cuh"
#include "../grid/grid_mapping.cuh"
#include "../grid/grid_even_share.cuh"
#include "../grid/grid_queue.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Algorithmic variants
 ******************************************************************************/


/**
 * \brief DeviceHistogramAlgorithm enumerates alternative algorithms for BlockRangeHistogram.
 */
enum DeviceHistogramAlgorithm
{

    /**
     * \par Overview
     * A two-kernel approach in which:
     * -# Thread blocks in the first kernel aggregate their own privatized
     *    histograms using block-wide sorting (see BlockHistogramAlgorithm::BLOCK_HISTO_SORT).
     * -# A single thread block in the second kernel reduces them into the output histogram(s).
     *
     * \par Performance Considerations
     * Delivers consistent throughput regardless of sample bin distribution.
     *
     * However, because histograms are privatized in shared memory, a large
     * number of bins (e.g., thousands) may adversely affect occupancy and
     * performance (or even the ability to launch).
     */
    DEVICE_HISTO_SORT,


    /**
     * \par Overview
     * A two-kernel approach in which:
     * -# Thread blocks in the first kernel aggregate their own privatized
     *    histograms using shared-memory \p atomicAdd().
     * -# A single thread block in the second kernel reduces them into the
     *    output histogram(s).
     *
     * \par Performance Considerations
     * Performance is strongly tied to the hardware implementation of atomic
     * addition, and may be significantly degraded for non uniformly-random
     * input distributions where many concurrent updates are likely to be
     * made to the same bin counter.
     *
     * However, because histograms are privatized in shared memory, a large
     * number of bins (e.g., thousands) may adversely affect occupancy and
     * performance (or even the ability to launch).
     */
    DEVICE_HISTO_SHARED_ATOMIC,


    /**
     * \par Overview
     * A single-kernel approach in which thread blocks update the output histogram(s) directly
     * using global-memory \p atomicAdd().
     *
     * \par Performance Considerations
     * Performance is strongly tied to the hardware implementation of atomic
     * addition, and may be significantly degraded for non uniformly-random
     * input distributions where many concurrent updates are likely to be
     * made to the same bin counter.
     *
     * Performance is not significantly impacted when computing histograms having large
     * numbers of bins (e.g., thousands).
     */
    DEVICE_HISTO_GLOBAL_ATOMIC,

};


/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for BlockRangeHistogram
 */
template <
    int                             _BLOCK_THREADS,         ///< Threads per thread block
    int                             _ITEMS_PER_THREAD,      ///< Items per thread (per tile of input)
    DeviceHistogramAlgorithm        _HISTO_ALGORITHM,       ///< Cooperative histogram algorithm to use
    GridMappingStrategy             _GRID_MAPPING>          ///< How to map tiles of input onto thread blocks
struct BlockRangeHistogramPolicy
{
    enum
    {
        BLOCK_THREADS       = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD    = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
    };

    static const DeviceHistogramAlgorithm   HISTO_ALGORITHM     = _HISTO_ALGORITHM;     ///< Cooperative histogram algorithm to use
    static const GridMappingStrategy        GRID_MAPPING        = _GRID_MAPPING;        ///< How to map tiles of input onto thread blocks
};



/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief BlockRangeHistogram implements a stateful abstraction of CUDA thread blocks for participating in device-wide selection across a range of tiles.
 */
template <
    typename    BlockRangeHistogramPolicy,      ///< Parameterized BlockRangeHistogramPolicy tuning policy type
    int         BINS,                           ///< Number of histogram bins per channel
    int         CHANNELS,                       ///< Number of channels interleaved in the input data (may be greater than the number of active channels being histogrammed)
    int         ACTIVE_CHANNELS,                ///< Number of channels actively being histogrammed
    typename    InputIterator,                  ///< Random-access input iterator type for reading samples.  Must have an an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1]
    typename    HistoCounter,                   ///< Integer type for counting sample occurrences per histogram bin
    typename    Offset>                         ///< Signed integer type for global offsets
struct BlockRangeHistogram
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Histogram grid algorithm
    static const DeviceHistogramAlgorithm HISTO_ALGORITHM = BlockRangeHistogramPolicy::HISTO_ALGORITHM;

    // Alternative internal implementation types
    typedef BlockRangeHistogramSort<            BlockRangeHistogramPolicy, BINS, CHANNELS, ACTIVE_CHANNELS, InputIterator, HistoCounter, Offset>   BlockRangeHistogramSortT;
    typedef BlockRangeHistogramSharedAtomic<    BlockRangeHistogramPolicy, BINS, CHANNELS, ACTIVE_CHANNELS, InputIterator, HistoCounter, Offset>   BlockRangeHistogramSharedAtomicT;
    typedef BlockRangeHistogramGlobalAtomic<    BlockRangeHistogramPolicy, BINS, CHANNELS, ACTIVE_CHANNELS, InputIterator, HistoCounter, Offset>   BlockRangeHistogramGlobalAtomicT;

    // Internal block sweep histogram type
    typedef typename If<(HISTO_ALGORITHM == DEVICE_HISTO_SORT),
        BlockRangeHistogramSortT,
        typename If<(HISTO_ALGORITHM == DEVICE_HISTO_SHARED_ATOMIC),
            BlockRangeHistogramSharedAtomicT,
            BlockRangeHistogramGlobalAtomicT>::Type>::Type InternalBlockDelegate;

    enum
    {
        TILE_ITEMS = InternalBlockDelegate::TILE_ITEMS,
    };


    // Temporary storage type
    typedef typename InternalBlockDelegate::TempStorage TempStorage;

    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    // Internal block delegate
    InternalBlockDelegate internal_delegate;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ BlockRangeHistogram(
        TempStorage         &temp_storage,                                  ///< Reference to temp_storage
        InputIterator     d_in,                                           ///< Input data to reduce
        HistoCounter*       (&d_out_histograms)[ACTIVE_CHANNELS])           ///< Reference to output histograms
    :
        internal_delegate(temp_storage, d_in, d_out_histograms)
    {}


    /**
     * \brief Reduce a consecutive segment of input tiles
     */
    __device__ __forceinline__ void ConsumeRange(
        Offset   block_offset,                       ///< [in] Threadblock begin offset (inclusive)
        Offset   block_end)                          ///< [in] Threadblock end offset (exclusive)
    {
        // Consume subsequent full tiles of input
        while (block_offset + TILE_ITEMS <= block_end)
        {
            internal_delegate.ConsumeTile<true>(block_offset);
            block_offset += TILE_ITEMS;
        }

        // Consume a partially-full tile
        if (block_offset < block_end)
        {
            int valid_items = block_end - block_offset;
            internal_delegate.ConsumeTile<false>(block_offset, valid_items);
        }

        // Aggregate output
        internal_delegate.AggregateOutput();
    }


    /**
     * Reduce a consecutive segment of input tiles
     */
    __device__ __forceinline__ void ConsumeRange(
        Offset                              num_items,          ///< [in] Total number of global input items
        GridEvenShare<Offset>               &even_share,        ///< [in] GridEvenShare descriptor
        GridQueue<Offset>                   &queue,             ///< [in,out] GridQueue descriptor
        Int2Type<GRID_MAPPING_EVEN_SHARE>   is_even_share)      ///< [in] Marker type indicating this is an even-share mapping
    {
        even_share.BlockInit();
        ConsumeRange(even_share.block_offset, even_share.block_end);
    }


    /**
     * Dequeue and reduce tiles of items as part of a inter-block scan
     */
    __device__ __forceinline__ void ConsumeRange(
        int                 num_items,          ///< Total number of input items
        GridQueue<Offset>   queue)              ///< Queue descriptor for assigning tiles of work to thread blocks
    {
        // Shared block offset
        __shared__ Offset shared_block_offset;

        // We give each thread block at least one tile of input.
        Offset block_offset      = blockIdx.x * TILE_ITEMS;
        Offset even_share_base   = gridDim.x * TILE_ITEMS;

        // Process full tiles of input
        while (block_offset + TILE_ITEMS <= num_items)
        {
            internal_delegate.ConsumeTile<true>(block_offset);

            // Dequeue up to TILE_ITEMS
            if (threadIdx.x == 0)
                shared_block_offset = queue.Drain(TILE_ITEMS) + even_share_base;

            __syncthreads();

            block_offset = shared_block_offset;

            __syncthreads();
        }

        // Consume a partially-full tile
        if (block_offset < num_items)
        {
            int valid_items = num_items - block_offset;
            internal_delegate.ConsumeTile<false>(block_offset, valid_items);
        }

        // Aggregate output
        internal_delegate.AggregateOutput();
    }


    /**
     * Dequeue and reduce tiles of items as part of a inter-block scan
     */
    __device__ __forceinline__ void ConsumeRange(
        Offset                          num_items,          ///< [in] Total number of global input items
        GridEvenShare<Offset>           &even_share,        ///< [in] GridEvenShare descriptor
        GridQueue<Offset>               &queue,             ///< [in,out] GridQueue descriptor
        Int2Type<GRID_MAPPING_DYNAMIC>  is_dynamic)         ///< [in] Marker type indicating this is a dynamic mapping
    {
        ConsumeRange(num_items, queue);
    }


};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

