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
 * cub::BlockRangeHistogramSort implements a stateful abstraction of CUDA thread blocks for histogramming multiple tiles as part of device-wide histogram using local sorting
 */

#pragma once

#include <iterator>

#include "../../block/block_radix_sort.cuh"
#include "../../block/block_discontinuity.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * BlockRangeHistogramSort implements a stateful abstraction of CUDA thread blocks for histogramming multiple tiles as part of device-wide histogram using local sorting
 */
template <
    typename    BlockRangeHistogramPolicy,          ///< Tuning policy
    int         BINS,                           ///< Number of histogram bins per channel
    int         CHANNELS,                       ///< Number of channels interleaved in the input data (may be greater than the number of active channels being histogrammed)
    int         ACTIVE_CHANNELS,                ///< Number of channels actively being histogrammed
    typename    InputIterator,                ///< The input iterator type \iterator.  Must have an an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1]
    typename    HistoCounter,                   ///< Integer type for counting sample occurrences per histogram bin
    typename    Offset>                          ///< Signed integer type for global offsets
struct BlockRangeHistogramSort
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Sample type
    typedef typename std::iterator_traits<InputIterator>::value_type SampleT;

    // Constants
    enum
    {
        BLOCK_THREADS               = BlockRangeHistogramPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD            = BlockRangeHistogramPolicy::ITEMS_PER_THREAD,
        TILE_CHANNEL_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
        TILE_ITEMS                  = TILE_CHANNEL_ITEMS * CHANNELS,

        STRIPED_COUNTERS_PER_THREAD = (BINS + BLOCK_THREADS - 1) / BLOCK_THREADS,
    };

    // Parameterize BlockRadixSort type for our thread block
    typedef BlockRadixSort<SampleT, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    // Parameterize BlockDiscontinuity type for our thread block
    typedef BlockDiscontinuity<SampleT, BLOCK_THREADS> BlockDiscontinuityT;

    /// Shared memory type required by this thread block
    union _TempStorage
    {
        // Storage for sorting bin values
        typename BlockRadixSortT::TempStorage sort;

        struct
        {
            // Storage for detecting discontinuities in the tile of sorted bin values
            typename BlockDiscontinuityT::TempStorage flag;

            // Storage for noting begin/end offsets of bin runs in the tile of sorted bin values
            int run_begin[BLOCK_THREADS * STRIPED_COUNTERS_PER_THREAD];
            int run_end[BLOCK_THREADS * STRIPED_COUNTERS_PER_THREAD];
        };
    };


    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    // Discontinuity functor
    struct DiscontinuityOp
    {
        // Reference to temp_storage
        _TempStorage &temp_storage;

        // Constructor
        __device__ __forceinline__ DiscontinuityOp(_TempStorage &temp_storage) :
            temp_storage(temp_storage)
        {}

        // Discontinuity predicate
        __device__ __forceinline__ bool operator()(const SampleT &a, const SampleT &b, int b_index)
        {
            if (a != b)
            {
                // Note the begin/end offsets in shared storage
                temp_storage.run_begin[b] = b_index;
                temp_storage.run_end[a] = b_index;

                return true;
            }
            else
            {
                return false;
            }
        }
    };


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Reference to temp_storage
    _TempStorage &temp_storage;

    /// Histogram counters striped across threads
    HistoCounter thread_counters[ACTIVE_CHANNELS][STRIPED_COUNTERS_PER_THREAD];

    /// Reference to output histograms
    HistoCounter* (&d_out_histograms)[ACTIVE_CHANNELS];

    /// Input data to reduce
    InputIterator d_in;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ BlockRangeHistogramSort(
        TempStorage         &temp_storage,                                  ///< Reference to temp_storage
        InputIterator     d_in,                                           ///< Input data to reduce
        HistoCounter*       (&d_out_histograms)[ACTIVE_CHANNELS])           ///< Reference to output histograms
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_out_histograms(d_out_histograms)
    {
        // Initialize histogram counters striped across threads
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        {
            #pragma unroll
            for (int COUNTER = 0; COUNTER < STRIPED_COUNTERS_PER_THREAD; ++COUNTER)
            {
                thread_counters[CHANNEL][COUNTER] = 0;
            }
        }
    }


    /**
     * Composite a tile of input items
     */
    __device__ __forceinline__ void Composite(
        SampleT   (&items)[ITEMS_PER_THREAD],                     ///< Tile of samples
        HistoCounter    thread_counters[STRIPED_COUNTERS_PER_THREAD])   ///< Histogram counters striped across threads
    {
        // Sort bytes in blocked arrangement
        BlockRadixSortT(temp_storage.sort).Sort(items);

        __syncthreads();

        // Initialize the shared memory's run_begin and run_end for each bin
        #pragma unroll
        for (int COUNTER = 0; COUNTER < STRIPED_COUNTERS_PER_THREAD; ++COUNTER)
        {
            temp_storage.run_begin[(COUNTER * BLOCK_THREADS) + threadIdx.x] = TILE_CHANNEL_ITEMS;
            temp_storage.run_end[(COUNTER * BLOCK_THREADS) + threadIdx.x] = TILE_CHANNEL_ITEMS;
        }

        __syncthreads();

        // Note the begin/end run offsets of bin runs in the sorted tile
        int flags[ITEMS_PER_THREAD];                // unused
        DiscontinuityOp flag_op(temp_storage);
        BlockDiscontinuityT(temp_storage.flag).FlagHeads(flags, items, flag_op);

        // Update begin for first item
        if (threadIdx.x == 0) temp_storage.run_begin[items[0]] = 0;

        __syncthreads();

        // Composite into histogram
        // Initialize the shared memory's run_begin and run_end for each bin
        #pragma unroll
        for (int COUNTER = 0; COUNTER < STRIPED_COUNTERS_PER_THREAD; ++COUNTER)
        {
            int          bin            = (COUNTER * BLOCK_THREADS) + threadIdx.x;
            HistoCounter run_length     = temp_storage.run_end[bin] - temp_storage.run_begin[bin];

            thread_counters[COUNTER] += run_length;
        }
    }


    /**
     * Process one channel within a tile.
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ConsumeTileChannel(
        int     channel,
        Offset   block_offset,
        int     valid_items)
    {
        // Load items in striped fashion
        if (FULL_TILE)
        {
            // Full tile of samples to read and composite
            SampleT items[ITEMS_PER_THREAD];

            // Unguarded loads
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                items[ITEM] = d_in[channel + block_offset + (ITEM * BLOCK_THREADS * CHANNELS) + (threadIdx.x * CHANNELS)];
            }

            // Composite our histogram data
            Composite(items, thread_counters[channel]);
        }
        else
        {
            // Only a partially-full tile of samples to read and composite
            SampleT items[ITEMS_PER_THREAD];

            // Assign our tid as the bin for out-of-bounds items (to give an even distribution), and keep track of how oob items to subtract out later
            int bounds = (valid_items - (threadIdx.x * CHANNELS));

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                items[ITEM] = ((ITEM * BLOCK_THREADS * CHANNELS) < bounds) ?
                    d_in[channel + block_offset + (ITEM * BLOCK_THREADS * CHANNELS) + (threadIdx.x * CHANNELS)] :
                    0;
            }

            // Composite our histogram data
            Composite(items, thread_counters[channel]);

            __syncthreads();

            // Correct the overcounting in the zero-bin from invalid (out-of-bounds) items
            if (threadIdx.x == 0)
            {
                int extra = (TILE_ITEMS - valid_items) / CHANNELS;
                thread_counters[channel][0] -= extra;
            }
        }
    }


    /**
     * Template iteration over channels (to silence not-unrolled warnings for SM10-13).  Inductive step.
     */
    template <bool FULL_TILE, int CHANNEL, int END>
    struct IterateChannels
    {
        /**
         * Process one channel within a tile.
         */
        static __device__ __forceinline__ void ConsumeTileChannel(
            BlockRangeHistogramSort *cta,
            Offset               block_offset,
            int                 valid_items)
        {
            __syncthreads();

            cta->ConsumeTileChannel<FULL_TILE>(CHANNEL, block_offset, valid_items);

            IterateChannels<FULL_TILE, CHANNEL + 1, END>::ConsumeTileChannel(cta, block_offset, valid_items);
        }
    };


    /**
     * Template iteration over channels (to silence not-unrolled warnings for SM10-13).  Base step.
     */
    template <bool FULL_TILE, int END>
    struct IterateChannels<FULL_TILE, END, END>
    {
        static __device__ __forceinline__ void ConsumeTileChannel(BlockRangeHistogramSort *cta, Offset block_offset, int valid_items) {}
    };


    /**
     * Process a single tile of input
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        Offset   block_offset,               ///< The offset the tile to consume
        int     valid_items = TILE_ITEMS)   ///< The number of valid items in the tile
    {
        // First channel
        ConsumeTileChannel<FULL_TILE>(0, block_offset, valid_items);

        // Iterate through remaining channels
        IterateChannels<FULL_TILE, 1, ACTIVE_CHANNELS>::ConsumeTileChannel(this, block_offset, valid_items);
    }


    /**
     * Aggregate results into output
     */
    __device__ __forceinline__ void AggregateOutput()
    {
        // Copy counters striped across threads into the histogram output
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        {
            int channel_offset  = (blockIdx.x * BINS);

            #pragma unroll
            for (int COUNTER = 0; COUNTER < STRIPED_COUNTERS_PER_THREAD; ++COUNTER)
            {
                int bin = (COUNTER * BLOCK_THREADS) + threadIdx.x;

                if ((STRIPED_COUNTERS_PER_THREAD * BLOCK_THREADS == BINS) || (bin < BINS))
                {
                    d_out_histograms[CHANNEL][channel_offset + bin] = thread_counters[CHANNEL][COUNTER];
                }
            }
        }
    }
};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

