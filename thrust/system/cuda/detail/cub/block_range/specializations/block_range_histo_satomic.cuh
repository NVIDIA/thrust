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
 * cub::BlockRangeHistogramSharedAtomic implements a stateful abstraction of CUDA thread blocks for histogramming multiple tiles as part of device-wide histogram using shared atomics
 */

#pragma once

#include <iterator>

#include "../../util_type.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * BlockRangeHistogramSharedAtomic implements a stateful abstraction of CUDA thread blocks for histogramming multiple tiles as part of device-wide histogram using shared atomics
 */
template <
    typename    BlockRangeHistogramPolicy,		///< Tuning policy
    int         BINS,                           ///< Number of histogram bins
    int         CHANNELS,                       ///< Number of channels interleaved in the input data (may be greater than the number of active channels being histogrammed)
    int         ACTIVE_CHANNELS,                ///< Number of channels actively being histogrammed
    typename    InputIterator,                	///< The input iterator type \iterator.  Must have an an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1]
    typename    HistoCounter,                   ///< Integer type for counting sample occurrences per histogram bin
    typename    Offset>                          ///< Signed integer type for global offsets
struct BlockRangeHistogramSharedAtomic
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Sample type
    typedef typename std::iterator_traits<InputIterator>::value_type SampleT;

    // Constants
    enum
    {
        BLOCK_THREADS       = BlockRangeHistogramPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD    = BlockRangeHistogramPolicy::ITEMS_PER_THREAD,
        TILE_CHANNEL_ITEMS  = BLOCK_THREADS * ITEMS_PER_THREAD,
        TILE_ITEMS          = TILE_CHANNEL_ITEMS * CHANNELS,
    };

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        HistoCounter histograms[ACTIVE_CHANNELS][BINS + 1];  // One word of padding between channel histograms to prevent warps working on different histograms from hammering on the same bank
    };


    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    /// Reference to temp_storage
    _TempStorage &temp_storage;

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
    __device__ __forceinline__ BlockRangeHistogramSharedAtomic(
        TempStorage         &temp_storage,                                  ///< Reference to temp_storage
        InputIterator     d_in,                                           ///< Input data to reduce
        HistoCounter*       (&d_out_histograms)[ACTIVE_CHANNELS])           ///< Reference to output histograms
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_out_histograms(d_out_histograms)
    {
        // Initialize histogram bin counts to zeros
        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        {
            int histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= BINS; histo_offset += BLOCK_THREADS)
            {
                this->temp_storage.histograms[CHANNEL][histo_offset + threadIdx.x] = 0;
            }
            // Finish up with guarded initialization if necessary
            if ((BINS % BLOCK_THREADS != 0) && (histo_offset + threadIdx.x < BINS))
            {
                this->temp_storage.histograms[CHANNEL][histo_offset + threadIdx.x] = 0;
            }
        }

        __syncthreads();
    }


    /**
     * Process a single tile of input
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        Offset   block_offset,               ///< The offset the tile to consume
        int     valid_items = TILE_ITEMS)   ///< The number of valid items in the tile
    {
        if (FULL_TILE)
        {
            // Full tile of samples to read and composite
            SampleT items[ITEMS_PER_THREAD][CHANNELS];

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < CHANNELS; ++CHANNEL)
                {
                    if (CHANNEL < ACTIVE_CHANNELS)
                    {
                        items[ITEM][CHANNEL] = d_in[block_offset + (ITEM * BLOCK_THREADS * CHANNELS) + (threadIdx.x * CHANNELS) + CHANNEL];
                    }
                }
            }

            __threadfence_block();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < CHANNELS; ++CHANNEL)
                {
                    if (CHANNEL < ACTIVE_CHANNELS)
                    {
                        atomicAdd(temp_storage.histograms[CHANNEL] + items[ITEM][CHANNEL], 1);
                    }
                }
            }

            __threadfence_block();
        }
        else
        {
            // Only a partially-full tile of samples to read and composite
            int bounds = valid_items - (threadIdx.x * CHANNELS);

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < CHANNELS; ++CHANNEL)
                {
                    if (((ACTIVE_CHANNELS == CHANNELS) || (CHANNEL < ACTIVE_CHANNELS)) && ((ITEM * BLOCK_THREADS * CHANNELS) + CHANNEL < bounds))
                    {
                        SampleT item = d_in[block_offset + (ITEM * BLOCK_THREADS * CHANNELS) + (threadIdx.x * CHANNELS) + CHANNEL];
                        atomicAdd(temp_storage.histograms[CHANNEL] + item, 1);
                    }
                }
            }

        }
    }


    /**
     * Aggregate results into output
     */
    __device__ __forceinline__ void AggregateOutput()
    {
        // Barrier to ensure shared memory histograms are coherent
        __syncthreads();

        // Copy shared memory histograms to output
        int channel_offset = (blockIdx.x * BINS);

        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        {
            int histo_offset = 0;

            #pragma unroll
            for(; histo_offset + BLOCK_THREADS <= BINS; histo_offset += BLOCK_THREADS)
            {
                HistoCounter count = temp_storage.histograms[CHANNEL][histo_offset + threadIdx.x];

                d_out_histograms[CHANNEL][channel_offset + histo_offset + threadIdx.x] = count;
            }

            // Finish up with guarded initialization if necessary
            if ((BINS % BLOCK_THREADS != 0) && (histo_offset + threadIdx.x < BINS))
            {
                HistoCounter count = temp_storage.histograms[CHANNEL][histo_offset + threadIdx.x];

                d_out_histograms[CHANNEL][channel_offset + histo_offset + threadIdx.x] = count;
            }
        }
    }
};



}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

