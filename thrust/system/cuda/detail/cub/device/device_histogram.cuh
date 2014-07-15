
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
 * cub::DeviceHistogram provides device-wide parallel operations for constructing histogram(s) from a sequence of samples data residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "dispatch/device_histogram_dispatch.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \brief DeviceHistogram provides device-wide parallel operations for constructing histogram(s) from a sequence of samples data residing within global memory. ![](histogram_logo.png)
 * \ingroup DeviceModule
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Histogram"><em>histogram</em></a>
 * counts the number of observations that fall into each of the disjoint categories (known as <em>bins</em>).
 *
 * \par Usage Considerations
 * \cdp_class{DeviceHistogram}
 *
 * \par Performance
 *
 * \image html histo_perf.png
 *
 */
struct DeviceHistogram
{
    /******************************************************************//**
     * \name Single-channel samples
     *********************************************************************/
    //@{


    /**
     * \brief Computes a device-wide histogram using fast block-wide sorting.
     *
     * \par
     * - The total number of samples across all channels (\p num_samples) must be a whole multiple of \p CHANNELS.
     * - Delivers consistent throughput regardless of sample diversity
     * - Histograms having a large number of bins (e.g., thousands) may adversely affect shared memory occupancy and performance (or even the ability to launch).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of a 8-bin histogram of
     * single-channel <tt>unsigned char</tt> samples.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histogram
     * int              num_samples;    // e.g., 12
     * unsigned char    *d_samples;     // e.g., [2, 6, 7, 5, 3, 0, 2, 1, 7, 0, 6, 2]
     * unsigned int     *d_histogram;   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceHistogram::SingleChannelSorting<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histogram
     * cub::DeviceHistogram::SingleChannelSorting<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [2, 1, 3, 1, 0, 1, 2, 2]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        typename            InputIterator,
        typename            HistoCounter>
    CUB_RUNTIME_FUNCTION
    static cudaError_t SingleChannelSorting(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Input samples
        HistoCounter*       d_histogram,                        ///< [out] Array of BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
                DEVICE_HISTO_SORT,
                BINS,
                1,
                1,
                InputIterator,
                HistoCounter,
                Offset>
            DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            &d_histogram,
            num_samples,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide histogram using shared-memory atomic read-modify-write operations.
     *
     * \par
     * - Input samples having lower diversity can cause performance to be degraded due to serializations from bin-collisions.
     * - Histograms having a large number of bins (e.g., thousands) may adversely affect shared memory occupancy and performance (or even the ability to launch).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of a 8-bin histogram of
     * single-channel <tt>unsigned char</tt> samples.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histogram
     * int              num_samples;    // e.g., 12
     * unsigned char    *d_samples;     // e.g., [2, 6, 7, 5, 3, 0, 2, 1, 7, 0, 6, 2]
     * unsigned int     *d_histogram;   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::SingleChannelSorting<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histogram
     * cub::DeviceHistogram::SingleChannelSharedAtomic<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [2, 1, 3, 1, 0, 1, 2, 2]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        typename            InputIterator,
        typename            HistoCounter>
    CUB_RUNTIME_FUNCTION
    static cudaError_t SingleChannelSharedAtomic(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Input samples
        HistoCounter*       d_histogram,                        ///< [out] Array of BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
                DEVICE_HISTO_SHARED_ATOMIC,
                BINS,
                1,
                1,
                InputIterator,
                HistoCounter,
                Offset>
            DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            &d_histogram,
            num_samples,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide histogram using global-memory atomic read-modify-write operations.
     *
     * \par
     * - Input samples having lower diversity can cause performance to be degraded due to serializations from bin-collisions.
     * - Performance is not significantly impacted when computing histograms having large numbers of bins (e.g., thousands).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of a 8-bin histogram of
     * single-channel <tt>unsigned char</tt> samples.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histogram
     * int              num_samples;    // e.g., 12
     * unsigned char    *d_samples;     // e.g., [2, 6, 7, 5, 3, 0, 2, 1, 7, 0, 6, 2]
     * unsigned int     *d_histogram;   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::SingleChannelSorting<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histogram
     * cub::DeviceHistogram::SingleChannelGlobalAtomic<8>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histogram, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [2, 1, 3, 1, 0, 1, 2, 2]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        typename            InputIterator,
        typename            HistoCounter>
    CUB_RUNTIME_FUNCTION
    static cudaError_t SingleChannelGlobalAtomic(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Input samples
        HistoCounter*       d_histogram,                        ///< [out] Array of BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Number of samples to process
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
                DEVICE_HISTO_GLOBAL_ATOMIC,
                BINS,
                1,
                1,
                InputIterator,
                HistoCounter,
                Offset>
            DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            &d_histogram,
            num_samples,
            stream,
            debug_synchronous);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Interleaved multi-channel samples
     *********************************************************************/
    //@{


    /**
     * \brief Computes a device-wide histogram from multi-channel data using fast block-sorting.
     *
     * \par
     * - The total number of samples across all channels (\p num_samples) must be a whole multiple of \p CHANNELS.
     * - Delivers consistent throughput regardless of sample diversity
     * - Histograms having a large number of bins (e.g., thousands) may adversely affect shared memory occupancy and performance (or even the ability to launch).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of three 256-bin histograms from
     * an input sequence of quad-channel (interleaved) <tt>unsigned char</tt> samples.
     * (E.g., RGB histograms from RGBA pixel samples.)
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histograms
     * int           num_samples;     // e.g., 20 (five pixels with four channels each)
     * unsigned char *d_samples;      // e.g., [(2, 6, 7, 5), (3, 0, 2, 1), (7, 0, 6, 2),
     *                                //        (0, 6, 7, 5), (3, 0, 2, 6)]
     * unsigned int  *d_histogram[3]; // e.g., [ [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ] ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::MultiChannelSorting<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::MultiChannelSorting<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [ [1, 0, 1, 2, 0, 0, 0, 1];
     * //                     [0, 3, 0, 0, 0, 0, 2, 0];
     * //                     [0, 0, 2, 0, 0, 0, 1, 2] ]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        int                 CHANNELS,
        int                 ACTIVE_CHANNELS,
        typename            InputIterator,
        typename            HistoCounter>
    CUB_RUNTIME_FUNCTION
    static cudaError_t MultiChannelSorting(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32b pixels where each pixel consists of four RGBA 8b samples).
        HistoCounter        *d_histograms[ACTIVE_CHANNELS],     ///< [out] Array of active channel histogram pointers, each pointing to an output array having BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Total number of samples to process in all channels, including non-active channels
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
            DEVICE_HISTO_SORT,
            BINS,
            CHANNELS,
            ACTIVE_CHANNELS,
            InputIterator,
            HistoCounter,
            Offset> DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histograms,
            num_samples,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide histogram from multi-channel data using shared-memory atomic read-modify-write operations.
     *
     * \par
     * - The total number of samples across all channels (\p num_samples) must be a whole multiple of \p CHANNELS.
     * - Input samples having lower diversity can cause performance to be degraded due to serializations from bin-collisions.
     * - Histograms having a large number of bins (e.g., thousands) may adversely affect shared memory occupancy and performance (or even the ability to launch).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of three 256-bin histograms from
     * an input sequence of quad-channel (interleaved) <tt>unsigned char</tt> samples.
     * (E.g., RGB histograms from RGBA pixel samples.)
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histograms
     * int           num_samples;     // e.g., 20 (five pixels with four channels each)
     * unsigned char *d_samples;      // e.g., [(2, 6, 7, 5), (3, 0, 2, 1), (7, 0, 6, 2),
     *                                //        (0, 6, 7, 5), (3, 0, 2, 6)]
     * unsigned int  *d_histogram[3]; // e.g., [ [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ] ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceHistogram::MultiChannelSharedAtomic<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::MultiChannelSharedAtomic<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [ [1, 0, 1, 2, 0, 0, 0, 1];
     * //                     [0, 3, 0, 0, 0, 0, 2, 0];
     * //                     [0, 0, 2, 0, 0, 0, 1, 2] ]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        int                 CHANNELS,
        int                 ACTIVE_CHANNELS,
        typename            InputIterator,
        typename            HistoCounter>
    CUB_RUNTIME_FUNCTION
    static cudaError_t MultiChannelSharedAtomic(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32b pixels where each pixel consists of four RGBA 8b samples).
        HistoCounter        *d_histograms[ACTIVE_CHANNELS],     ///< [out] Array of active channel histogram pointers, each pointing to an output array having BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Total number of samples to process in all channels, including non-active channels
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
            DEVICE_HISTO_SHARED_ATOMIC,
            BINS,
            CHANNELS,
            ACTIVE_CHANNELS,
            InputIterator,
            HistoCounter,
            Offset> DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histograms,
            num_samples,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide histogram from multi-channel data using global-memory atomic read-modify-write operations.
     *
     * \par
     * - The total number of samples across all channels (\p num_samples) must be a whole multiple of \p CHANNELS.
     * - Input samples having lower diversity can cause performance to be degraded due to serializations from bin-collisions.
     * - Performance is not significantly impacted when computing histograms having large numbers of bins (e.g., thousands).
     * - Performance is often improved when referencing input samples through a texture-caching iterator (e.g., cub::TexObjInputIterator).
     * - \devicestorage
     * - \cdp
     *
     * \par Snippet
     * The code snippet below illustrates the computation of three 256-bin histograms from
     * an input sequence of quad-channel (interleaved) <tt>unsigned char</tt> samples.
     * (E.g., RGB histograms from RGBA pixel samples.)
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and histograms
     * int           num_samples;     // e.g., 20 (five pixels with four channels each)
     * unsigned char *d_samples;      // e.g., [(2, 6, 7, 5), (3, 0, 2, 1), (7, 0, 6, 2),
     *                                //        (0, 6, 7, 5), (3, 0, 2, 6)]
     * unsigned int  *d_histogram[3]; // e.g., [ [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ];
     *                                //         [ ,  ,  ,  ,  ,  ,  ,  ] ]
     * ...
     *
     * // Wrap d_samples device pointer in a random-access texture iterator
     * cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
     * d_samples_tex_itr.BindTexture(d_samples, num_samples * sizeof(unsigned char));
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceHistogram::MultiChannelGlobalAtomic<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Compute histograms
     * cub::DeviceHistogram::MultiChannelGlobalAtomic<8, 4, 3>(d_temp_storage, temp_storage_bytes, d_samples_tex_itr, d_histograms, num_samples);
     *
     * // Unbind texture iterator
     * d_samples_tex_itr.UnbindTexture();
     *
     * // d_histogram   <-- [ [1, 0, 1, 2, 0, 0, 0, 1];
     * //                     [0, 3, 0, 0, 0, 0, 2, 0];
     * //                     [0, 0, 2, 0, 0, 0, 1, 2] ]
     *
     * \endcode
     *
     * \tparam BINS                 Number of histogram bins per channel
     * \tparam CHANNELS             Number of channels interleaved in the input data (may be greater than the number of channels being actively histogrammed)
     * \tparam ACTIVE_CHANNELS      <b>[inferred]</b> Number of channels actively being histogrammed
     * \tparam InputIterator        <b>[inferred]</b> Random-access input iterator type for reading input samples. (Must have an InputIterator::value_type that, when cast as an integer, falls in the range [0..BINS-1])  \iterator
     * \tparam HistoCounter         <b>[inferred]</b> Integer type for counting sample occurrences per histogram bin
     */
    template <
        int                 BINS,
        int                 CHANNELS,
        int                 ACTIVE_CHANNELS,
        typename            InputIterator,
        typename            HistoCounter>
    CUB_RUNTIME_FUNCTION
    static cudaError_t MultiChannelGlobalAtomic(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_samples,                          ///< [in] Pointer to the input sequence of sample items. The samples from different channels are assumed to be interleaved (e.g., an array of 32b pixels where each pixel consists of four RGBA 8b samples).
        HistoCounter        *d_histograms[ACTIVE_CHANNELS],     ///< [out] Array of active channel histogram pointers, each pointing to an output array having BINS counters of integral type \p HistoCounter.
        int                 num_samples,                        ///< [in] Total number of samples to process in all channels, including non-active channels
        cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Dispatch type
        typedef DeviceHistogramDispatch<
                DEVICE_HISTO_GLOBAL_ATOMIC,
                BINS,
                CHANNELS,
                ACTIVE_CHANNELS,
                InputIterator,
                HistoCounter,
                Offset>
            DeviceHistogramDispatch;

        return DeviceHistogramDispatch::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histograms,
            num_samples,
            stream,
            debug_synchronous);
    }

    //@}  end member group

};

/**
 * \example example_device_histogram.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


