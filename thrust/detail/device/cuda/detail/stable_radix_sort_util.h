/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
{

template <class PreProcess, typename T>
struct minmax_transform
{
    PreProcess preprocess;

    minmax_transform(PreProcess _preprocess) : preprocess(_preprocess) {}

    __host__ __device__
    thrust::pair<T,T>
        operator()(T value)
        {
            return thrust::make_pair(preprocess(value), preprocess(value));
        }
};

template <typename T>
struct minmax_reduction
{
    __host__ __device__
    thrust::pair<T,T>
        operator()(thrust::pair<T,T> a, thrust::pair<T,T> b)
        {
            return thrust::make_pair(thrust::min<T>(a.first,  b.first),
                                     thrust::max<T>(a.second, b.second));
        }
};


template <class PreProcess>
thrust::pair<unsigned int, unsigned int> compute_minmax(const unsigned int * keys,
                                                        const unsigned int numElements,
                                                        PreProcess preprocess)
{
    return thrust::transform_reduce(thrust::device_ptr<const unsigned int>(keys),
                                    thrust::device_ptr<const unsigned int>(keys + numElements),
                                    minmax_transform<PreProcess,unsigned int>(preprocess),
                                    thrust::pair<unsigned int,unsigned int>(0xffffffff, 0x00000000),
                                    minmax_reduction<unsigned int>());
}

inline unsigned int compute_keyBits(thrust::pair<unsigned int, unsigned int>& minmax)
{
    unsigned int keyBits = 0;
    unsigned int range = minmax.second - minmax.first;

    while(range)
    {
        keyBits++;
        range >>= 1;
    }

    return keyBits;
}

template <class PreProcess, typename T>
struct modified_preprocess
{
    T min_value;
    PreProcess preprocess;

    modified_preprocess(PreProcess _preprocess, T _min_value)
        : preprocess(_preprocess), min_value(_min_value) {}

    __host__ __device__
    T operator()(T x)
    {
        return preprocess(x) - min_value;
    }
};

template <class PostProcess, typename T>
struct modified_postprocess
{
    T min_value;
    PostProcess postprocess;

    modified_postprocess(PostProcess _postprocess, T _min_value)
        : postprocess(_postprocess), min_value(_min_value) {}

    __host__ __device__
    T operator()(T x)
    {
        return postprocess(x + min_value);
    }
};

} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

