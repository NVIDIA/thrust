/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

// do not attempt to compile this file with any other compiler
#ifdef __CUDACC__

#include <limits>

#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/type_traits.h>

#include <thrust/sorting/detail/device/cuda/stable_radix_sort_bits.h>

namespace thrust
{

namespace sorting
{

namespace detail
{

namespace device
{

namespace cuda
{

//////////////////
// 8 BIT TYPES //
//////////////////

template <typename KeyType>
void stable_radix_sort_key_small_dev(KeyType * keys, unsigned int num_elements)
{
    // encode the small types in 32-bit unsigned ints
    thrust::detail::raw_device_buffer<unsigned int> full_keys(num_elements);

    thrust::transform(thrust::device_ptr<KeyType>(keys), 
                      thrust::device_ptr<KeyType>(keys) + num_elements,
                      full_keys.begin(),
                      encode_uint<KeyType>());

    // sort the 32-bit unsigned ints
    stable_radix_sort(full_keys.begin(), full_keys.end());
    
    // decode the 32-bit unsigned ints
    thrust::transform(full_keys.begin(),
                      full_keys.end(),
                      thrust::device_ptr<KeyType>(keys),
                      decode_uint<KeyType>());
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 1>)
{
    stable_radix_sort_key_small_dev(keys, num_elements);
}


//////////////////
// 16 BIT TYPES //
//////////////////

    
template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 2>)
{
    stable_radix_sort_key_small_dev(keys, num_elements);
}


//////////////////
// 32 BIT TYPES //
//////////////////

template <typename KeyType> 
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 4>,
                               thrust::detail::integral_constant<bool, true>,   
                               thrust::detail::integral_constant<bool, false>)  // uint32
{
    radix_sort((unsigned int *) keys, num_elements, encode_uint<KeyType>(), encode_uint<KeyType>());
}

template <typename KeyType> 
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 4>,
                               thrust::detail::integral_constant<bool, true>,
                               thrust::detail::integral_constant<bool, true>)   // int32
{
    // find the smallest value in the array
    KeyType min_val = thrust::reduce(thrust::device_ptr<KeyType>(keys),
                                      thrust::device_ptr<KeyType>(keys + num_elements),
                                      (KeyType) 0,
                                      thrust::minimum<KeyType>());

    if(min_val < 0)
        //negatives present, sort all 32 bits
        radix_sort((unsigned int*) keys, num_elements, encode_uint<KeyType>(), decode_uint<KeyType>(), 32);
    else
        //all keys are positive, treat keys as unsigned ints
        radix_sort((unsigned int *) keys, num_elements, encode_uint<KeyType>(), encode_uint<KeyType>());
}

template <typename KeyType> 
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 4>,
                               thrust::detail::integral_constant<bool, false>,
                               thrust::detail::integral_constant<bool, true>)  // float32
{
    radix_sort((unsigned int*) keys, num_elements, encode_uint<KeyType>(), decode_uint<KeyType>(), 32);
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 4>)
{
    stable_radix_sort_key_dev(keys, num_elements,
                              thrust::detail::integral_constant<int, 4>(),
                              thrust::detail::integral_constant<bool, std::numeric_limits<KeyType>::is_exact>(),
                              thrust::detail::integral_constant<bool, std::numeric_limits<KeyType>::is_signed>());
}

//////////////////
// 64 BIT TYPES //
//////////////////

template <typename KeyType,
          typename LowerBits, typename UpperBits, 
          typename LowerBitsExtractor, typename UpperBitsExtractor>
void stable_radix_sort_key_large_dev(KeyType * keys, unsigned int num_elements,
                                     LowerBitsExtractor extract_lower_bits,
                                     UpperBitsExtractor extract_upper_bits)
{
    // first sort on the lower 32-bits of the keys
    thrust::detail::raw_device_buffer<unsigned int> partial_keys(num_elements);
    thrust::transform(thrust::device_ptr<KeyType>(keys), 
                      thrust::device_ptr<KeyType>(keys) + num_elements,
                      partial_keys.begin(),
                      extract_lower_bits);

    thrust::detail::raw_device_buffer<unsigned int> permutation(num_elements);
    thrust::sequence(permutation.begin(), permutation.end());
    
    stable_radix_sort_by_key((LowerBits *) thrust::raw_pointer_cast(&partial_keys[0]),
                             (LowerBits *) thrust::raw_pointer_cast(&partial_keys[0]) + num_elements,
                             thrust::raw_pointer_cast(&permutation[0]));

    // permute full keys so lower bits are sorted
    thrust::detail::raw_device_buffer<KeyType> permuted_keys(num_elements);
    thrust::gather(permuted_keys.begin(),
                   permuted_keys.end(),
                   permutation.begin(),
                   thrust::device_ptr<KeyType>(keys));
    
    // now sort on the upper 32 bits of the keys
    thrust::transform(permuted_keys.begin(),
                      permuted_keys.end(),
                      partial_keys.begin(),
                      extract_upper_bits);
    thrust::sequence(permutation.begin(), permutation.end());
    
    stable_radix_sort_by_key((UpperBits *) thrust::raw_pointer_cast(&partial_keys[0]),
                             (UpperBits *) thrust::raw_pointer_cast(&partial_keys[0]) + num_elements,
                             thrust::raw_pointer_cast(&permutation[0]));

    // store sorted keys
    thrust::gather(thrust::device_ptr<KeyType>(keys), 
                   thrust::device_ptr<KeyType>(keys) + num_elements,
                   permutation.begin(),
                   permuted_keys.begin());
}

    
template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 8>,
                               thrust::detail::integral_constant<bool, true>,
                               thrust::detail::integral_constant<bool, false>)  // uint64
{
    stable_radix_sort_key_large_dev<KeyType, unsigned int, unsigned int, lower_32_bits<KeyType>, upper_32_bits<KeyType> >
        (keys, num_elements, lower_32_bits<KeyType>(), upper_32_bits<KeyType>());
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 8>,
                               thrust::detail::integral_constant<bool, true>,
                               thrust::detail::integral_constant<bool, true>)   // int64
{
    stable_radix_sort_key_large_dev<KeyType, unsigned int, int, lower_32_bits<KeyType>, upper_32_bits<KeyType> >
        (keys, num_elements, lower_32_bits<KeyType>(), upper_32_bits<KeyType>());
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 8>,
                               thrust::detail::integral_constant<bool, false>,
                               thrust::detail::integral_constant<bool, true>)  // float64
{
    typedef unsigned long long uint64;
    stable_radix_sort_key_large_dev<uint64, unsigned int, unsigned int, lower_32_bits<KeyType>, upper_32_bits<KeyType> >
        (reinterpret_cast<uint64 *>(keys), num_elements, lower_32_bits<KeyType>(), upper_32_bits<KeyType>());
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               thrust::detail::integral_constant<int, 8>)
{
    stable_radix_sort_key_dev(keys, num_elements,
                              thrust::detail::integral_constant<int, 8>(),
                              thrust::detail::integral_constant<bool, std::numeric_limits<KeyType>::is_exact>(),
                              thrust::detail::integral_constant<bool, std::numeric_limits<KeyType>::is_signed>());
}

/////////////////
// Entry Point //
/////////////////

template<typename RandomAccessIterator>
void stable_radix_sort(RandomAccessIterator first,
                       RandomAccessIterator last)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;

    // TODO static_assert< is_pod<KeyType> >

    // RandomAccessIterator should be a trivial iterator
    KeyType * keys = thrust::raw_pointer_cast(&*first);

    // we only handle < 2^32 elements right now
    unsigned int num_elements = last - first;

    // dispatch on sizeof(KeyType)
    stable_radix_sort_key_dev(keys, num_elements, thrust::detail::integral_constant<int, sizeof(KeyType)>());
}


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace sorting

} // end namespace thrust

#endif // __CUDACC__

