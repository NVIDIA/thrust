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

//////////////////////
// Common Functions //
//////////////////////

template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_small_dev(KeyType * keys, ValueType * values, unsigned int num_elements)
{
    // encode the small types in 32-bit unsigned ints
    thrust::detail::raw_device_buffer<unsigned int> full_keys(num_elements);
    thrust::transform(thrust::device_ptr<KeyType>(keys), 
                      thrust::device_ptr<KeyType>(keys + num_elements),
                      full_keys.begin(),
                      encode_uint<KeyType>());
    
    // sort the 32-bit unsigned ints
    stable_radix_sort_by_key(full_keys.begin(), full_keys.end(), values);
    
    // decode the 32-bit unsigned ints
    thrust::transform(full_keys.begin(),
                      full_keys.end(),
                      thrust::device_ptr<KeyType>(keys),
                      decode_uint<KeyType>());
}


/////////////////
// 8 BIT TYPES //
/////////////////

template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 1>)
{
    stable_radix_sort_key_value_small_dev(keys, values, num_elements);
}

//////////////////
// 16 BIT TYPES //
//////////////////

template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 2>)
{
    stable_radix_sort_key_value_small_dev(keys, values, num_elements);
}


//////////////////
// 32 BIT TYPES //
//////////////////

template <typename KeyType, typename ValueType> 
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 4>,
                                     thrust::detail::integral_constant<int, 4>,
                                     thrust::detail::integral_constant<bool, true>,   
                                     thrust::detail::integral_constant<bool, false>)  // uint32
{
    radix_sort_by_key((unsigned int*) keys, (unsigned int *) values, num_elements, encode_uint<KeyType>(), decode_uint<KeyType>());
}

template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 4>,
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
        // negatives present, sort all 32 bits
        radix_sort_by_key((unsigned int*) keys, (unsigned int*) values, num_elements, encode_uint<KeyType>(), decode_uint<KeyType>(), 32);
    else
        // all keys are positive, treat keys as unsigned ints
        radix_sort_by_key((unsigned int*) keys, (unsigned int*) values, num_elements, encode_uint<KeyType>(), decode_uint<KeyType>());
}

template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 4>,
                                     thrust::detail::integral_constant<int, 4>,
                                     thrust::detail::integral_constant<bool, false>,
                                     thrust::detail::integral_constant<bool, true>)  // float32
{
    // sort all 32 bits
    radix_sort_by_key((unsigned int*) keys, (unsigned int*) values, num_elements, encode_uint<KeyType>(), decode_uint<KeyType>(), 32);
}

template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 4>)
{
    stable_radix_sort_key_value_dev(keys, values, num_elements,
                                    thrust::detail::integral_constant<int, 4>(),
                                    thrust::detail::integral_constant<int, 4>(),
                                    thrust::detail::integral_constant<bool, std::numeric_limits<KeyType>::is_exact>(),
                                    thrust::detail::integral_constant<bool, std::numeric_limits<KeyType>::is_signed>());
}

//////////////////
// 64 BIT TYPES //
//////////////////

template <typename KeyType, typename ValueType,
          typename LowerBits, typename UpperBits, 
          typename LowerBitsExtractor, typename UpperBitsExtractor>
void stable_radix_sort_key_value_large_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
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

    // permute full keys and values so lower bits are sorted
    thrust::detail::raw_device_buffer<KeyType> permuted_keys(num_elements);
    thrust::gather(permuted_keys.begin(),
                   permuted_keys.end(),
                   permutation.begin(),
                   thrust::device_ptr<KeyType>(keys));
    
    thrust::detail::raw_device_buffer<ValueType> permuted_values(num_elements);
    thrust::gather(permuted_values.begin(),
                   permuted_values.end(),
                   permutation.begin(),
                   thrust::device_ptr<ValueType>(values));

    // now sort on the upper 32 bits of the keys
    thrust::transform(permuted_keys.begin(),
                      permuted_keys.end(),
                      partial_keys.begin(),
                      extract_upper_bits);
    thrust::sequence(permutation.begin(), permutation.end());
    
    stable_radix_sort_by_key((UpperBits *) thrust::raw_pointer_cast(&partial_keys[0]),
                             (UpperBits *) thrust::raw_pointer_cast(&partial_keys[0]) + num_elements,
                             thrust::raw_pointer_cast(&permutation[0]));

    // store sorted keys and values
    thrust::gather(thrust::device_ptr<KeyType>(keys), 
                   thrust::device_ptr<KeyType>(keys) + num_elements,
                   permutation.begin(),
                   permuted_keys.begin());
    thrust::gather(thrust::device_ptr<ValueType>(values), 
                   thrust::device_ptr<ValueType>(values) + num_elements,
                   permutation.begin(),
                   permuted_values.begin());
}

    
template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 8>,
                                     thrust::detail::integral_constant<bool, true>,
                                     thrust::detail::integral_constant<bool, false>)  // uint64
{
    stable_radix_sort_key_value_large_dev<KeyType, ValueType, unsigned int, unsigned int, lower_32_bits<KeyType>, upper_32_bits<KeyType> > 
        (keys, values, num_elements, lower_32_bits<KeyType>(), upper_32_bits<KeyType>());
}

template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 8>,
                                     thrust::detail::integral_constant<bool, true>,
                                     thrust::detail::integral_constant<bool, true>)   // int64
{
    stable_radix_sort_key_value_large_dev<KeyType, ValueType, unsigned int, int, lower_32_bits<KeyType>, upper_32_bits<KeyType> > 
        (keys, values, num_elements, lower_32_bits<KeyType>(), upper_32_bits<KeyType>());
}

template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 8>,
                                     thrust::detail::integral_constant<bool, false>,
                                     thrust::detail::integral_constant<bool, true>)  // float64
{
    typedef unsigned long long uint64;
    stable_radix_sort_key_value_large_dev<uint64, ValueType, unsigned int, unsigned int, lower_32_bits<KeyType>, upper_32_bits<KeyType> >
        (reinterpret_cast<uint64 *>(keys), values, num_elements, lower_32_bits<KeyType>(), upper_32_bits<KeyType>());
}

template <typename KeyType, typename ValueType>
void stable_radix_sort_key_value_dev(KeyType * keys, ValueType * values, unsigned int num_elements,
                                     thrust::detail::integral_constant<int, 8>)
{
    stable_radix_sort_key_value_dev(keys, values, num_elements,
                                    thrust::detail::integral_constant<int, 8>(),
                                    thrust::detail::integral_constant<bool, std::numeric_limits<KeyType>::is_exact>(),
                                    thrust::detail::integral_constant<bool, std::numeric_limits<KeyType>::is_signed>());
}



////////////////////////////
// ValueIterator Dispatch //
////////////////////////////

template <typename KeyType, typename ValueIterator>
void stable_radix_sort_key_value_dev_native_values(KeyType * keys, ValueIterator values, unsigned int num_elements, thrust::detail::true_type)
{
    // we can safely cast ValueIterator to unsigned int *
    stable_radix_sort_key_value_dev(keys, thrust::raw_pointer_cast(&*values), num_elements, thrust::detail::integral_constant<int, sizeof(KeyType)>());
}

template <typename KeyType, typename ValueIterator>
void stable_radix_sort_key_value_dev_native_values(KeyType * keys, ValueIterator values, unsigned int num_elements, thrust::detail::false_type)
{
    typedef typename thrust::iterator_traits<ValueIterator>::value_type ValueType;

    // Sort with integer values and permute the real values accordingly
    thrust::detail::raw_device_buffer<unsigned int> permutation(num_elements);
    thrust::sequence(permutation.begin(), permutation.end());

    stable_radix_sort_key_value_dev_native_values(keys, permutation.begin(), num_elements, thrust::detail::true_type());
    
    // copy values into temp vector and then permute
    thrust::detail::raw_device_buffer<ValueType> temp_values(values, values + num_elements);
    
    thrust::gather(values, values + num_elements, permutation.begin(), temp_values.begin());
}


/////////////////
// Entry Point //
/////////////////

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void stable_radix_sort_by_key(RandomAccessIterator1 keys_first,
                              RandomAccessIterator1 keys_last,
                              RandomAccessIterator2 values_first)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
    typedef typename thrust::iterator_traits<RandomAccessIterator2>::value_type ValueType;

    // TODO static_assert< is_pod<KeyType> >

    // RandomAccessIterator should be a trivial iterator
    KeyType * keys = thrust::raw_pointer_cast(&*keys_first);

    // we only handle < 2^32 elements right now
    unsigned int num_elements = keys_last - keys_first;
    
    // radix_sort natively sorts uint32 values 
    static const bool native_values = thrust::detail::is_trivial_iterator<RandomAccessIterator2>::value &&
                                      thrust::detail::is_pod<ValueType>::value &&
                                      sizeof(ValueType) == 4;

    stable_radix_sort_key_value_dev_native_values(keys, values_first, num_elements,
                                                  thrust::detail::integral_constant<bool, native_values>());
}

} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace sorting

} // end namespace thrust

#endif // __CUDACC__

