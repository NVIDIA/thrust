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


/*! \file sort.inl
 *  \brief Inline file for sort.h
 */

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>

#include <thrust/detail/device/cuda/detail/stable_merge_sort.h>
#include <thrust/detail/device/cuda/detail/stable_radix_sort.h>

#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/raw_buffer.h>

/*
 *  This file implements the following dispatch procedure for cuda::stable_sort()
 *  and cuda::stable_sort_by_key().  All iterators are assumed to be "trivial
 *  iterators" (i.e. pointer wrappers).  The functions
 *      indirect_stable_sort()
 *      indirect_stable_sort_by_key()
 *      permutation_stable_sort_by_key()
 *  are meta sorting functions that convert one sorting problem to another.
 *  Specifically, indirect_sort() converts a sort on T to a sort on integers
 *  that index into an array of type T.  Similarly, permutation_sort_by_key()
 *  converts a (key,value) sort into a (key,index) sort where the indices
 *  record the permutation used to sort the keys.  The permuted indices are
 *  then used to reorder the values.  In either case, the meta sorting functions
 *  are used to convert an ill-suited problem (i.e. sorting with large keys 
 *  or large values) into a problem more amenable to the true sorting algorithms.
 * 
 *   stable_sort() dispatch procedure
 *
 *       Level 1:
 *          if is_primitive<KeyType> && is_equal< StrictWeakOrdering, less<KeyType> > 
 *              stable_radix_sort()
 *          else
 *              Level2 stable_merge_sort()
 *
 *       Level2:
 *          if sizeof(KeyType) > 16
 *               add indirection to keys
 *               stable_merge_sort()
 *               permute keys
 *          else
 *               stable_merge_sort()
 *     
 *   stable_sort_by_key() dispatch procedure
 *
 *       Level 1:
 *          if is_primitive<KeyType> && is_equal< StrictWeakOrdering, less<KeyType> > 
 *              stable_radix_sort_by_key()
 *          else
 *              Level2 stable_merge_sort_by_key()
 *
 *       Level2:
 *          if sizeof(KeyType) > 16
 *               add indirection to keys
 *               Level3 stable_merge_sort_by_key()
 *               permute keys
 *          else
 *               Level3 stable_merge_sort_by_key()
 *
 *       Level3:
 *          if sizeof(ValueType) != 4
 *              add indirection to values
 *              stable_merge_sort_by_key()
 *              permute values
 *          else 
 *              stable_merge_sort_by_key()
 */



namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

namespace second_dispatch
{
    // second level of the dispatch decision tree

    // add one level of indirection to the StrictWeakOrdering comp
    template <typename RandomAccessIterator, typename StrictWeakOrdering> 
    struct indirect_comp
    {
        RandomAccessIterator first;
        StrictWeakOrdering   comp;
    
        indirect_comp(RandomAccessIterator first, StrictWeakOrdering comp)
            : first(first), comp(comp) {}
    
        template <typename IndexKeyTypeype>
        __host__ __device__
        bool operator()(IndexKeyTypeype a, IndexKeyTypeype b)
        {
            return comp(thrust::detail::device::dereference(first, a),
                        thrust::detail::device::dereference(first, b));
        }    
    };

    template<typename RandomAccessIterator,
             typename StrictWeakOrdering>
      void stable_merge_sort(RandomAccessIterator first,
                             RandomAccessIterator last,
                             StrictWeakOrdering comp,
                             thrust::detail::true_type)
    {
        // sizeof(KeyType) is large, use indirection and permute keys
        typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
        thrust::detail::raw_device_buffer<unsigned int> permutation(last - first);
        thrust::sequence(permutation.begin(), permutation.end());
    
        thrust::detail::device::cuda::detail::stable_merge_sort
            (permutation.begin(), permutation.end(), indirect_comp<RandomAccessIterator,StrictWeakOrdering>(first, comp));
    
        thrust::detail::raw_device_buffer<KeyType> temp(first, last);
        thrust::gather(first, last, permutation.begin(), temp.begin());
    }
    
    template<typename RandomAccessIterator,
             typename StrictWeakOrdering>
      void stable_merge_sort(RandomAccessIterator first,
                             RandomAccessIterator last,
                             StrictWeakOrdering comp,
                             thrust::detail::false_type)
    {
        // sizeof(KeyType) is small, use stable_merge_sort() directly
        thrust::detail::device::cuda::detail::stable_merge_sort(first, last, comp);
    }
    
    template<typename RandomAccessIterator1,
             typename RandomAccessIterator2,
             typename StrictWeakOrdering>
    void stable_merge_sort_by_key(RandomAccessIterator1 keys_first,
                                  RandomAccessIterator1 keys_last,
                                  RandomAccessIterator2 values_first,
                                  StrictWeakOrdering comp,
                                  thrust::detail::true_type)
    {
        // sizeof(KeyType) is large, use indirection and permute keys
        typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
        thrust::detail::raw_device_buffer<unsigned int> permutation(keys_last - keys_first);
        thrust::sequence(permutation.begin(), permutation.end());
    
        thrust::detail::device::cuda::detail::stable_merge_sort_by_key
            (permutation.begin(), permutation.end(), values_first, indirect_comp<RandomAccessIterator1,StrictWeakOrdering>(keys_first, comp));
    
        thrust::detail::raw_device_buffer<KeyType> temp(keys_first, keys_last);
        thrust::gather(keys_first, keys_last, permutation.begin(), temp.begin());
    }
    
    template<typename RandomAccessIterator1,
             typename RandomAccessIterator2,
             typename StrictWeakOrdering>
    void stable_merge_sort_by_key(RandomAccessIterator1 keys_first,
                                  RandomAccessIterator1 keys_last,
                                  RandomAccessIterator2 values_first,
                                  StrictWeakOrdering comp,
                                  thrust::detail::false_type)
    {
        // sizeof(KeyType) is small, sort keys directly
        thrust::detail::device::cuda::detail::stable_merge_sort_by_key
            (keys_first, keys_last, values_first, comp);
    }

} // end namespace second_dispatch


namespace first_dispatch
{
    // first level of the dispatch decision tree

    template<typename RandomAccessIterator,
             typename StrictWeakOrdering>
      void stable_sort(RandomAccessIterator first,
                       RandomAccessIterator last,
                       StrictWeakOrdering comp,
                       thrust::detail::true_type)
    {
        // CUDA path for thrust::stable_sort with primitive keys
        // (e.g. int, float, short, etc.) and the default less<T> comparison
        // method is implemented with stable_radix_sort_by_key
        thrust::detail::device::cuda::detail::stable_radix_sort(first, last);
    }
    
    template<typename RandomAccessIterator,
             typename StrictWeakOrdering>
      void stable_sort(RandomAccessIterator first,
                       RandomAccessIterator last,
                       StrictWeakOrdering comp,
                       thrust::detail::false_type)
    {
        // decide whether we should use indirect sort or not
        typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
        static const bool add_key_indirection = sizeof(KeyType) > 16;  
        
        // XXX  magic constant determined by limited empirical testing
        // TODO more extensive tuning, consider vector types (e.g. int4)
    
        // device path for thrust::stable_sort with general keys 
        // and comparison methods is implemented with stable_merge_sort
        thrust::detail::device::cuda::second_dispatch::stable_merge_sort
            (first, last, comp,
                thrust::detail::integral_constant<bool, add_key_indirection>());
    }
    
    template<typename RandomAccessIterator1,
             typename RandomAccessIterator2,
             typename StrictWeakOrdering>
      void stable_sort_by_key(RandomAccessIterator1 keys_first,
                              RandomAccessIterator1 keys_last,
                              RandomAccessIterator2 values_first,
                              StrictWeakOrdering comp,
                              thrust::detail::true_type)
    {
        // device path for thrust::stable_sort_by_key with primitive keys
        // (e.g. int, float, short, etc.) and the default less<T> comparison
        // method is implemented with stable_radix_sort_by_key
        thrust::detail::device::cuda::detail::stable_radix_sort_by_key(keys_first, keys_last, values_first);
    }
    
    template<typename RandomAccessIterator1,
             typename RandomAccessIterator2,
             typename StrictWeakOrdering>
      void stable_sort_by_key(RandomAccessIterator1 keys_first,
                              RandomAccessIterator1 keys_last,
                              RandomAccessIterator2 values_first,
                              StrictWeakOrdering comp,
                              thrust::detail::false_type)
    {
        // decide whether we should use indirect_sort or not
        typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
        static const bool add_key_indirection = sizeof(KeyType) > 16;  
        
        // XXX  magic constant determined by limited empirical testing
        // TODO more extensive tuning, consider vector types (e.g. int4)
    
        // device path for thrust::stable_sort with general keys 
        // and comparison methods is implemented with stable_merge_sort
        thrust::detail::device::cuda::second_dispatch::stable_merge_sort_by_key
            (keys_first, keys_last, values_first, comp,
                thrust::detail::integral_constant<bool, add_key_indirection>());
    }

} // end namespace first_dispatch


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
    // dispatch on whether we can use radix_sort
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
    static const bool use_radix_sort = thrust::detail::is_pod<KeyType>::value &&
                                       thrust::detail::is_same<StrictWeakOrdering, typename thrust::less<KeyType> >::value;

    first_dispatch::stable_sort(first, last, comp,
            thrust::detail::integral_constant<bool, use_radix_sort>());
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
    // dispatch on whether we can use radix_sort
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
    static const bool use_radix_sort = thrust::detail::is_pod<KeyType>::value &&
                                       thrust::detail::is_same<StrictWeakOrdering, typename thrust::less<KeyType> >::value;
    
    first_dispatch::stable_sort_by_key(keys_first, keys_last, values_first, comp,
            thrust::detail::integral_constant<bool, use_radix_sort>());
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

