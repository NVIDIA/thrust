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


#include <thrust/iterator/iterator_traits.h>

#include <thrust/host_vector.h>

namespace thrust
{
namespace detail
{
namespace host
{
namespace detail
{

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator merge(InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 first2,
                     InputIterator2 last2,
                     OutputIterator output,
                     StrictWeakOrdering comp)
{
    while(first1 != last1 && first2 != last2)
    {
        if(!comp(*first2, *first1))
        {
            // *first1 <= *first2
            *output = *first1;
            ++first1;
        }
        else
        {
            // *first1 > first2
            *output = *first2;
            ++first2;
        }

        ++output;
    }
    
    while(first1 != last1)
    {
        *output = *first1;
        ++first1;
        ++output;
    }
    
    while(first2 != last2)
    {
        *output = *first2;
        ++first2;
        ++output;
    }

    return output;
}


template <typename RandomAccessIterator,
          typename StrictWeakOrdering>
void inplace_merge(RandomAccessIterator first,
                   RandomAccessIterator middle,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
    typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

    thrust::host_vector<value_type> a( first, middle);
    thrust::host_vector<value_type> b(middle,   last);

    thrust::detail::host::detail::merge(a.begin(), a.end(), b.begin(), b.end(), first, comp);
}


template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 first2,
                 InputIterator2 last2,
                 InputIterator3 first3,
                 InputIterator4 first4,
                 OutputIterator1 output1,
                 OutputIterator2 output2,
                 StrictWeakOrdering comp)
{
    while(first1 != last1 && first2 != last2)
    {
        if(!comp(*first2, *first1))
        {
            // *first1 <= *first2
            *output1 = *first1;
            *output2 = *first3;
            ++first1;
            ++first3;
        }
        else
        {
            // *first1 > first2
            *output1 = *first2;
            *output2 = *first4;
            ++first2;
            ++first4;
        }

        ++output1;
        ++output2;
    }
    
    while(first1 != last1)
    {
        *output1 = *first1;
        *output2 = *first3;
        ++first1;
        ++first3;
        ++output1;
        ++output2;
    }
    
    while(first2 != last2)
    {
        *output1 = *first2;
        *output2 = *first4;
        ++first2;
        ++first4;
        ++output1;
        ++output2;
    }

    return thrust::make_pair(output1, output2);
}


template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
void inplace_merge_by_key(RandomAccessIterator1 first1,
                          RandomAccessIterator1 middle1,
                          RandomAccessIterator1 last1,
                          RandomAccessIterator2 first2,
                          StrictWeakOrdering comp)
{
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
    typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

    RandomAccessIterator2 middle2 = first2 + (middle1 - first1);
    RandomAccessIterator2 last2   = first2 + (last1   - first1);

    thrust::host_vector<value_type1> lhs1( first1, middle1);
    thrust::host_vector<value_type1> rhs1(middle1,   last1);
    thrust::host_vector<value_type2> lhs2( first2, middle2);
    thrust::host_vector<value_type2> rhs2(middle2,   last2);

    thrust::detail::host::detail::merge_by_key
        (lhs1.begin(), lhs1.end(), rhs1.begin(), rhs1.end(),
         lhs2.begin(), rhs2.begin(),
         first1, first2, comp);
}

} // end namespace detail
} // end namespace host
} // end namespace detail
} // end namespace thrust

