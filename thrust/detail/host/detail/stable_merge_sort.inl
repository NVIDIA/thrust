/*
 *  Copyright 2008-2010 NVIDIA Corporation
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


#include <algorithm>
#include <thrust/host_vector.h>

#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace host
{
namespace detail
{

//////////////
// Key Sort //
//////////////

template <typename RandomAccessIterator,
          typename StrictWeakOrdering>
void insertion_sort(RandomAccessIterator first,
                    RandomAccessIterator last,
                    StrictWeakOrdering comp)
{
    typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

    if (first == last) return;

    for(RandomAccessIterator i = first + 1; i != last; ++i)
    {
        value_type tmp = *i;

        if (comp(tmp, *first))
        {
            // tmp is the smallest value encountered so far
            std::copy_backward(first, i, i + 1);
            *first = tmp;
        }
        else
        {
            // tmp is not the smallest value, can avoid checking for j == first
            RandomAccessIterator j = i;
            RandomAccessIterator k = i - 1;

            while(comp(tmp, *k))
            {
                *j = *k;
                j = k;
                --k;
            }

            *j = tmp;
        }
    }
}

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
void insertion_sort_by_key(RandomAccessIterator1 first1,
                           RandomAccessIterator1 last1,
                           RandomAccessIterator2 first2,
                           StrictWeakOrdering comp)
{
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
    typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

    if (first1 == last1) return;

    RandomAccessIterator1 i1 = first1 + 1;
    RandomAccessIterator2 i2 = first2 + 1;

    for(; i1 != last1; ++i1, ++i2)
    {
        value_type1 tmp1 = *i1;
        value_type2 tmp2 = *i2;

        if (comp(tmp1, *first1))
        {
            // tmp is the smallest value encountered so far
            std::copy_backward(first1, i1, i1 + 1);
            std::copy_backward(first2, i2, i2 + 1);
            *first1 = tmp1;
            *first2 = tmp2;
        }
        else
        {
            // tmp is not the smallest value, can avoid checking for j == first
            RandomAccessIterator1 j1 = i1;
            RandomAccessIterator1 k1 = i1 - 1;
            
            RandomAccessIterator2 j2 = i2;
            RandomAccessIterator2 k2 = i2 - 1;

            while(comp(tmp1, *k1))
            {
                *j1 = *k1;
                *j2 = *k2;

                j1 = k1;
                j2 = k2;

                --k1;
                --k2;
            }

            *j1 = tmp1;
            *j2 = tmp2;
        }
    }
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

    std::merge(a.begin(), a.end(), b.begin(), b.end(), first, comp);
}

template <typename RandomAccessIterator,
          typename StrictWeakOrdering>
void stable_merge_sort(RandomAccessIterator first,
                       RandomAccessIterator last,
                       StrictWeakOrdering comp)
{
    if (last - first < 32)
    {
        insertion_sort(first, last, comp);
    }
    else
    {
        RandomAccessIterator middle = first + (last - first) / 2;

        stable_merge_sort(first, middle, comp);
        stable_merge_sort(middle,  last, comp);
        inplace_merge(first, middle, last, comp);
    }
}


////////////////////
// Key-Value Sort //
////////////////////

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename RandomAccessIterator3,
          typename RandomAccessIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
void merge_by_key(RandomAccessIterator1 first1,
                  RandomAccessIterator1 last1,
                  RandomAccessIterator2 first2,
                  RandomAccessIterator2 last2,
                  RandomAccessIterator3 first3,
                  RandomAccessIterator4 first4,
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

    // XXX this should really return pair(output1, output2)
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

    merge_by_key(lhs1.begin(), lhs1.end(), rhs1.begin(), rhs1.end(),
                 lhs2.begin(), rhs2.begin(),
                 first1, first2, comp);
}

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
void stable_merge_sort_by_key(RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2,
                              StrictWeakOrdering comp)
{
    if (last1 - first1 <= 32)
    {
        insertion_sort_by_key(first1, last1, first2, comp);
    }
    else
    {
        RandomAccessIterator1 middle1 = first1 + (last1 - first1) / 2;
        RandomAccessIterator2 middle2 = first2 + (last1 - first1) / 2;

        stable_merge_sort_by_key(first1, middle1, first2,  comp);
        stable_merge_sort_by_key(middle1,  last1, middle2, comp);
        inplace_merge_by_key(first1, middle1, last1, first2, comp);
    }
}
    

} // end namespace detail
} // end namespace host
} // end namespace detail
} // end namespace thrust

