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


#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cstddef>

namespace testing
{
namespace detail
{

// Integer hash functions
template <typename IndexType, typename T>
struct random_integer_functor : public thrust::unary_function<IndexType,T>
{
    size_t seed;

    random_integer_functor(const size_t seed)
        : seed(seed) {}

    // source: http://www.concentric.net/~ttwang/tech/inthash.htm
    __host__ __device__
    T hash(const IndexType i, thrust::detail::false_type) const
    {
        unsigned int h = (unsigned int) i ^ (unsigned int) seed;
        h = ~h + (h << 15);
        h =  h ^ (h >> 12);
        h =  h + (h <<  2);
        h =  h ^ (h >>  4);
        h =  h + (h <<  3) + (h << 11);
        h =  h ^ (h >> 16);
        return T(h);
    }

    __host__ __device__
    T hash(const IndexType i, thrust::detail::true_type) const
    {
        unsigned long long h = (unsigned long long) i ^ (unsigned long long) seed;
        h = ~h + (h << 21);
        h =  h ^ (h >> 24);
        h = (h + (h <<  3)) + (h << 8);
        h =  h ^ (h >> 14);
        h = (h + (h <<  2)) + (h << 4);
        h =  h ^ (h >> 28);
        h =  h + (h << 31);
        return T(h);
    }

    __host__ __device__
    T operator()(const IndexType i) const
    {
        return hash(i, typename thrust::detail::integral_constant<bool, sizeof(IndexType) == 8 || sizeof(T) == 8>::type());
    }
};

template <typename UnsignedInteger, typename Real>
struct integer_to_real : public thrust::unary_function<UnsignedInteger,Real>
{
    __host__ __device__
    Real operator()(const UnsignedInteger i) const
    {
        const Real integer_bound = Real(UnsignedInteger(1) << (4 * sizeof(UnsignedInteger))) * Real(UnsignedInteger(1) << (4 * sizeof(UnsignedInteger)));
        return Real(i) / integer_bound;
    }
};

template <typename T>
struct random_integer_iterator
{
    public:
    typedef           ptrdiff_t                                               IndexType;
    typedef typename thrust::counting_iterator<IndexType>                     CountingIterator;
    typedef          random_integer_functor<IndexType,T>                      Functor;
    typedef typename thrust::transform_iterator<Functor, CountingIterator, T> TransformIterator;

    typedef TransformIterator type;

    static type make(const size_t seed)
    {
        return type(CountingIterator(0), Functor(seed));
    }
};

template <typename T>
struct random_real_iterator
{};

template <>
struct random_real_iterator<float>
{
    typedef random_integer_iterator<unsigned int>::type                RandomIterator;
    typedef integer_to_real<unsigned int, float>                       Functor;
    typedef thrust::transform_iterator<Functor, RandomIterator, float> TransformIterator;
    
    typedef TransformIterator type;

    static type make(const size_t seed)
    {
        return type(random_integer_iterator<unsigned int>::make(seed), Functor());
    }
};

template <>
struct random_real_iterator<double>
{
    typedef random_integer_iterator<unsigned long long>::type           RandomIterator;
    typedef integer_to_real<unsigned long long, double>                 Functor;
    typedef thrust::transform_iterator<Functor, RandomIterator, double> TransformIterator;

    typedef TransformIterator type;

    static type make(const size_t seed)
    {
        return type(random_integer_iterator<unsigned long long>::make(seed), Functor());
    }
};

} // end namespace detail


/////////////////////
// Implicit Ranges //
/////////////////////

template <typename T>
class random_integers
{
  typedef typename detail::random_integer_iterator<T>::type iterator;
  typedef typename thrust::iterator_difference<iterator>    difference_type;
  typedef T value_type;

  protected:
  iterator m_begin;
  iterator m_end;

  public:
  random_integers(const size_t n, const size_t seed = 0)
    : m_begin(testing::detail::random_integer_iterator<T>::make(seed)),
      m_end  (testing::detail::random_integer_iterator<T>::make(seed) + n)
  {}

  iterator begin(void) const { return m_begin; }
  iterator end  (void) const { return m_end;   }

  difference_type size(void) const { return m_end - m_begin; }
};

//template <typename T>
//class random_reals : public cusp::array1d_view<typename detail::random_real_iterator<T>::type>
//{
//    protected:
//    typedef typename detail::random_real_iterator<T>::type Iterator;
//    typedef typename cusp::array1d_view<Iterator>          Parent;
//
//    public:
//    random_reals(const size_t n, const size_t seed = 0)
//        : Parent(detail::random_real_iterator<T>::make(seed), 
//                 detail::random_real_iterator<T>::make(seed) + n)
//    {}
//};

} // end namespace testing

