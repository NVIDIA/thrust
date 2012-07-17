#include <unittest/unittest.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>


struct my_system : thrust::device_system<my_system> {};

template<typename ForwardIterator>
void sequence(my_system, ForwardIterator first, ForwardIterator)
{
    *first = 13;
}

void TestSequenceDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys;
    thrust::sequence(sys, vec.begin(), vec.end());

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSequenceDispatchExplicit);

void TestSequenceDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::sequence(thrust::retag<my_system>(vec.begin()),
                     thrust::retag<my_system>(vec.end()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSequenceDispatchImplicit);


template <class Vector>
void TestSequenceSimple(void)
{
    typedef typename Vector::value_type T;
    
    Vector v(5);

    thrust::sequence(v.begin(), v.end());

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
    ASSERT_EQUAL(v[3], 3);
    ASSERT_EQUAL(v[4], 4);

    thrust::sequence(v.begin(), v.end(), 10);

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(v[2], 12);
    ASSERT_EQUAL(v[3], 13);
    ASSERT_EQUAL(v[4], 14);
    
    thrust::sequence(v.begin(), v.end(), 10, 2);

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 12);
    ASSERT_EQUAL(v[2], 14);
    ASSERT_EQUAL(v[3], 16);
    ASSERT_EQUAL(v[4], 18);
}
DECLARE_VECTOR_UNITTEST(TestSequenceSimple);


template <typename T>
void TestSequence(size_t n)
{
    thrust::host_vector<T>   h_data(n);
    thrust::device_vector<T> d_data(n);

    thrust::sequence(h_data.begin(), h_data.end());
    thrust::sequence(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10));
    thrust::sequence(d_data.begin(), d_data.end(), T(10));

    ASSERT_EQUAL(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10), T(2));
    thrust::sequence(d_data.begin(), d_data.end(), T(10), T(2));

    ASSERT_EQUAL(h_data, d_data);
    
    thrust::sequence(h_data.begin(), h_data.end(), size_t(10), size_t(2));
    thrust::sequence(d_data.begin(), d_data.end(), size_t(10), size_t(2));

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestSequence);

template <typename T>
void TestSequenceToDiscardIterator(size_t n)
{
    thrust::host_vector<T>   h_data(n);
    thrust::device_vector<T> d_data(n);

    thrust::sequence(thrust::discard_iterator<thrust::device_system_tag>(),
                     thrust::discard_iterator<thrust::device_system_tag>(13),
                     T(10),
                     T(2));

    // nothing to check -- just make sure it compiles
}
DECLARE_VARIABLE_UNITTEST(TestSequenceToDiscardIterator);

