#include <unittest/unittest.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/shift.h>

template<typename Vector>
void TestShiftLeftSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data = {1, 2, 3, 4, 5, 6 }

    typename Vector::iterator end = thrust::shift_left(data.begin(),
                                                       data.end(),
                                                       2);

    ASSERT_EQUAL(data.end() - end, 2);

    ASSERT_EQUAL(data[0], 3);
    ASSERT_EQUAL(data[1], 4);
    ASSERT_EQUAL(data[2], 5);
    ASSERT_EQUAL(data[3], 6);
}
DECLARE_VECTOR_UNITTEST(TestShiftLeftSimple);


template<typename ForwardIterator>
ForwardIterator shift_left(my_system &system,
                           ForwardIterator first,
                           ForwardIterator,
                           typename thrust::iterator_traits<ForwardIterator>::difference_type)
{
    system.validate_dispatch();
    return first;
}

void TestShiftLeftDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::shift_left(sys, vec.begin(), vec.end(), 1);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestShiftLeftDispatchExplicit);


template<typename ForwardIterator,
         typename T>
ForwardIterator shift_left(my_tag,
                           ForwardIterator first,
                           ForwardIterator,
                           typename thrust::iterator_traits<ForwardIterator>::difference_type)
{
    *first = 13;
    return first;
}

void TestShiftLeftDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::shift_left(thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       2);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestShiftLeftDispatchImplicit);


template<typename Vector>
void TestShiftRightSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data = {1, 2, 3, 4, 5, 6 }

    typename Vector::iterator begin = thrust::shift_right(data.begin(),
                                                          data.end(),
                                                          2);

    ASSERT_EQUAL(begin - data.begin(), 2);

    ASSERT_EQUAL(data[2], 1);
    ASSERT_EQUAL(data[3], 2);
    ASSERT_EQUAL(data[4], 3);
    ASSERT_EQUAL(data[5], 4);
}
DECLARE_VECTOR_UNITTEST(TestShiftRightSimple);


template<typename ForwardIterator>
ForwardIterator shift_right(my_system &system,
                            ForwardIterator first,
                            ForwardIterator,
                            typename thrust::iterator_traits<ForwardIterator>::difference_type)
{
    system.validate_dispatch();
    return first;
}

void TestShiftRightDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::shift_right(sys, vec.begin(), vec.end(), 1);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestShiftRightDispatchExplicit);


template<typename ForwardIterator,
         typename T>
ForwardIterator shift_right(my_tag,
                            ForwardIterator first,
                            ForwardIterator,
                            typename thrust::iterator_traits<ForwardIterator>::difference_type)
{
    *first = 13;
    return first;
}

void TestShiftRightDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::shift_right(thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       2);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestShiftRightDispatchImplicit);


template<typename T>
void TestShiftLeft(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t h_size = thrust::shift_left(h_data.begin(), h_data.end(), 2) - h_data.begin();
    size_t d_size = thrust::shift_left(d_data.begin(), d_data.end(), 2) - d_data.begin();

    ASSERT_EQUAL(h_size, d_size);

    h_data.resize(h_size);
    d_data.resize(d_size);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestShiftLeft);


template<typename T>
void TestShiftRight(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    auto h_res = thrust::shift_right(h_data.begin(), h_data.end(), 2);
    auto d_res = thrust::shift_right(d_data.begin(), d_data.end(), 2);

    size_t h_size = h_res - h_data.begin();
    size_t d_size = d_res - d_data.begin();

    ASSERT_EQUAL(h_size, d_size);

    h_data.remove(h_data.begin(), h_res);
    d_data.remove(d_data.begin(), d_res);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestShiftRight);


template<typename T>
void TestShiftLeftBoundaries(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::host_vector<T>   reference = h_data;
    thrust::device_vector<T> d_data = h_data;

    auto h_res_zero_shift = thrust::shift_left(h_data.begin(), h_data.end(), 0);
    auto d_res_zero_shift = thrust::shift_left(d_data.begin(), d_data.end(), 0);
    ASSERT_EQUAL(h_res_zero_shift, h_data.end());
    ASSERT_EQUAL(d_res_zero_shift, d_data.end());
    ASSERT_EQUAL(h_data, reference);
    ASSERT_EQUAL(d_data, reference);

    auto h_res_negative_shift = thrust::shift_left(h_data.begin(), h_data.end(), -5);
    auto d_res_negative_shift = thrust::shift_left(d_data.begin(), d_data.end(), -5);
    ASSERT_EQUAL(h_res_zero_shift, h_data.end());
    ASSERT_EQUAL(d_res_zero_shift, d_data.end());
    ASSERT_EQUAL(h_data, reference);
    ASSERT_EQUAL(d_data, reference);

    auto h_res_full_shift = thrust::shift_left(h_data.begin(), h_data.end(), n);
    auto d_res_full_shift = thrust::shift_left(d_data.begin(), d_data.end(), n);
    ASSERT_EQUAL(h_res_zero_shift, h_data.begin());
    ASSERT_EQUAL(d_res_zero_shift, d_data.begin());
    ASSERT_EQUAL(h_data, reference);
    ASSERT_EQUAL(d_data, reference);

    auto h_res_beyond_range_shift = thrust::shift_left(h_data.begin(), h_data.end(), n + 2);
    auto d_res_beyond_range_shift = thrust::shift_left(d_data.begin(), d_data.end(), n + 2);
    ASSERT_EQUAL(h_res_zero_shift, h_data.begin());
    ASSERT_EQUAL(d_res_zero_shift, d_data.begin());
    ASSERT_EQUAL(h_data, reference);
    ASSERT_EQUAL(d_data, reference);
}
DECLARE_VARIABLE_UNITTEST(TestShiftLeftBoundaries);


template<typename T>
void TestShiftRightBoundaries(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::host_vector<T>   reference = h_data;
    thrust::device_vector<T> d_data = h_data;

    auto h_res_zero_shift = thrust::shift_right(h_data.begin(), h_data.end(), 0);
    auto d_res_zero_shift = thrust::shift_right(d_data.begin(), d_data.end(), 0);
    ASSERT_EQUAL(h_res_zero_shift, h_data.begin());
    ASSERT_EQUAL(d_res_zero_shift, d_data.begin());
    ASSERT_EQUAL(h_data, reference);
    ASSERT_EQUAL(d_data, reference);

    auto h_res_negative_shift = thrust::shift_right(h_data.begin(), h_data.end(), -5);
    auto d_res_negative_shift = thrust::shift_right(d_data.begin(), d_data.end(), -5);
    ASSERT_EQUAL(h_res_negative_shift, h_data.begin());
    ASSERT_EQUAL(d_res_negative_shift, d_data.begin());
    ASSERT_EQUAL(h_data, reference);
    ASSERT_EQUAL(d_data, reference);

    auto h_res_full_shift = thrust::shift_right(h_data.begin(), h_data.end(), n);
    auto d_res_full_shift = thrust::shift_right(d_data.begin(), d_data.end(), n);
    ASSERT_EQUAL(h_res_full_shift, h_data.end());
    ASSERT_EQUAL(d_res_full_shift, d_data.end());
    ASSERT_EQUAL(h_data, reference);
    ASSERT_EQUAL(d_data, reference);

    auto h_res_beyond_range_shift = thrust::shift_right(h_data.begin(), h_data.end(), n + 2);
    auto d_res_beyond_range_shift = thrust::shift_right(d_data.begin(), d_data.end(), n + 2);
    ASSERT_EQUAL(h_res_beyond_range_shift, h_data.end());
    ASSERT_EQUAL(d_res_beyond_range_shift, d_data.end());
    ASSERT_EQUAL(h_data, reference);
    ASSERT_EQUAL(d_data, reference);
}
DECLARE_VARIABLE_UNITTEST(TestShiftRightBoundaries);
