#include <unittest/unittest.h>
#include <thrust/find.h>
#include <thrust/iterator/retag.h>


template <typename T>
struct equal_to_value_pred
{
    T value;

    equal_to_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v == value; }
};

template <typename T>
struct not_equal_to_value_pred
{
    T value;

    not_equal_to_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v != value; }
};

template<typename T>
struct less_than_value_pred
{
    T value;

    less_than_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v < value; }
};

template <class Vector>
void TestFindSimple(void)
{
    Vector vec(5);
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;
    vec[3] = 3;
    vec[4] = 5;

    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 0) - vec.begin(), 5);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 1) - vec.begin(), 0);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 2) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 3) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 4) - vec.begin(), 5);
    ASSERT_EQUAL(thrust::find(vec.begin(), vec.end(), 5) - vec.begin(), 4);
}
DECLARE_VECTOR_UNITTEST(TestFindSimple);

template<typename InputIterator, typename T>
InputIterator find(my_system &system, InputIterator first, InputIterator, const T&)
{
    system.validate_dispatch();
    return first;
}

void TestFindDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::find(sys,
                 vec.begin(),
                 vec.end(),
                 0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestFindDispatchExplicit);


template<typename InputIterator, typename T>
InputIterator find(my_tag, InputIterator first, InputIterator, const T&)
{
    *first = 13;
    return first;
}

void TestFindDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::find(thrust::retag<my_tag>(vec.begin()),
                 thrust::retag<my_tag>(vec.end()),
                 0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestFindDispatchImplicit);


template <class Vector>
void TestFindIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;
    vec[3] = 3;
    vec[4] = 5;

    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(0)) - vec.begin(), 5);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(1)) - vec.begin(), 0);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(2)) - vec.begin(), 1);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(3)) - vec.begin(), 2);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(4)) - vec.begin(), 5);
    ASSERT_EQUAL(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(5)) - vec.begin(), 4);
}
DECLARE_VECTOR_UNITTEST(TestFindIfSimple);

template<typename InputIterator, typename Predicate>
InputIterator find_if(my_system &system, InputIterator first, InputIterator, Predicate)
{
    system.validate_dispatch();
    return first;
}

void TestFindIfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::find_if(sys,
                    vec.begin(),
                    vec.end(),
                    thrust::identity<int>());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestFindIfDispatchExplicit);


template<typename InputIterator, typename Predicate>
InputIterator find_if(my_tag, InputIterator first, InputIterator, Predicate)
{
    *first = 13;
    return first;
}

void TestFindIfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::find_if(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.end()),
                    thrust::identity<int>());

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestFindIfDispatchImplicit);


template <class Vector>
void TestFindIfNotSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);
    vec[0] = 0;
    vec[1] = 1;
    vec[2] = 2;
    vec[3] = 3;
    vec[4] = 4;

    ASSERT_EQUAL(0, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(0)) - vec.begin());
    ASSERT_EQUAL(1, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(1)) - vec.begin());
    ASSERT_EQUAL(2, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(2)) - vec.begin());
    ASSERT_EQUAL(3, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(3)) - vec.begin());
    ASSERT_EQUAL(4, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(4)) - vec.begin());
    ASSERT_EQUAL(5, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(5)) - vec.begin());
}
DECLARE_VECTOR_UNITTEST(TestFindIfNotSimple);


template<typename InputIterator, typename Predicate>
InputIterator find_if_not(my_system &system, InputIterator first, InputIterator, Predicate)
{
    system.validate_dispatch();
    return first;
}

void TestFindIfNotDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::find_if_not(sys,
                        vec.begin(),
                        vec.end(),
                        thrust::identity<int>());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestFindIfNotDispatchExplicit);


template<typename InputIterator, typename Predicate>
InputIterator find_if_not(my_tag, InputIterator first, InputIterator, Predicate)
{
    *first = 13;
    return first;
}

void TestFindIfNotDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::find_if_not(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        thrust::identity<int>());

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestFindIfNotDispatchImplicit);


template <typename T>
struct TestFind
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_iter;
        typename thrust::device_vector<T>::iterator d_iter;

        h_iter = thrust::find(h_data.begin(), h_data.end(), T(0));
        d_iter = thrust::find(d_data.begin(), d_data.end(), T(0));
        ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());

        for (size_t i = 1; i < n; i *= 2)
        {
            T sample = h_data[i];
            h_iter = thrust::find(h_data.begin(), h_data.end(), sample);
            d_iter = thrust::find(d_data.begin(), d_data.end(), sample);
            ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());
        }
    }
};
VariableUnitTest<TestFind, SignedIntegralTypes> TestFindInstance;


template <typename T>
struct TestFindIf
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_iter;
        typename thrust::device_vector<T>::iterator d_iter;

        h_iter = thrust::find_if(h_data.begin(), h_data.end(), equal_to_value_pred<T>(0));
        d_iter = thrust::find_if(d_data.begin(), d_data.end(), equal_to_value_pred<T>(0));
        ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());

        for (size_t i = 1; i < n; i *= 2)
        {
            T sample = h_data[i];
            h_iter = thrust::find_if(h_data.begin(), h_data.end(), equal_to_value_pred<T>(sample));
            d_iter = thrust::find_if(d_data.begin(), d_data.end(), equal_to_value_pred<T>(sample));
            ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());
        }
    }
};
VariableUnitTest<TestFindIf, SignedIntegralTypes> TestFindIfInstance;


template <typename T>
struct TestFindIfNot
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_iter;
        typename thrust::device_vector<T>::iterator d_iter;

        h_iter = thrust::find_if_not(h_data.begin(), h_data.end(), not_equal_to_value_pred<T>(0));
        d_iter = thrust::find_if_not(d_data.begin(), d_data.end(), not_equal_to_value_pred<T>(0));
        ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());

        for (size_t i = 1; i < n; i *= 2)
        {
            T sample = h_data[i];
            h_iter = thrust::find_if_not(h_data.begin(), h_data.end(), not_equal_to_value_pred<T>(sample));
            d_iter = thrust::find_if_not(d_data.begin(), d_data.end(), not_equal_to_value_pred<T>(sample));
            ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());
        }
    }
};
VariableUnitTest<TestFindIfNot, SignedIntegralTypes> TestFindIfNotInstance;

