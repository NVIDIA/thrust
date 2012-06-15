#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

struct my_tag : thrust::device_system_tag {};

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void sort_by_key(my_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1, RandomAccessIterator2)
{
    *keys_first = 13;
}

void TestSortByKeyDispatch()
{
    thrust::device_vector<int> vec(1);

    thrust::sort_by_key(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSortByKeyDispatch);


template <class Vector>
void InitializeSimpleKeyValueSortTest(Vector& unsorted_keys, Vector& unsorted_values,
                                      Vector& sorted_keys,   Vector& sorted_values)
{
    unsorted_keys.resize(7);   
    unsorted_values.resize(7);   
    unsorted_keys[0] = 1;  unsorted_values[0] = 0;
    unsorted_keys[1] = 3;  unsorted_values[1] = 1;
    unsorted_keys[2] = 6;  unsorted_values[2] = 2;
    unsorted_keys[3] = 5;  unsorted_values[3] = 3;
    unsorted_keys[4] = 2;  unsorted_values[4] = 4;
    unsorted_keys[5] = 0;  unsorted_values[5] = 5;
    unsorted_keys[6] = 4;  unsorted_values[6] = 6;
    
    sorted_keys.resize(7);
    sorted_values.resize(7);
    sorted_keys[0] = 0;  sorted_values[1] = 0;  
    sorted_keys[1] = 1;  sorted_values[3] = 1;  
    sorted_keys[2] = 2;  sorted_values[6] = 2;
    sorted_keys[3] = 3;  sorted_values[5] = 3;
    sorted_keys[4] = 4;  sorted_values[2] = 4;
    sorted_keys[5] = 5;  sorted_values[0] = 5;
    sorted_keys[6] = 6;  sorted_values[4] = 6;
}


template <class Vector>
void TestSortByKeySimple(void)
{
    Vector unsorted_keys, unsorted_values;
    Vector   sorted_keys,   sorted_values;

    InitializeSimpleKeyValueSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

    thrust::sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin());

    ASSERT_EQUAL(unsorted_keys,   sorted_keys);
    ASSERT_EQUAL(unsorted_values, sorted_values);
}
DECLARE_VECTOR_UNITTEST(TestSortByKeySimple);


template <typename T>
void TestSortAscendingKeyValue(const size_t n)
{
    thrust::host_vector<T>   h_keys = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_keys = h_keys;
    
    thrust::host_vector<T>   h_values = h_keys;
    thrust::device_vector<T> d_values = d_keys;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), thrust::less<T>());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), thrust::less<T>());

    ASSERT_EQUAL(h_keys,   d_keys);
    ASSERT_EQUAL(h_values, d_values);
}
DECLARE_VARIABLE_UNITTEST(TestSortAscendingKeyValue);


template <typename T>
void TestSortDescendingKeyValue(const size_t n)
{
    thrust::host_vector<int>   h_keys = unittest::random_integers<int>(n);
    thrust::device_vector<int> d_keys = h_keys;
    
    thrust::host_vector<int>   h_values = h_keys;
    thrust::device_vector<int> d_values = d_keys;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), thrust::greater<int>());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), thrust::greater<int>());

    ASSERT_EQUAL(h_keys,   d_keys);
    ASSERT_EQUAL(h_values, d_values);
}
DECLARE_VARIABLE_UNITTEST(TestSortDescendingKeyValue);


template <class Vector>
void TestSortByKeyUnalignedSimple(void)
{
    typedef typename Vector::value_type T;

    Vector unsorted_keys, unsorted_values;
    Vector   sorted_keys,   sorted_values;

    InitializeSimpleKeyValueSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

    for(int offset = 1; offset < 16; offset++){
        size_t n = unsorted_keys.size() + offset;

        Vector   unaligned_unsorted_keys(n, 0);
        Vector     unaligned_sorted_keys(n, 0);
        Vector unaligned_unsorted_values(n, 0);
        Vector   unaligned_sorted_values(n, 0);
        
        thrust::copy(  unsorted_keys.begin(),   unsorted_keys.end(),   unaligned_unsorted_keys.begin() + offset);
        thrust::copy(    sorted_keys.begin(),     sorted_keys.end(),     unaligned_sorted_keys.begin() + offset);
        thrust::copy(unsorted_values.begin(), unsorted_values.end(), unaligned_unsorted_values.begin() + offset);
        thrust::copy(  sorted_values.begin(),   sorted_values.end(),   unaligned_sorted_values.begin() + offset);
   
        thrust::sort_by_key(unaligned_unsorted_keys.begin() + offset, unaligned_unsorted_keys.end(), unaligned_unsorted_values.begin() + offset);

        ASSERT_EQUAL(  unaligned_unsorted_keys,   unaligned_sorted_keys);
        ASSERT_EQUAL(unaligned_unsorted_values, unaligned_sorted_values);
    }
}
DECLARE_VECTOR_UNITTEST(TestSortByKeyUnalignedSimple);


void TestSortByKeyBool(void)
{
    const size_t n = 10027;

    thrust::host_vector<bool>   h_keys = unittest::random_integers<bool>(n);
    thrust::host_vector<int>    h_values = unittest::random_integers<int>(n);

    thrust::device_vector<bool> d_keys = h_keys;
    thrust::device_vector<int>  d_values = h_values;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    ASSERT_EQUAL(h_keys, d_keys);
    ASSERT_EQUAL(h_values, d_values);
}
DECLARE_UNITTEST(TestSortByKeyBool);


void TestSortByKeyBoolDescending(void)
{
    const size_t n = 10027;

    thrust::host_vector<bool>   h_keys = unittest::random_integers<bool>(n);
    thrust::host_vector<int>    h_values = unittest::random_integers<int>(n);

    thrust::device_vector<bool> d_keys = h_keys;
    thrust::device_vector<int>  d_values = h_values;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), thrust::greater<bool>());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), thrust::greater<bool>());

    ASSERT_EQUAL(h_keys, d_keys);
    ASSERT_EQUAL(h_values, d_values);
}
DECLARE_UNITTEST(TestSortByKeyBoolDescending);


