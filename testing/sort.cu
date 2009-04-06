#include <komradetest/unittest.h>
#include <komrade/sort.h>
#include <komrade/functional.h>
#include <utility>

template <typename T>
struct less_div_10
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return ((int) lhs) / 10 < ((int) rhs) / 10;}
};

template <typename T>
struct greater_div_10
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return ((int) lhs) / 10 > ((int) rhs) / 10;}
};



template <class Vector>
void InitializeSimpleKeySortTest(Vector& unsorted_keys, Vector& sorted_keys)
{
    unsorted_keys.resize(7);
    unsorted_keys[0] = 1; 
    unsorted_keys[1] = 3; 
    unsorted_keys[2] = 6;
    unsorted_keys[3] = 5;
    unsorted_keys[4] = 2;
    unsorted_keys[5] = 0;
    unsorted_keys[6] = 4;

    sorted_keys.resize(7); 
    sorted_keys[0] = 0; 
    sorted_keys[1] = 1; 
    sorted_keys[2] = 2;
    sorted_keys[3] = 3;
    sorted_keys[4] = 4;
    sorted_keys[5] = 5;
    sorted_keys[6] = 6;
}

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
void InitializeSimpleStableKeySortTest(Vector& unsorted_keys, Vector& sorted_keys)
{
    unsorted_keys.resize(9);   
    unsorted_keys[0] = 25; 
    unsorted_keys[1] = 14; 
    unsorted_keys[2] = 35; 
    unsorted_keys[3] = 16; 
    unsorted_keys[4] = 26; 
    unsorted_keys[5] = 34; 
    unsorted_keys[6] = 36; 
    unsorted_keys[7] = 24; 
    unsorted_keys[8] = 15; 
    
    sorted_keys.resize(9);
    sorted_keys[0] = 14; 
    sorted_keys[1] = 16; 
    sorted_keys[2] = 15; 
    sorted_keys[3] = 25; 
    sorted_keys[4] = 26; 
    sorted_keys[5] = 24; 
    sorted_keys[6] = 35; 
    sorted_keys[7] = 34; 
    sorted_keys[8] = 36; 
}

template <class Vector>
void InitializeSimpleStableKeyValueSortTest(Vector& unsorted_keys, Vector& unsorted_values,
                                            Vector& sorted_keys,   Vector& sorted_values)
{
    unsorted_keys.resize(9);   
    unsorted_values.resize(9);   
    unsorted_keys[0] = 25;   unsorted_values[0] = 0;   
    unsorted_keys[1] = 14;   unsorted_values[1] = 1; 
    unsorted_keys[2] = 35;   unsorted_values[2] = 2; 
    unsorted_keys[3] = 16;   unsorted_values[3] = 3; 
    unsorted_keys[4] = 26;   unsorted_values[4] = 4; 
    unsorted_keys[5] = 34;   unsorted_values[5] = 5; 
    unsorted_keys[6] = 36;   unsorted_values[6] = 6; 
    unsorted_keys[7] = 24;   unsorted_values[7] = 7; 
    unsorted_keys[8] = 15;   unsorted_values[8] = 8; 
    
    sorted_keys.resize(9);
    sorted_values.resize(9);
    sorted_keys[0] = 14;   sorted_values[0] = 1;    
    sorted_keys[1] = 16;   sorted_values[1] = 3; 
    sorted_keys[2] = 15;   sorted_values[2] = 8; 
    sorted_keys[3] = 25;   sorted_values[3] = 0; 
    sorted_keys[4] = 26;   sorted_values[4] = 4; 
    sorted_keys[5] = 24;   sorted_values[5] = 7; 
    sorted_keys[6] = 35;   sorted_values[6] = 2; 
    sorted_keys[7] = 34;   sorted_values[7] = 5; 
    sorted_keys[8] = 36;   sorted_values[8] = 6; 
}

template <class Vector>
void TestSortSimple(void)
{
    typedef typename Vector::value_type T;

    Vector unsorted_keys;
    Vector   sorted_keys;

    InitializeSimpleKeySortTest(unsorted_keys, sorted_keys);

    komrade::sort(unsorted_keys.begin(), unsorted_keys.end());

    ASSERT_EQUAL(unsorted_keys, sorted_keys);
}
DECLARE_VECTOR_UNITTEST(TestSortSimple);


template <class Vector>
void TestSortByKeySimple(void)
{
    Vector unsorted_keys, unsorted_values;
    Vector   sorted_keys,   sorted_values;

    InitializeSimpleKeyValueSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

    komrade::sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin());

    ASSERT_EQUAL(unsorted_keys,   sorted_keys);
    ASSERT_EQUAL(unsorted_values, sorted_values);
}
DECLARE_VECTOR_UNITTEST(TestSortByKeySimple);


template <class Vector>
void TestStableSortSimple(void)
{
    typedef typename Vector::value_type T;

    Vector unsorted_keys;
    Vector   sorted_keys;

    InitializeSimpleStableKeySortTest(unsorted_keys, sorted_keys);

    komrade::stable_sort(unsorted_keys.begin(), unsorted_keys.end(), less_div_10<T>());

    ASSERT_EQUAL(unsorted_keys,   sorted_keys);
}
DECLARE_VECTOR_UNITTEST(TestStableSortSimple);


template <class Vector>
void TestStableSortByKeySimple(void)
{
    typedef typename Vector::value_type T;

    Vector unsorted_keys, unsorted_values;
    Vector   sorted_keys,   sorted_values;

    InitializeSimpleStableKeyValueSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

    komrade::stable_sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin(), less_div_10<T>());

    ASSERT_EQUAL(unsorted_keys,   sorted_keys);
    ASSERT_EQUAL(unsorted_values, sorted_values);
}
DECLARE_VECTOR_UNITTEST(TestStableSortByKeySimple);


template <typename T>
void TestSortAscendingKey(const size_t n)
{
    komrade::host_vector<T>   h_data = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_data = h_data;

    komrade::sort(h_data.begin(), h_data.end(), komrade::less<T>());
    komrade::sort(d_data.begin(), d_data.end(), komrade::less<T>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestSortAscendingKey);

void TestSortDescendingKey(void)
{
    const size_t n = 10027;

    komrade::host_vector<int>   h_data = komradetest::random_integers<int>(n);
    komrade::device_vector<int> d_data = h_data;

    komrade::sort(h_data.begin(), h_data.end(), komrade::greater<int>());
    komrade::sort(d_data.begin(), d_data.end(), komrade::greater<int>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_UNITTEST(TestSortDescendingKey);


template <typename T>
void TestStableSortAscendingKey(const size_t n)
{
    komrade::host_vector<T>   h_data = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_data = h_data;

    komrade::stable_sort(h_data.begin(), h_data.end(), less_div_10<T>());
    komrade::stable_sort(d_data.begin(), d_data.end(), less_div_10<T>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestStableSortAscendingKey);


void TestStableSortDescendingKey(void)
{
    const size_t n = 10027;

    komrade::host_vector<int>   h_data = komradetest::random_integers<int>(n);
    komrade::device_vector<int> d_data = h_data;

    komrade::stable_sort(h_data.begin(), h_data.end(), greater_div_10<int>());
    komrade::stable_sort(d_data.begin(), d_data.end(), greater_div_10<int>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_UNITTEST(TestStableSortDescendingKey);



template <typename T>
void TestSortAscendingKeyValue(const size_t n)
{
    komrade::host_vector<T>   h_keys = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_keys = h_keys;
    
    komrade::host_vector<T>   h_values = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_values = h_values;

    komrade::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), komrade::less<T>());
    komrade::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), komrade::less<T>());

    ASSERT_EQUAL(h_keys,   d_keys);
    ASSERT_EQUAL(h_values, d_values);
}
DECLARE_VARIABLE_UNITTEST(TestSortAscendingKeyValue);


void TestSortDescendingKeyValue(void)
{
    const size_t n = 10027;

    komrade::host_vector<int>   h_keys = komradetest::random_integers<int>(n);
    komrade::device_vector<int> d_keys = h_keys;
    
    komrade::host_vector<int>   h_values = komradetest::random_integers<int>(n);
    komrade::device_vector<int> d_values = h_values;

    komrade::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), komrade::greater<int>());
    komrade::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), komrade::greater<int>());

    ASSERT_EQUAL(h_keys,   d_keys);
    ASSERT_EQUAL(h_values, d_values);
}
DECLARE_UNITTEST(TestSortDescendingKeyValue);


template <typename T>
void TestStableSortAscendingKeyValue(const size_t n)
{
    komrade::host_vector<T>   h_keys = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_keys = h_keys;
    
    komrade::host_vector<T>   h_values = komradetest::random_integers<T>(n);
    komrade::device_vector<T> d_values = h_values;

    komrade::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), less_div_10<T>());
    komrade::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), less_div_10<T>());

    ASSERT_EQUAL(h_keys,   d_keys);
    ASSERT_EQUAL(h_values, d_values);
}
DECLARE_VARIABLE_UNITTEST(TestStableSortAscendingKeyValue);


void TestStableSortDescendingKeyValue(void)
{
    const size_t n = 10027;

    komrade::host_vector<int>   h_keys = komradetest::random_integers<int>(n);
    komrade::device_vector<int> d_keys = h_keys;
    
    komrade::host_vector<int>   h_values = komradetest::random_integers<int>(n);
    komrade::device_vector<int> d_values = h_values;

    komrade::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), greater_div_10<int>());
    komrade::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), greater_div_10<int>());

    ASSERT_EQUAL(h_keys,   d_keys);
    ASSERT_EQUAL(h_values, d_values);
}
DECLARE_UNITTEST(TestStableSortDescendingKeyValue);



template <typename T>
void TestSortAscendingKeyLargeKey(const size_t n)
{
    komradetest::random_integer<T> rnd;

    komrade::host_vector< komradetest::test_pair<T,T> > h_data(n);
    for(size_t i = 0; i < n; i++){
        h_data[i].first  = rnd();
        h_data[i].second = rnd();
    }
    komrade::device_vector< komradetest::test_pair<T,T> > d_data = h_data;


    komrade::sort(h_data.begin(), h_data.end());
    komrade::sort(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestSortAscendingKeyLargeKey);


template <typename T>
void TestSortAscendingKeyValueLargeKey(const size_t n)
{
    komradetest::random_integer<T> rnd;

    komrade::host_vector< komradetest::test_pair<T,T> > h_data(n);
    for(size_t i = 0; i < n; i++){
        h_data[i].first  = rnd();
        h_data[i].second = rnd();
    }
    komrade::device_vector< komradetest::test_pair<T,T> > d_data = h_data;
    
    komrade::host_vector<int>   h_values = komradetest::random_integers<int>(n);
    komrade::device_vector<int> d_values = h_values;

    komrade::sort_by_key(h_data.begin(), h_data.end(), h_values.begin());
    komrade::sort_by_key(d_data.begin(), d_data.end(), d_values.begin());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestSortAscendingKeyValueLargeKey);


template <typename T>
void TestStableSortAscendingKeyLargeKey(const size_t n)
{
    komradetest::random_integer<T> rnd;

    komrade::host_vector< komradetest::test_pair<T,T> > h_data(n);
    for(size_t i = 0; i < n; i++){
        h_data[i].first  = rnd();
        h_data[i].second = rnd();
    }
    komrade::device_vector< komradetest::test_pair<T,T> > d_data = h_data;


    komrade::stable_sort(h_data.begin(), h_data.end());
    komrade::stable_sort(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestStableSortAscendingKeyLargeKey);


template <typename T>
void TestStableSortAscendingKeyValueLargeKey(const size_t n)
{
    komradetest::random_integer<T> rnd;

    komrade::host_vector< komradetest::test_pair<T,T> > h_data(n);
    for(size_t i = 0; i < n; i++){
        h_data[i].first  = rnd();
        h_data[i].second = rnd();
    }
    komrade::device_vector< komradetest::test_pair<T,T> > d_data = h_data;
    
    komrade::host_vector<int>   h_values = komradetest::random_integers<int>(n);
    komrade::device_vector<int> d_values = h_values;

    komrade::stable_sort_by_key(h_data.begin(), h_data.end(), h_values.begin());
    komrade::stable_sort_by_key(d_data.begin(), d_data.end(), d_values.begin());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestStableSortAscendingKeyValueLargeKey);



