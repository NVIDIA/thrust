#include <thrusttest/unittest.h>
#include <vector>
#include <list>
#include <limits>

template <class Vector>
void TestVectorCppZeroSize(void)
{
    Vector v;
    ASSERT_EQUAL(v.size(), 0);
    ASSERT_EQUAL((v.begin() == v.end()), true);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppZeroSize);


void TestVectorCppBool(void)
{
    thrust::host_vector<bool> h(3);
    thrust::device_vector<bool> d(3);

    h[0] = true; h[1] = false; h[2] = true;
    d[0] = true; d[1] = false; d[2] = true;

    ASSERT_EQUAL(h[0], true);
    ASSERT_EQUAL(h[1], false);
    ASSERT_EQUAL(h[2], true);

    ASSERT_EQUAL(d[0], true);
    ASSERT_EQUAL(d[1], false);
    ASSERT_EQUAL(d[2], true);
}
DECLARE_UNITTEST(TestVectorCppBool);


template <class Vector>
void TestVectorCppFrontBack(void)
{
    typedef typename Vector::value_type T;

    Vector v(3);
    v[0] = 0; v[1] = 1; v[2] = 2;

    ASSERT_EQUAL(v.front(), 0);
    ASSERT_EQUAL(v.back(),  2);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppFrontBack);


template <class Vector>
void TestVectorCppAssignment(void)
{
    typedef typename Vector::value_type T;

    Vector v(3);

    v[0] = 0; v[1] = 1; v[2] = 2;

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);

    v[0] = 10; v[1] = 11; v[2] = 12;

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(v[2], 12);

    Vector w(3);
    w[0] = v[0];
    w[1] = v[1];
    w[2] = v[2];

    ASSERT_EQUAL(v, w);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppAssignment);


template <class Vector>
void TestVectorCppFromSTLVector(void)
{
    typedef typename Vector::value_type T;

    std::vector<T> stl_vector(3);
    stl_vector[0] = 0;
    stl_vector[1] = 1;
    stl_vector[2] = 2;

    thrust::host_vector<T> v(stl_vector);

    ASSERT_EQUAL(v.size(), 3);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);

    v = stl_vector;
    
    ASSERT_EQUAL(v.size(), 3);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppFromSTLVector);


template <class Vector>
void TestVectorCppFromBiDirectionalIterator(void)
{
    typedef typename Vector::value_type T;
    
    std::list<T> stl_list;
    stl_list.push_back(0);
    stl_list.push_back(1);
    stl_list.push_back(2);

    thrust::host_vector<int> v(stl_list.begin(), stl_list.end());

    ASSERT_EQUAL(v.size(), 3);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppFromBiDirectionalIterator);


template <class Vector>
void TestVectorCppToAndFromHostVector(void)
{
    typedef typename Vector::value_type T;

    thrust::host_vector<T> h(3);
    h[0] = 0;
    h[1] = 1;
    h[2] = 2;

    Vector v(h);

    ASSERT_EQUAL(v, h);

    v = v;

    ASSERT_EQUAL(v, h);

    v[0] = 10;
    v[1] = 11;
    v[2] = 12;

    ASSERT_EQUAL(h[0], 0);  ASSERT_EQUAL(v[0], 10); 
    ASSERT_EQUAL(h[1], 1);  ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(h[2], 2);  ASSERT_EQUAL(v[2], 12);

    h = v;

    ASSERT_EQUAL(v, h);

    h[1] = 11;

    v = h;

    ASSERT_EQUAL(v, h);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppToAndFromHostVector);

template <class Vector>
void TestVectorCppToAndFromDeviceVector(void)
{
    typedef typename Vector::value_type T;

    thrust::device_vector<T> h(3);
    h[0] = 0;
    h[1] = 1;
    h[2] = 2;

    Vector v(h);

    ASSERT_EQUAL(v, h);
    
    v = v;

    ASSERT_EQUAL(v, h);

    v[0] = 10;
    v[1] = 11;
    v[2] = 12;

    ASSERT_EQUAL(h[0], 0);  ASSERT_EQUAL(v[0], 10); 
    ASSERT_EQUAL(h[1], 1);  ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(h[2], 2);  ASSERT_EQUAL(v[2], 12);

    h = v;

    ASSERT_EQUAL(v, h);

    h[1] = 11;

    v = h;

    ASSERT_EQUAL(v, h);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppToAndFromDeviceVector);


template <class Vector>
void TestVectorCppWithInitialValue(void)
{
    typedef typename Vector::value_type T;

    const T init = 17;

    Vector v(3, init);

    ASSERT_EQUAL(v.size(), 3);
    ASSERT_EQUAL(v[0], init);
    ASSERT_EQUAL(v[1], init);
    ASSERT_EQUAL(v[2], init);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppWithInitialValue);


template <class Vector>
void TestVectorCppSwap(void)
{
    typedef typename Vector::value_type T;

    Vector v(3);
    v[0] = 0; v[1] = 1; v[2] = 2;

    Vector u(3);
    u[0] = 10; u[1] = 11; u[2] = 12;

    v.swap(u);

    ASSERT_EQUAL(v[0], 10); ASSERT_EQUAL(u[0], 0);  
    ASSERT_EQUAL(v[1], 11); ASSERT_EQUAL(u[1], 1);
    ASSERT_EQUAL(v[2], 12); ASSERT_EQUAL(u[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppSwap);


template <class Vector>
void TestVectorCppErasePosition(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    v.erase(v.begin() + 2);

    ASSERT_EQUAL(v.size(), 4); 
    ASSERT_EQUAL(v[0], 0); 
    ASSERT_EQUAL(v[1], 1); 
    ASSERT_EQUAL(v[2], 3); 
    ASSERT_EQUAL(v[3], 4); 
    
    v.erase(v.begin() + 0);

    ASSERT_EQUAL(v.size(), 3); 
    ASSERT_EQUAL(v[0], 1); 
    ASSERT_EQUAL(v[1], 3); 
    ASSERT_EQUAL(v[2], 4); 
    
    v.erase(v.begin() + 2);

    ASSERT_EQUAL(v.size(), 2); 
    ASSERT_EQUAL(v[0], 1); 
    ASSERT_EQUAL(v[1], 3); 
    
    v.erase(v.begin() + 1);

    ASSERT_EQUAL(v.size(), 1); 
    ASSERT_EQUAL(v[0], 1); 

    v.erase(v.begin() + 0);

    ASSERT_EQUAL(v.size(), 0); 
}
DECLARE_VECTOR_UNITTEST(TestVectorCppErasePosition);


template <class Vector>
void TestVectorCppEraseRange(void)
{
    typedef typename Vector::value_type T;

    Vector v(6);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4; v[5] = 5;

    v.erase(v.begin() + 1, v.begin() + 3);

    ASSERT_EQUAL(v.size(), 4); 
    ASSERT_EQUAL(v[0], 0); 
    ASSERT_EQUAL(v[1], 3); 
    ASSERT_EQUAL(v[2], 4); 
    ASSERT_EQUAL(v[3], 5); 
    
    v.erase(v.begin() + 2, v.end());

    ASSERT_EQUAL(v.size(), 2); 
    ASSERT_EQUAL(v[0], 0); 
    ASSERT_EQUAL(v[1], 3); 
    
    v.erase(v.begin() + 0, v.begin() + 1);

    ASSERT_EQUAL(v.size(), 1); 
    ASSERT_EQUAL(v[0], 3); 
    
    v.erase(v.begin(), v.end());

    ASSERT_EQUAL(v.size(), 0); 
}
DECLARE_VECTOR_UNITTEST(TestVectorCppEraseRange);


void TestVectorCppEquality(void)
{
    thrust::host_vector<int> h_a(3);
    thrust::host_vector<int> h_b(3);
    thrust::host_vector<int> h_c(3);
    h_a[0] = 0;    h_a[1] = 1;    h_a[2] = 2;
    h_b[0] = 0;    h_b[1] = 1;    h_b[2] = 3;
    h_b[0] = 0;    h_b[1] = 1;

    thrust::device_vector<int> d_a(3);
    thrust::device_vector<int> d_b(3);
    thrust::device_vector<int> d_c(3);
    d_a[0] = 0;    d_a[1] = 1;    d_a[2] = 2;
    d_b[0] = 0;    d_b[1] = 1;    d_b[2] = 3;
    d_b[0] = 0;    d_b[1] = 1;

    ASSERT_EQUAL((h_a == h_a), true); ASSERT_EQUAL((h_a == d_a), true); ASSERT_EQUAL((d_a == h_a), true);  ASSERT_EQUAL((d_a == d_a), true); 
    ASSERT_EQUAL((h_b == h_b), true); ASSERT_EQUAL((h_b == d_b), true); ASSERT_EQUAL((d_b == h_b), true);  ASSERT_EQUAL((d_b == d_b), true);
    ASSERT_EQUAL((h_c == h_c), true); ASSERT_EQUAL((h_c == d_c), true); ASSERT_EQUAL((d_c == h_c), true);  ASSERT_EQUAL((d_c == d_c), true);

    ASSERT_EQUAL((h_a == h_b), false); ASSERT_EQUAL((h_a == d_b), false); ASSERT_EQUAL((d_a == h_b), false); ASSERT_EQUAL((d_a == d_b), false); 
    ASSERT_EQUAL((h_b == h_a), false); ASSERT_EQUAL((h_b == d_a), false); ASSERT_EQUAL((d_b == h_a), false); ASSERT_EQUAL((d_b == d_a), false);
    ASSERT_EQUAL((h_a == h_c), false); ASSERT_EQUAL((h_a == d_c), false); ASSERT_EQUAL((d_a == h_c), false); ASSERT_EQUAL((d_a == d_c), false);
    ASSERT_EQUAL((h_c == h_a), false); ASSERT_EQUAL((h_c == d_a), false); ASSERT_EQUAL((d_c == h_a), false); ASSERT_EQUAL((d_c == d_a), false);
    ASSERT_EQUAL((h_b == h_c), false); ASSERT_EQUAL((h_b == d_c), false); ASSERT_EQUAL((d_b == h_c), false); ASSERT_EQUAL((d_b == d_c), false);
    ASSERT_EQUAL((h_c == h_b), false); ASSERT_EQUAL((h_c == d_b), false); ASSERT_EQUAL((d_c == h_b), false); ASSERT_EQUAL((d_c == d_b), false);
}
DECLARE_UNITTEST(TestVectorCppEquality);


template <class Vector>
void TestVectorCppResizing(void)
{
    typedef typename Vector::value_type T;

    Vector v;

    v.resize(3);

    ASSERT_EQUAL(v.size(), 3);

    v[0] = 0; v[1] = 1; v[2] = 2;

    v.resize(5);

    ASSERT_EQUAL(v.size(), 5);

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);

    v[3] = 3; v[4] = 4;

    v.resize(4);

    ASSERT_EQUAL(v.size(), 4);

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
    ASSERT_EQUAL(v[3], 3);

    v.resize(0);

    ASSERT_EQUAL(v.size(), 0);

    // depending on sizeof(T), we will receive one
    // of two possible exceptions
    try
    {
      v.resize(std::numeric_limits<size_t>::max());
    }
    catch(std::length_error e) {}
    catch(std::bad_alloc e) {}

    ASSERT_EQUAL(v.size(), 0);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppResizing);



template <class Vector>
void TestVectorCppReserving(void)
{
    typedef typename Vector::value_type T;

    Vector v;

    v.reserve(3);

    ASSERT_GEQUAL(v.capacity(), 3);

    size_t old_capacity = v.capacity();

    v.reserve(0);

    ASSERT_EQUAL(v.capacity(), old_capacity);

    try
    {
      v.reserve(std::numeric_limits<size_t>::max());
    }
    catch(std::length_error e) {}
    catch(std::bad_alloc e) {}

    ASSERT_EQUAL(v.capacity(), old_capacity);
}
DECLARE_VECTOR_UNITTEST(TestVectorCppReserving)



template <class Vector>
void TestVectorManipulation(size_t n)
{
    typedef typename Vector::value_type T;

    thrust::host_vector<T> src = thrusttest::random_samples<T>(n);
    ASSERT_EQUAL(src.size(), n);

    // basic initialization
    Vector test0(n);
    Vector test1(n, (T) 3);
    ASSERT_EQUAL(test0.size(), n);
    ASSERT_EQUAL(test1.size(), n);
    ASSERT_EQUAL((test1 == std::vector<T>(n, (T) 3)), true);
   
    // initializing from other vector
    std::vector<T> stl_vector(src.begin(), src.end());
    Vector cpy0 = src;
    Vector cpy1(stl_vector);
    Vector cpy2(stl_vector.begin(), stl_vector.end());
    ASSERT_EQUAL(cpy0, src);
    ASSERT_EQUAL(cpy1, src);
    ASSERT_EQUAL(cpy2, src);

    // resizing
    Vector vec1(src);
    vec1.resize(n + 3);
    ASSERT_EQUAL(vec1.size(), n + 3);
    vec1.resize(n);
    ASSERT_EQUAL(vec1.size(), n);
    ASSERT_EQUAL(vec1, src); 
    
    vec1.resize(n + 20, (T) 11);
    Vector tail(vec1.begin() + n, vec1.end());
    ASSERT_EQUAL( (tail == std::vector<T>(20, (T) 11)), true);

    vec1.resize(0);
    ASSERT_EQUAL(vec1.size(), 0);
    ASSERT_EQUAL(vec1.empty(), true);
    vec1.resize(10);
    ASSERT_EQUAL(vec1.size(), 10);
    vec1.clear();
    ASSERT_EQUAL(vec1.size(), 0);
    vec1.resize(5);
    ASSERT_EQUAL(vec1.size(), 5);

    // appending
    Vector vec2;
    for(size_t i = 0; i < 10; ++i){
        ASSERT_EQUAL(vec2.size(), i);
        vec2.push_back( (T) i );
        ASSERT_EQUAL(vec2.size(), i + 1);
        for(size_t j = 0; j <= i; j++)
            ASSERT_EQUAL(vec2[j],     j);
        ASSERT_EQUAL(vec2.back(), i);
    }

    //TODO test swap, erase(pos), erase(begin, end)
}

template <typename T>
void TestDeviceVectorCppManipulation(size_t n)
{
    TestVectorManipulation< thrust::device_vector<T> >(n);
}
DECLARE_VARIABLE_UNITTEST(TestDeviceVectorCppManipulation);

template <typename T>
void TestHostVectorCppManipulation(size_t n)
{
    TestVectorManipulation< thrust::host_vector<T> >(n);
}
DECLARE_VARIABLE_UNITTEST(TestHostVectorCppManipulation);

