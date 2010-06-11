#include <unittest/unittest.h>
#include <thrust/range/algorithm/copy.h>
#include <thrust/range/algorithm/sequence.h>

#include <list>

void TestRangeCopyFromLazyFill(void)
{
  KNOWN_FAILURE;

//    typedef int T;
//
//    std::vector<T> v(5);
//    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;
//
//    std::vector<int>::const_iterator begin = v.begin();
//    std::vector<int>::const_iterator end = v.end();
//
//    // copy to host_vector
//    thrust::host_vector<T> h(5, (T) 10);
//    thrust::host_vector<T>::iterator h_result = thrust::copy(begin, end, h.begin());
//    ASSERT_EQUAL(h[0], 0);
//    ASSERT_EQUAL(h[1], 1);
//    ASSERT_EQUAL(h[2], 2);
//    ASSERT_EQUAL(h[3], 3);
//    ASSERT_EQUAL(h[4], 4);
//    ASSERT_EQUAL_QUIET(h_result, h.end());
//
//    // copy to device_vector
//    thrust::device_vector<T> d(5, (T) 10);
//    thrust::device_vector<T>::iterator d_result = thrust::copy(begin, end, d.begin());
//    ASSERT_EQUAL(d[0], 0);
//    ASSERT_EQUAL(d[1], 1);
//    ASSERT_EQUAL(d[2], 2);
//    ASSERT_EQUAL(d[3], 3);
//    ASSERT_EQUAL(d[4], 4);
//    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_UNITTEST(TestRangeCopyFromLazyFill);

template <class Vector>
void TestRangeCopyMatchingTypes(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    using namespace thrust::experimental::range;

    // copy to host_vector
    thrust::host_vector<T> h(5, (T) 10);

    int h_result_size = copy(v, h).size();

    ASSERT_EQUAL(0, h[0]);
    ASSERT_EQUAL(1, h[1]);
    ASSERT_EQUAL(2, h[2]);
    ASSERT_EQUAL(3, h[3]);
    ASSERT_EQUAL(4, h[4]);
    ASSERT_EQUAL_QUIET(0, h_result_size);

    // copy to device_vector
    thrust::device_vector<T> d(5, (T) 10);
    int d_result_size = copy(v, d).size();
    ASSERT_EQUAL(0, d[0]);
    ASSERT_EQUAL(1, d[1]);
    ASSERT_EQUAL(2, d[2]);
    ASSERT_EQUAL(3, d[3]);
    ASSERT_EQUAL(4, d[4]);
    ASSERT_EQUAL_QUIET(0, d_result_size);
}
DECLARE_VECTOR_UNITTEST(TestRangeCopyMatchingTypes);

template <class Vector>
void TestRangeCopyMixedTypes(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    using namespace thrust::experimental::range;

    // copy to host_vector with different type
    thrust::host_vector<float> h(5, (float) 10);
    int h_result_size = copy(v, h).size();

    ASSERT_EQUAL(0, h[0]);
    ASSERT_EQUAL(1, h[1]);
    ASSERT_EQUAL(2, h[2]);
    ASSERT_EQUAL(3, h[3]);
    ASSERT_EQUAL(4, h[4]);
    ASSERT_EQUAL_QUIET(0, h_result_size);

    // copy to device_vector with different type
    thrust::device_vector<float> d(5, (float) 10);
    int d_result_size = copy(v, d).size();
    ASSERT_EQUAL(0, d[0]);
    ASSERT_EQUAL(1, d[1]);
    ASSERT_EQUAL(2, d[2]);
    ASSERT_EQUAL(3, d[3]);
    ASSERT_EQUAL(4, d[4]);
    ASSERT_EQUAL_QUIET(0, d_result_size);
}
DECLARE_VECTOR_UNITTEST(TestRangeCopyMixedTypes);


void TestRangeCopyVectorBool(void)
{
  // XXX nvcc 3.0 freaks out about begin & end used on std::vector
  // because begin & end are __host__ __device__ but std::vector.begin is __host__
  KNOWN_FAILURE;

  //std::vector<bool> v(3);
  //v[0] = true; v[1] = false; v[2] = true;

  //thrust::host_vector<bool> h(3);
  //thrust::device_vector<bool> d(3);

  //using namespace thrust::experimental::range;
  //
  //copy(v, h);
  //copy(v, d);

  //ASSERT_EQUAL(true,  h[0]);
  //ASSERT_EQUAL(false, h[1]);
  //ASSERT_EQUAL(true,  h[2]);

  //ASSERT_EQUAL(true,  d[0]);
  //ASSERT_EQUAL(false, d[1]);
  //ASSERT_EQUAL(true,  d[2]);
}
DECLARE_UNITTEST(TestRangeCopyVectorBool);


template <class Vector>
void TestRangeCopyListTo(void)
{
  // XXX nvcc 3.0 freaks out about begin & end used on std::list
  // because begin & end are __host__ __device__ but std::list.begin is __host__
  KNOWN_FAILURE;

  //typedef typename Vector::value_type T;

  //// copy from list to Vector
  //std::list<T> l;
  //l.push_back(0);
  //l.push_back(1);
  //l.push_back(2);
  //l.push_back(3);
  //l.push_back(4);
  //
  //Vector v(l.size());

  //typename Vector::iterator v_result = thrust::copy(l.begin(), l.end(), v.begin());

  //ASSERT_EQUAL(v[0], 0);
  //ASSERT_EQUAL(v[1], 1);
  //ASSERT_EQUAL(v[2], 2);
  //ASSERT_EQUAL(v[3], 3);
  //ASSERT_EQUAL(v[4], 4);
  //ASSERT_EQUAL_QUIET(v_result, v.end());

  //l.clear();

  //std::back_insert_iterator< std::list<T> > l_result = thrust::copy(v.begin(), v.end(), std::back_insert_iterator< std::list<T> >(l));

  //ASSERT_EQUAL(l.size(), 5);

  //typename std::list<T>::const_iterator iter = l.begin();
  //ASSERT_EQUAL(*iter, 0);  iter++;
  //ASSERT_EQUAL(*iter, 1);  iter++;
  //ASSERT_EQUAL(*iter, 2);  iter++;
  //ASSERT_EQUAL(*iter, 3);  iter++;
  //ASSERT_EQUAL(*iter, 4);  iter++;
}
DECLARE_VECTOR_UNITTEST(TestRangeCopyListTo);


template<typename T>
struct is_even
{
    __host__ __device__
    bool operator()(T x) { return (x & 1) == 0; }
};

template<typename T>
struct is_true
{
    __host__ __device__
    bool operator()(T x) { return x ? true : false; }
};
    

template <class Vector>
void TestRangeCopyIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    Vector dest(3);

    using namespace thrust::experimental::range;

    int result_size = copy_if(v, dest, is_even<T>()).size();

    ASSERT_EQUAL(0, dest[0]);
    ASSERT_EQUAL(2, dest[1]);
    ASSERT_EQUAL(4, dest[2]);
    ASSERT_EQUAL_QUIET(0, result_size);
}
DECLARE_VECTOR_UNITTEST(TestRangeCopyIfSimple);


template <typename T>
void TestRangeCopyIf(const size_t n)
{
    thrust::host_vector<T> h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    is_true<T> pred;

    using namespace thrust::experimental::range;

    int h_leftover_size = copy_if(h_data, h_result, pred).size();
    int d_leftover_size = copy_if(d_data, d_result, pred).size();

    h_result.resize(h_result.size() - h_leftover_size);
    d_result.resize(d_result.size() - d_leftover_size);

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRangeCopyIf);


template <class Vector>
void TestRangeCopyIfStencilSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    Vector s(5);
    s[0] = 1; s[1] = 1; s[2] = 0; s[3] = 1; s[4] = 0;

    Vector dest(3);

    int leftover_size = thrust::experimental::range::copy_if(v, s, dest, is_true<T>()).size();

    ASSERT_EQUAL(0, dest[0]);
    ASSERT_EQUAL(1, dest[1]);
    ASSERT_EQUAL(3, dest[2]);
    ASSERT_EQUAL_QUIET(0, leftover_size);
}
DECLARE_VECTOR_UNITTEST(TestRangeCopyIfStencilSimple);


template <typename T>
void TestRangeCopyIfStencil(const size_t n)
{
    thrust::host_vector<T> h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_stencil = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_stencil = unittest::random_samples<T>(n);

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    is_true<T> pred;

    typename thrust::host_vector<T>::iterator   h_new_end = thrust::experimental::range::copy_if(h_data, h_stencil, h_result, pred).begin();
    typename thrust::device_vector<T>::iterator d_new_end = thrust::experimental::range::copy_if(d_data, d_stencil, d_result, pred).begin();

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRangeCopyIfStencil);

template <typename Vector>
void TestRangeCopySequence(void)
{
    typedef typename Vector::value_type T;

    Vector vec(4);

    using namespace thrust::experimental::range;

    copy(sequence(1,5), vec);

    ASSERT_EQUAL(1, vec[0]);
    ASSERT_EQUAL(2, vec[1]);
    ASSERT_EQUAL(3, vec[2]);
    ASSERT_EQUAL(4, vec[3]);
}
DECLARE_VECTOR_UNITTEST(TestRangeCopySequence);


