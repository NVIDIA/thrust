#include <komradetest/unittest.h>
#include <komrade/uninitialized_copy.h>

template <class Vector>
void TestUninitializedCopySimplePOD(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    // copy to device_vector
    komrade::host_vector<T> h(5);
    komrade::uninitialized_copy(v.begin(), v.end(), h.begin());
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);

    // copy to device_vector
    komrade::device_vector<T> d(5);
    komrade::uninitialized_copy(v.begin(), v.end(), d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedCopySimplePOD);


template<typename T>
struct Wrap
{
  T x;

  Wrap(void){}

  explicit Wrap(T exemplar):x(exemplar){}

  Wrap(const Wrap &exemplar):x(exemplar.x + 13){}
};

template <class Vector>
struct TestUninitializedCopySimpleNonPOD
{
  void operator()(const size_t dummy)
  {
    KNOWN_FAILURE
  }
};
VectorUnitTest<TestUninitializedCopySimpleNonPOD, komradetest::type_list<Wrap<int>, Wrap<float> >, komrade::device_vector, komrade::device_allocator> gTestUninitializedCopyNonPODSimpleDeviceInstance;
VectorUnitTest<TestUninitializedCopySimpleNonPOD, komradetest::type_list<Wrap<int>, Wrap<float> >, komrade::host_vector, std::allocator> gTestUninitializedCopyNonPODSimpleHostInstance;

