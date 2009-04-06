#include <komradetest/unittest.h>
#include <komrade/uninitialized_fill.h>

template <class Vector>
void TestUninitializedFillPOD(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    T exemplar(7);

    komrade::uninitialized_fill(v.begin() + 1, v.begin() + 4, exemplar);

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], exemplar);
    ASSERT_EQUAL(v[2], exemplar);
    ASSERT_EQUAL(v[3], exemplar);
    ASSERT_EQUAL(v[4], 4);

    exemplar = 8;
    
    komrade::uninitialized_fill(v.begin() + 0, v.begin() + 3, exemplar);
    
    ASSERT_EQUAL(v[0], exemplar);
    ASSERT_EQUAL(v[1], exemplar);
    ASSERT_EQUAL(v[2], exemplar);
    ASSERT_EQUAL(v[3], 7);
    ASSERT_EQUAL(v[4], 4);

    exemplar = 9;
    
    komrade::uninitialized_fill(v.begin() + 2, v.end(), exemplar);
    
    ASSERT_EQUAL(v[0], 8);
    ASSERT_EQUAL(v[1], 8);
    ASSERT_EQUAL(v[2], exemplar);
    ASSERT_EQUAL(v[3], exemplar);
    ASSERT_EQUAL(v[4], 9);

    exemplar = 1;

    komrade::uninitialized_fill(v.begin(), v.end(), exemplar);
    
    ASSERT_EQUAL(v[0], exemplar);
    ASSERT_EQUAL(v[1], exemplar);
    ASSERT_EQUAL(v[2], exemplar);
    ASSERT_EQUAL(v[3], exemplar);
    ASSERT_EQUAL(v[4], exemplar);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedFillPOD);


template<typename T>
struct Wrap
{
  T x;

  Wrap(void){}

  explicit Wrap(T exemplar):x(exemplar){}

  Wrap(const Wrap &exemplar):x(exemplar.x + 13){}
};


template<class Vector>
struct TestUninitializedFillNonPOD
{
  void operator()(const size_t dummy)
  {
    KNOWN_FAILURE
    //typedef komrade::device_vector< Wrap<int> > Vector;
    //typedef typename Vector::value_type T;

    //Vector v(5, T(0));

    //T exemplar(7);

    //// use the copy constructor on the reference
    //T reference(exemplar);

    //komrade::uninitialized_fill(v.begin() + 1, v.begin() + 4, exemplar);

    //ASSERT_EQUAL(v[0], T(0));
    //ASSERT_EQUAL(v[1], reference);
    //ASSERT_EQUAL(v[2], reference);
    //ASSERT_EQUAL(v[3], reference);
    //ASSERT_EQUAL(v[4], T(0));
  }
};
VectorUnitTest<TestUninitializedFillNonPOD, komradetest::type_list<Wrap<int>, Wrap<float> >, komrade::device_vector, komrade::device_allocator> gTestUninitializedFillNonPODDeviceInstance;
VectorUnitTest<TestUninitializedFillNonPOD, komradetest::type_list<Wrap<int>, Wrap<float> >, komrade::host_vector, std::allocator> gTestUninitializedFillNonPODHostInstance;

