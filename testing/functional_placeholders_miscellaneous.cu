#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

template<typename T>
  struct saxpy_reference
{
  __host__ __device__ saxpy_reference(const T &aa)
    : a(aa)
  {}

  __host__ __device__ T operator()(const T &x, const T &y) const
  {
    return a * x + y;
  }

  T a;
};

template<typename Vector>
  void TestFunctionalPlaceholdersValue(void)
{
  const size_t n = 10000;
  typedef typename Vector::value_type T;

  T a(13);

  Vector x = unittest::random_integers<T>(n);
  Vector y = unittest::random_integers<T>(n);
  Vector result(n), reference(n);

  thrust::transform(x.begin(), x.end(), y.begin(), reference.begin(), saxpy_reference<T>(a));

  using namespace thrust::placeholders;
  thrust::transform(x.begin(), x.end(), y.begin(), result.begin(), a * _1 + _2);

  ASSERT_ALMOST_EQUAL(reference, result);
}
DECLARE_VECTOR_UNITTEST(TestFunctionalPlaceholdersValue);

