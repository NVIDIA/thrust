#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

static const size_t num_samples = 10000;

#define BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor) \
template<typename Vector> \
  void TestFunctionalPlaceholders##name(void) \
{ \
  typedef typename Vector::value_type T; \
  Vector lhs = unittest::random_samples<T>(num_samples); \
  Vector rhs = unittest::random_samples<T>(num_samples); \
  thrust::replace(rhs.begin(), rhs.end(), T(0), T(1)); \
\
  Vector reference(lhs.size()); \
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), functor<T>()); \
\
  using namespace thrust::placeholders; \
  Vector result(lhs.size()); \
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 reference_operator _2); \
\
  ASSERT_ALMOST_EQUAL(reference, result); \
} \
DECLARE_VECTOR_UNITTEST(TestFunctionalPlaceholders##name);

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Plus, +, thrust::plus);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Minus, -, thrust::minus);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Multiplies, *, thrust::multiplies);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Divides, /, thrust::divides);

void TestFunctionalPlaceholdersModulusHost(void)
{
  typedef unsigned int T;
  typedef thrust::host_vector<T> Vector;

  Vector lhs = unittest::random_samples<T>(num_samples);
  Vector rhs = unittest::random_samples<T>(num_samples);
  thrust::replace(rhs.begin(), rhs.end(), T(0), T(1));
 
  Vector reference(lhs.size());
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), thrust::modulus<T>());

  using namespace thrust::placeholders;
  Vector result(lhs.size());
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 % _2);

  ASSERT_ALMOST_EQUAL(reference, result);
}
DECLARE_UNITTEST(TestFunctionalPlaceholdersModulusHost);

void TestFunctionalPlaceholdersModulusDevice(void)
{
  typedef unsigned int T;
  typedef thrust::device_vector<T> Vector;

  Vector lhs = unittest::random_samples<T>(num_samples);
  Vector rhs = unittest::random_samples<T>(num_samples);
  thrust::replace(rhs.begin(), rhs.end(), T(0), T(1));
 
  Vector reference(lhs.size());
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), thrust::modulus<T>());

  using namespace thrust::placeholders;
  Vector result(lhs.size());
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 % _2);

  ASSERT_ALMOST_EQUAL(reference, result);
}
DECLARE_UNITTEST(TestFunctionalPlaceholdersModulusDevice);

#define UNARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor) \
template<typename Vector> \
  void TestFunctionalPlaceholders##name(void) \
{ \
  typedef typename Vector::value_type T; \
  Vector input = unittest::random_samples<T>(num_samples); \
\
  Vector reference(input.size()); \
  thrust::transform(input.begin(), input.end(), reference.begin(), functor<T>()); \
\
  using namespace thrust::placeholders; \
  Vector result(input.size()); \
  thrust::transform(input.begin(), input.end(), result.begin(), reference_operator _1); \
\
  ASSERT_EQUAL(reference, result); \
} \
DECLARE_VECTOR_UNITTEST(TestFunctionalPlaceholders##name);

template<typename T>
  struct unary_plus_reference
{
  __host__ __device__ T operator()(const T &x) const
  {
    return +x;
  }
};

UNARY_FUNCTIONAL_PLACEHOLDERS_TEST(UnaryPlus, +, unary_plus_reference);
UNARY_FUNCTIONAL_PLACEHOLDERS_TEST(Negate,    -, thrust::negate);

