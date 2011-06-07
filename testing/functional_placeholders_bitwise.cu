#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

static const size_t num_samples = 10000;

template<typename Vector, typename U> struct rebind_vector;

template<typename T, typename U>
  struct rebind_vector<thrust::host_vector<T>, U>
{
  typedef thrust::host_vector<U> type;
};

template<typename T, typename U>
  struct rebind_vector<thrust::device_vector<T>, U>
{
  typedef thrust::device_vector<U> type;
};

#define BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor) \
template<typename Vector> \
  void TestFunctionalPlaceholders##name(void) \
{ \
  typedef typename Vector::value_type T; \
  typedef typename rebind_vector<Vector,bool>::type bool_vector; \
  Vector lhs = unittest::random_samples<T>(num_samples); \
  Vector rhs = unittest::random_samples<T>(num_samples); \
\
  bool_vector reference(lhs.size()); \
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), functor<T>()); \
\
  using namespace thrust::placeholders; \
  bool_vector result(lhs.size()); \
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 reference_operator _2); \
\
  ASSERT_EQUAL(reference, result); \
} \
DECLARE_VECTOR_UNITTEST(TestFunctionalPlaceholders##name);

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitAnd, &, thrust::bit_and);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitOr,  |, thrust::bit_or);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitXor, ^, thrust::bit_xor);

template<typename T>
  struct bit_negate_reference
{
  __host__ __device__ T operator()(const T &x) const
  {
    return ~x;
  }
};

template<typename Vector>
  void TestFunctionalPlaceholdersBitNegate(void)
{
  typedef typename Vector::value_type T;
  typedef typename rebind_vector<Vector,bool>::type bool_vector;
  Vector input = unittest::random_samples<T>(num_samples);

  bool_vector reference(input.size());
  thrust::transform(input.begin(), input.end(), reference.begin(), bit_negate_reference<T>());

  using namespace thrust::placeholders;
  bool_vector result(input.size());
  thrust::transform(input.begin(), input.end(), result.begin(), ~_1);

  ASSERT_EQUAL(reference, result);
}
DECLARE_VECTOR_UNITTEST(TestFunctionalPlaceholdersBitNegate);

