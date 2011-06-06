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
  Vector lhs_reference = lhs; \
  Vector reference(lhs.size()); \
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), functor<T>()); \
\
  using namespace thrust::placeholders; \
  Vector result(lhs_reference.size()); \
  thrust::transform(lhs_reference.begin(), lhs_reference.end(), rhs.begin(), result.begin(), _1 reference_operator _2); \
\
  ASSERT_ALMOST_EQUAL(lhs_reference, lhs); \
  ASSERT_ALMOST_EQUAL(reference, result); \
} \
DECLARE_VECTOR_UNITTEST(TestFunctionalPlaceholders##name);

template<typename T>
  struct plus_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs += rhs; }
};

template<typename T>
  struct minus_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs -= rhs; }
};

template<typename T>
  struct multiplies_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs *= rhs; }
};

template<typename T>
  struct divides_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs /= rhs; }
};

template<typename T>
  struct modulus_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs %= rhs; }
};

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(PlusEqual,       +=, plus_equal_reference);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(MinusEqual,      -=, minus_equal_reference);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(MultipliesEqual, *=, multiplies_equal_reference);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(DividesEqual,    /=, divides_equal_reference);

void TestFunctionalPlaceholdersModulusEqualHost(void)
{
  typedef unsigned int T;
  typedef thrust::host_vector<T> Vector;

  Vector lhs = unittest::random_samples<T>(num_samples);
  Vector rhs = unittest::random_samples<T>(num_samples);
  thrust::replace(rhs.begin(), rhs.end(), T(0), T(1));
 
  Vector lhs_reference = lhs;
  Vector reference(lhs.size());
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), modulus_equal_reference<T>());

  using namespace thrust::placeholders;
  Vector result(lhs.size());
  thrust::transform(lhs_reference.begin(), lhs_reference.end(), rhs.begin(), result.begin(), _1 %= _2);

  ASSERT_ALMOST_EQUAL(lhs_reference, lhs);
  ASSERT_ALMOST_EQUAL(reference, result);
}
DECLARE_UNITTEST(TestFunctionalPlaceholdersModulusEqualHost);

void TestFunctionalPlaceholdersModulusEqualDevice(void)
{
  typedef unsigned int T;
  typedef thrust::device_vector<T> Vector;

  Vector lhs = unittest::random_samples<T>(num_samples);
  Vector rhs = unittest::random_samples<T>(num_samples);
  thrust::replace(rhs.begin(), rhs.end(), T(0), T(1));
 
  Vector lhs_reference = lhs;
  Vector reference(lhs.size());
  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), modulus_equal_reference<T>());

  using namespace thrust::placeholders;
  Vector result(lhs.size());
  thrust::transform(lhs_reference.begin(), lhs_reference.end(), rhs.begin(), result.begin(), _1 %= _2);

  ASSERT_ALMOST_EQUAL(lhs_reference, lhs);
  ASSERT_ALMOST_EQUAL(reference, result);
}
DECLARE_UNITTEST(TestFunctionalPlaceholdersModulusEqualDevice);

template<typename T>
  struct bit_and_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs &= rhs; }
};

template<typename T>
  struct bit_or_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs |= rhs; }
};

template<typename T>
  struct bit_xor_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs ^= rhs; }
};

template<typename T>
  struct bit_lshift_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs <<= rhs; }
};

template<typename T>
  struct bit_rshift_equal_reference
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs >>= rhs; }
};

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitAndEqual,    &=,  bit_and_equal_reference);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitOrEqual,     |=,  bit_or_equal_reference);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitXorEqual,    ^=,  bit_xor_equal_reference);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitLshiftEqual, <<=, bit_lshift_equal_reference);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitRshiftEqual, >>=, bit_rshift_equal_reference);

template<typename T>
  struct prefix_increment_reference
{
  __host__ __device__ T& operator()(T &x) const { return ++x; }
};

template<typename T>
  struct suffix_increment_reference
{
  __host__ __device__ T operator()(T &x) const { return x++; }
};

template<typename T>
  struct prefix_decrement_reference
{
  __host__ __device__ T& operator()(T &x) const { return --x; }
};

template<typename T>
  struct suffix_decrement_reference
{
  __host__ __device__ T operator()(T &x) const { return x--; }
};

#define PREFIX_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor) \
template<typename Vector> \
  void TestFunctionalPlaceholdersPrefix##name(void) \
{ \
  typedef typename Vector::value_type T; \
  Vector input = unittest::random_samples<T>(num_samples); \
\
  Vector input_reference = input; \
  Vector reference(input.size()); \
  thrust::transform(input.begin(), input.end(), reference.begin(), functor<T>()); \
\
  using namespace thrust::placeholders; \
  Vector result(input_reference.size()); \
  thrust::transform(input_reference.begin(), input_reference.end(), result.begin(), reference_operator _1); \
\
  ASSERT_ALMOST_EQUAL(input_reference, input); \
  ASSERT_ALMOST_EQUAL(reference, result); \
} \
DECLARE_VECTOR_UNITTEST(TestFunctionalPlaceholdersPrefix##name);

PREFIX_FUNCTIONAL_PLACEHOLDERS_TEST(Increment,  ++,  prefix_increment_reference);
PREFIX_FUNCTIONAL_PLACEHOLDERS_TEST(Decrement,  --,  prefix_decrement_reference);

#define SUFFIX_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor) \
template<typename Vector> \
  void TestFunctionalPlaceholdersSuffix##name(void) \
{ \
  typedef typename Vector::value_type T; \
  Vector input = unittest::random_samples<T>(num_samples); \
\
  Vector input_reference = input; \
  Vector reference(input.size()); \
  thrust::transform(input.begin(), input.end(), reference.begin(), functor<T>()); \
\
  using namespace thrust::placeholders; \
  Vector result(input_reference.size()); \
  thrust::transform(input_reference.begin(), input_reference.end(), result.begin(), _1 reference_operator); \
\
  ASSERT_ALMOST_EQUAL(input_reference, input); \
  ASSERT_ALMOST_EQUAL(reference, result); \
} \
DECLARE_VECTOR_UNITTEST(TestFunctionalPlaceholdersSuffix##name);

SUFFIX_FUNCTIONAL_PLACEHOLDERS_TEST(Increment,  ++,  suffix_increment_reference);
SUFFIX_FUNCTIONAL_PLACEHOLDERS_TEST(Decrement,  --,  suffix_decrement_reference);


