#include <unittest/unittest.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/transform.h>

struct basic_arithmetic_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x,
				const thrust::complex<T> &y)
  {
    // exercise unary and binary arithmetic operators
    // Should return approximately 1
    return (+x + +y) + (x * y) / (y * x) + (-y + -x);
  } // end operator()()
}; // end make_pair_functor

struct general_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    // exercise general functions
    // Should return approximately 1
    return thrust::proj( (thrust::polar(abs(x),arg(x)) * conj(x))/norm(x));
  } // end operator()()
}; // end make_pair_functor

struct power_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x,
				const thrust::complex<T> &y)
  {
    // exercise power functions
    return pow(x,y)+sqrt(x);
  } // end operator()()
}; // end make_pair_functor

struct exponential_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    // exercise power functions
    // should return approximately 1
    return log(exp(x))/(T(2.30258509299404568402)*log10(exp(x)));
  } // end operator()()
}; // end make_pair_functor

struct trigonometric_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    // exercise power functions
    // might return approximately 1
    return acos(cos(x))+asin(sin(x))+atan(tan(x))-
      (acosh(cos(x)) + asinh(sinh(x)) + atanh(tanh(x)));
  } // end operator()()
}; // end make_pair_functor


template <typename T>
struct TestComplexTransform
{
  void operator()(const size_t n)
  {
    typedef thrust::complex<T> type;

    thrust::host_vector<type>   h_p1 = unittest::random_samples<T>(n);
    thrust::host_vector<type>   h_p2 = unittest::random_samples<T>(n);
    thrust::host_vector<type>   h_result(n);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_p2 = h_p2;
    thrust::device_vector<type> d_result(n);

    // run basic arithmetic on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), basic_arithmetic_functor());
    // run basic arithmetic on the host
    thrust::transform(d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), basic_arithmetic_functor());
    ASSERT_EQUAL_QUIET(h_result, d_result);

    // run general functions on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), general_functor());
    // run general functions on the host
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), general_functor());
    ASSERT_EQUAL_QUIET(h_result, d_result);

    // run power functions on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), power_functor());
    // run power functions on the host
    thrust::transform(d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), power_functor());
    ASSERT_EQUAL_QUIET(h_result, d_result);


    // run exponential functions on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), exponential_functor());
    // run exponential functions on the host
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), exponential_functor());
    ASSERT_EQUAL_QUIET(h_result, d_result);

    // run trigonometric functions on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), trigonometric_functor());
    // run trigonometric functions on the host
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), trigonometric_functor());
    ASSERT_EQUAL_QUIET(h_result, d_result);
  }
};
VariableUnitTest<TestComplexTransform, FloatingPointTypes> TestComplexTransformInstance;
