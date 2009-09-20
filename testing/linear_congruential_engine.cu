#include <thrusttest/unittest.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/generate.h>

template<typename Engine, typename Engine::result_type validation>
  struct ValidateEngine
{
  __host__ __device__
  bool operator()(void) const
  {
    Engine e;
    e.discard(9999);

    // get the 10Kth result
    return e() == validation;
  }
}; // end ValidateEngine


void TestMinstdRand0Validation(void)
{
  typedef typename thrust::experimental::random::minstd_rand0 Engine;

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngine<Engine,1043618065>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngine<Engine,1043618065>());

  ASSERT_EQUAL(true, d[0]);
}
DECLARE_UNITTEST(TestMinstdRand0Validation);


void TestMinstdRandValidation(void)
{
  typedef typename thrust::experimental::random::minstd_rand Engine;

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngine<Engine,399268537>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngine<Engine,399268537>());

  ASSERT_EQUAL(true, d[0]);
}
DECLARE_UNITTEST(TestMinstdRandValidation);

