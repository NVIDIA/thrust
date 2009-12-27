#include <thrusttest/unittest.h>
#include <thrust/random/linear_feedback_shift_engine.h>
#include <thrust/random/xor_combine_engine.h>
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


template<typename Engine>
  struct ValidateEngineMax
{
  __host__ __device__
  bool operator()(void) const
  {
    bool result = true;

    Engine e;
    for(int i = 0; i < 10000; ++i)
    {
      result &= (e() <= Engine::max);
    }

    return result;
  }
}; // end ValidateEngineMax


template<typename Engine>
  struct ValidateEngineMin
{
  __host__ __device__
  bool operator()(void) const
  {
    bool result = true;

    Engine e;
    for(int i = 0; i < 10000; ++i)
    {
      result &= (e() >= Engine::min);
    }

    return result;
  }
}; // end ValidateEngineMin


void TestTaus88Validation(void)
{
  using namespace thrust::experimental::random;

  typedef linear_feedback_shift_engine<unsigned int, 32, 31, 13, 12> lsf1;
  typedef linear_feedback_shift_engine<unsigned int, 32, 29,  2,  4> lsf2;
  typedef linear_feedback_shift_engine<unsigned int, 32, 28,  3, 17> lsf3;

  typedef xor_combine_engine<
    lsf1,
    0,
    xor_combine_engine<
      lsf2, 0,
      lsf3, 0
    >,
    0
  > taus88;

  typedef taus88 Engine;

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngine<Engine,3535848941>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngine<Engine,3535848941>());

  ASSERT_EQUAL(true, d[0]);
}
DECLARE_UNITTEST(TestTaus88Validation);


void TestTaus88Min(void)
{
  using namespace thrust::experimental::random;

  typedef linear_feedback_shift_engine<unsigned int, 32, 31, 13, 12> lsf1;
  typedef linear_feedback_shift_engine<unsigned int, 32, 29,  2,  4> lsf2;
  typedef linear_feedback_shift_engine<unsigned int, 32, 28,  3, 17> lsf3;

  typedef xor_combine_engine<
    lsf1,
    0,
    xor_combine_engine<
      lsf2, 0,
      lsf3, 0
    >,
    0
  > taus88;

  typedef taus88 Engine;

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngineMin<Engine>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngineMin<Engine>());

  ASSERT_EQUAL(true, d[0]);
}
DECLARE_UNITTEST(TestTaus88Min);


void TestTaus88Max(void)
{
  using namespace thrust::experimental::random;

  typedef linear_feedback_shift_engine<unsigned int, 32, 31, 13, 12> lsf1;
  typedef linear_feedback_shift_engine<unsigned int, 32, 29,  2,  4> lsf2;
  typedef linear_feedback_shift_engine<unsigned int, 32, 28,  3, 17> lsf3;

  typedef xor_combine_engine<
    lsf1,
    0,
    xor_combine_engine<
      lsf2, 0,
      lsf3, 0
    >,
    0
  > taus88;

  typedef taus88 Engine;

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngineMax<Engine>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngineMax<Engine>());

  ASSERT_EQUAL(true, d[0]);
}
DECLARE_UNITTEST(TestTaus88Max);

