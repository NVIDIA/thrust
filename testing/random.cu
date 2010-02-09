#include <thrusttest/unittest.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/subtract_with_carry_engine.h>
#include <thrust/random/linear_feedback_shift_engine.h>
#include <thrust/random/xor_combine_engine.h>
#include <thrust/generate.h>
#include <sstream>

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


template<typename Engine,
         bool trivial_min = (Engine::min == 0)>
  struct ValidateEngineMin
{
  __host__ __device__
  bool operator()(void) const
  {
    Engine e;

    bool result = true;

    for(int i = 0; i < 10000; ++i)
    {
      result &= (e() >= Engine::min);
    }

    return result;
  }
}; // end ValidateEngineMin

template<typename Engine>
  struct ValidateEngineMin<Engine,true>
{
  __host__ __device__
  bool operator()(void) const
  {
    return true;
  }
};


template<typename Engine>
  struct ValidateEngineMax
{
  __host__ __device__
  bool operator()(void) const
  {
    Engine e;

    bool result = true;

    for(int i = 0; i < 10000; ++i)
    {
      result &= (e() <= Engine::max);
    }

    return result;
  }
}; // end ValidateEngineMax


template<typename Engine, unsigned long long state_10000>
void TestEngineValidation(void)
{
  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngine<Engine,state_10000>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngine<Engine,state_10000>());

  ASSERT_EQUAL(true, d[0]);
}


template<typename Engine>
void TestEngineMax(void)
{
  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngineMax<Engine>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngineMax<Engine>());

  ASSERT_EQUAL(true, d[0]);
}


template<typename Engine>
void TestEngineMin(void)
{
  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngineMin<Engine>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngineMin<Engine>());

  ASSERT_EQUAL(true, d[0]);
}


template<typename Engine>
void TestEngineSaveRestore(void)
{
  // create a default engine
  Engine e0;

  // run it for a while
  e0.discard(10000);

  // save it
  std::stringstream ss;
  ss << e0;

  // run it a while longer
  e0.discard(10000);

  // restore old state
  Engine e1;
  ss >> e1;

  // run e1 a while longer
  e1.discard(10000);

  // both should return the same result

  ASSERT_EQUAL(e0(), e1());
}


void TestRanlux24BaseValidation(void)
{
  typedef typename thrust::experimental::random::ranlux24_base Engine;

  TestEngineValidation<Engine,7937952u>();
}
DECLARE_UNITTEST(TestRanlux24BaseValidation);


void TestRanlux24BaseMin(void)
{
  typedef typename thrust::experimental::random::ranlux24_base Engine;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestRanlux24BaseMin);


void TestRanlux24BaseMax(void)
{
  typedef typename thrust::experimental::random::ranlux24_base Engine;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestRanlux24BaseMax);


void TestRanlux24SaveRestore(void)
{
  typedef typename thrust::experimental::random::ranlux24_base Engine;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestRanlux24SaveRestore);


void TestRanlux48BaseValidation(void)
{
  // XXX nvcc 3.0 complains that the validation number is too large
  KNOWN_FAILURE

//  typedef typename thrust::experimental::random::ranlux24_base Engine;
//
//  TestEngineValidation<Engine,61839128582725ull>();
}
DECLARE_UNITTEST(TestRanlux48BaseValidation);


void TestRanlux48BaseMin(void)
{
  // XXX nvcc 3.0 complains that the validation number is too large
  KNOWN_FAILURE

//  typedef typename thrust::experimental::random::ranlux48_base Engine;
//
//  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestRanlux48BaseMin);


void TestRanlux48BaseMax(void)
{
  // XXX nvcc 3.0 complains that the validation number is too large
  KNOWN_FAILURE

//  typedef typename thrust::experimental::random::ranlux48_base Engine;
//
//  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestRanlux48BaseMax);


void TestMinstdRandValidation(void)
{
  typedef typename thrust::experimental::random::minstd_rand Engine;

  TestEngineValidation<Engine,399268537u>();
}
DECLARE_UNITTEST(TestMinstdRandValidation);


void TestMinstdRandMin(void)
{
  typedef typename thrust::experimental::random::minstd_rand Engine;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandMin);


void TestMinstdRandMax(void)
{
  typedef typename thrust::experimental::random::minstd_rand Engine;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandMax);


void TestMinstdRandSaveRestore(void)
{
  typedef typename thrust::experimental::random::minstd_rand Engine;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandSaveRestore);

void TestMinstdRand0Validation(void)
{
  typedef typename thrust::experimental::random::minstd_rand0 Engine;

  TestEngineValidation<Engine,1043618065u>();
}
DECLARE_UNITTEST(TestMinstdRand0Validation);


void TestMinstdRand0Min(void)
{
  typedef typename thrust::experimental::random::minstd_rand0 Engine;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Min);


void TestMinstdRand0Max(void)
{
  typedef typename thrust::experimental::random::minstd_rand0 Engine;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Max);


void TestMinstdRand0SaveRestore(void)
{
  typedef typename thrust::experimental::random::minstd_rand0 Engine;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0SaveRestore);


void TestTaus88Validation(void)
{
  using namespace thrust::experimental::random;

  typedef linear_feedback_shift_engine<uint32_t, 32u, 31u, 13u, 12u> lsf1;
  typedef linear_feedback_shift_engine<uint32_t, 32u, 29u,  2u,  4u> lsf2;
  typedef linear_feedback_shift_engine<uint32_t, 32u, 28u,  3u, 17u> lsf3;

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

  TestEngineValidation<Engine,3535848941ull>();
}
DECLARE_UNITTEST(TestTaus88Validation);


void TestTaus88Min(void)
{
  using namespace thrust::experimental::random;

  typedef linear_feedback_shift_engine<uint32_t, 32u, 31u, 13u, 12u> lsf1;
  typedef linear_feedback_shift_engine<uint32_t, 32u, 29u,  2u,  4u> lsf2;
  typedef linear_feedback_shift_engine<uint32_t, 32u, 28u,  3u, 17u> lsf3;

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

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestTaus88Min);


void TestTaus88Max(void)
{
  using namespace thrust::experimental::random;

  typedef linear_feedback_shift_engine<uint32_t, 32u, 31u, 13u, 12u> lsf1;
  typedef linear_feedback_shift_engine<uint32_t, 32u, 29u,  2u,  4u> lsf2;
  typedef linear_feedback_shift_engine<uint32_t, 32u, 28u,  3u, 17u> lsf3;

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

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestTaus88Max);


void TestTaus88SaveRestore(void)
{
  KNOWN_FAILURE;

//  using namespace thrust::experimental::random;
//
//  typedef linear_feedback_shift_engine<uint32_t, 32u, 31u, 13u, 12u> lsf1;
//  typedef linear_feedback_shift_engine<uint32_t, 32u, 29u,  2u,  4u> lsf2;
//  typedef linear_feedback_shift_engine<uint32_t, 32u, 28u,  3u, 17u> lsf3;
//
//  typedef xor_combine_engine<
//    lsf1,
//    0,
//    xor_combine_engine<
//      lsf2, 0,
//      lsf3, 0
//    >,
//    0
//  > taus88;
//
//  typedef taus88 Engine;
//
//  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestTaus88SaveRestore);

