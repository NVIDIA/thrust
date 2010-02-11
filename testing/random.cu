#include <thrusttest/unittest.h>
#include <thrust/random.h>
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


template<typename Engine>
  struct ValidateEngineEqual
{
  __host__ __device__
  bool operator()(void) const
  {
    bool result = true;

    // test from default constructor
    Engine e0, e1;
    result &= (e0 == e1);

    // advance engines
    e0.discard(10000);
    e1.discard(10000);
    result &= (e0 == e1);

    // test from identical seeds
    Engine e2(13), e3(13);
    result &= (e2 == e3);

    // test different seeds aren't equal
    Engine e4(7), e5(13);
    result &= !(e4 == e5);

    // test reseeding engine to the same seed causes equality
    e4.seed(13);
    result &= (e4 == e5);

    return result;
  }
};


template<typename Engine>
  struct ValidateEngineUnequal
{
  __host__ __device__
  bool operator()(void) const
  {
    bool result = true;

    // test from default constructor
    Engine e0, e1;
    result &= !(e0 != e1);

    // advance engines
    e0.discard(10000);
    e1.discard(10000);
    result &= !(e0 != e1);

    // test from identical seeds
    Engine e2(13), e3(13);
    result &= !(e2 != e3);

    // test different seeds aren't equal
    Engine e4(7), e5(13);
    result &= (e4 != e5);

    // test reseeding engine to the same seed causes equality
    e4.seed(13);
    result &= !(e4 != e5);

    // test different discards causes inequality
    Engine e6(13), e7(13);
    e6.discard(5000);
    e7.discard(10000);
    result &= (e6 != e7);

    return result;
  }
};


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


template<typename Engine>
void TestEngineEqual(void)
{
  ValidateEngineEqual<Engine> f;

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), f);

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), f);

  ASSERT_EQUAL(true, d[0]);
}


template<typename Engine>
void TestEngineUnequal(void)
{
  ValidateEngineUnequal<Engine> f;

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), f);

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), f);

  ASSERT_EQUAL(true, d[0]);
}


void TestRanlux24BaseValidation(void)
{
  typedef thrust::random::ranlux24_base Engine;

  TestEngineValidation<Engine,7937952u>();
}
DECLARE_UNITTEST(TestRanlux24BaseValidation);


void TestRanlux24BaseMin(void)
{
  typedef thrust::random::ranlux24_base Engine;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestRanlux24BaseMin);


void TestRanlux24BaseMax(void)
{
  typedef thrust::random::ranlux24_base Engine;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestRanlux24BaseMax);


void TestRanlux24SaveRestore(void)
{
  typedef thrust::random::ranlux24_base Engine;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestRanlux24SaveRestore);


void TestRanlux24Equal(void)
{
  typedef thrust::random::ranlux24_base Engine;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestRanlux24Equal);


void TestRanlux24Unequal(void)
{
  typedef thrust::random::ranlux24_base Engine;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestRanlux24Unequal);


void TestRanlux48BaseValidation(void)
{
  // XXX nvcc 3.0 complains that the validation number is too large
  KNOWN_FAILURE

//  typedef thrust::random::ranlux24_base Engine;
//
//  TestEngineValidation<Engine,61839128582725ull>();
}
DECLARE_UNITTEST(TestRanlux48BaseValidation);


void TestRanlux48BaseMin(void)
{
  // XXX nvcc 3.0 complains that the shift count is too large
  KNOWN_FAILURE

//  typedef thrust::random::ranlux48_base Engine;
//
//  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestRanlux48BaseMin);


void TestRanlux48BaseMax(void)
{
  // XXX nvcc 3.0 complains that the shift count is too large
  KNOWN_FAILURE

//  typedef thrust::random::ranlux48_base Engine;
//
//  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestRanlux48BaseMax);


void TestRanlux48SaveRestore(void)
{
  // XXX nvcc 3.0 complains that the shift count is too large
  KNOWN_FAILURE

//  typedef thrust::random::ranlux48_base Engine;
//
//  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestRanlux48SaveRestore);


void TestRanlux48Equal(void)
{
  // XXX nvcc 3.0 complains that the shift count is too large
  KNOWN_FAILURE

//  typedef thrust::random::ranlux48_base Engine;
//
//  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestRanlux48Equal);


void TestRanlux48Unequal(void)
{
  // XXX nvcc 3.0 complains that the shift count is too large
  KNOWN_FAILURE

// typedef thrust::random::ranlux48_base Engine;
//
// TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestRanlux48Unequal);


void TestMinstdRandValidation(void)
{
  typedef thrust::random::minstd_rand Engine;

  TestEngineValidation<Engine,399268537u>();
}
DECLARE_UNITTEST(TestMinstdRandValidation);


void TestMinstdRandMin(void)
{
  typedef thrust::random::minstd_rand Engine;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandMin);


void TestMinstdRandMax(void)
{
  typedef thrust::random::minstd_rand Engine;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandMax);


void TestMinstdRandSaveRestore(void)
{
  typedef thrust::random::minstd_rand Engine;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandSaveRestore);


void TestMinstdRandEqual(void)
{
  typedef thrust::random::minstd_rand Engine;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandEqual);


void TestMinstdRandUnequal(void)
{
  typedef thrust::random::minstd_rand Engine;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandUnequal);


void TestMinstdRand0Validation(void)
{
  typedef thrust::random::minstd_rand0 Engine;

  TestEngineValidation<Engine,1043618065u>();
}
DECLARE_UNITTEST(TestMinstdRand0Validation);


void TestMinstdRand0Min(void)
{
  typedef thrust::random::minstd_rand0 Engine;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Min);


void TestMinstdRand0Max(void)
{
  typedef thrust::random::minstd_rand0 Engine;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Max);


void TestMinstdRand0SaveRestore(void)
{
  typedef thrust::random::minstd_rand0 Engine;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0SaveRestore);


void TestMinstdRand0Equal(void)
{
  typedef thrust::random::minstd_rand0 Engine;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Equal);


void TestMinstdRand0Unequal(void)
{
  typedef thrust::random::minstd_rand0 Engine;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Unequal);


void TestTaus88Validation(void)
{
  typedef thrust::random::taus88 Engine;

  TestEngineValidation<Engine,3535848941ull>();
}
DECLARE_UNITTEST(TestTaus88Validation);


void TestTaus88Min(void)
{
  typedef thrust::random::taus88 Engine;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestTaus88Min);


void TestTaus88Max(void)
{
  typedef thrust::random::taus88 Engine;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestTaus88Max);


void TestTaus88SaveRestore(void)
{
  KNOWN_FAILURE;

//  typedef thrust::random::taus88 Engine;
//
//  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestTaus88SaveRestore);


void TestTaus88Equal(void)
{
  typedef thrust::random::taus88 Engine;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestTaus88Equal);


void TestTaus88Unequal(void)
{
  typedef thrust::random::taus88 Engine;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestTaus88Unequal);


