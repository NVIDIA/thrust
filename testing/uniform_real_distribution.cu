#include <thrusttest/unittest.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/generate.h>

template<typename Engine, typename Distribution>
  struct ValidateDistribution
{
  __host__ __device__
  bool operator()(void)
  {
    bool is_valid = true;

    for(unsigned int i = 0;
        i != 10000;
        ++i)
    {
      typename Distribution::result_type f_x = m_d(m_e);
      is_valid &= f_x >= m_d.min();
      is_valid &= f_x <= m_d.max();
    }

    return is_valid;
  }

  Engine m_e;
  Distribution m_d;
}; // end ValidateDistribution

template<typename T>
  struct TestUniformRealDistribution
{
  void operator()(void)
  {
    typedef typename thrust::minstd_rand Engine;
    typedef typename thrust::random::experimental::uniform_real_distribution<float> Distribution;

    Engine eng;
    Distribution dist(-13, 7);

    ValidateDistribution<Engine,Distribution> v = {eng,dist};

    // test host
    thrust::host_vector<bool> h(1);
    thrust::generate(h.begin(), h.end(), v);

    ASSERT_EQUAL(true, h[0]);

    // test device
    thrust::device_vector<bool> d(1);
    thrust::generate(d.begin(), d.end(), v);

    ASSERT_EQUAL(true, d[0]);
  }
};
SimpleUnitTest<TestUniformRealDistribution, FloatTypes> TestUniformRealDistributionInstance;

