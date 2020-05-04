#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <unittest/unittest.h>
#include <map>

template <typename Vector>
void TestShuffleSimple() {
  Vector data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 2;
  data[3] = 3;
  data[4] = 4;
  Vector shuffled(data.begin(), data.end());
  thrust::default_random_engine g(2);
  thrust::shuffle(shuffled.begin(), shuffled.end(), g);
  thrust::sort(shuffled.begin(), shuffled.end());
  // Check all of our data is present
  // This only tests for strange conditions like duplicated elements
  ASSERT_EQUAL(shuffled, data);
}
DECLARE_VECTOR_UNITTEST(TestShuffleSimple);

template <typename Vector>
void TestShuffleCopySimple() {
  Vector data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 2;
  data[3] = 3;
  data[4] = 4;
  Vector shuffled(5);
  thrust::default_random_engine g(2);
  thrust::shuffle_copy(data.begin(), data.end(), shuffled.begin(), g);
  g.seed(2);
  thrust::shuffle(data.begin(), data.end(), g);
  ASSERT_EQUAL(shuffled, data);
}
DECLARE_VECTOR_UNITTEST(TestShuffleCopySimple);

template <typename T>
void TestHostDeviceIdentical(size_t m) {
  thrust::host_vector<T> host_result(m);
  thrust::host_vector<T> device_result(m);
  thrust::sequence(host_result.begin(), host_result.end(), 0llu);
  thrust::sequence(device_result.begin(), device_result.end(), 0llu);

  thrust::default_random_engine host_g(183);
  thrust::default_random_engine device_g(183);

  thrust::shuffle(host_result.begin(), host_result.end(), host_g);
  thrust::shuffle(device_result.begin(), device_result.end(), device_g);

  ASSERT_EQUAL(device_result, host_result);
}
DECLARE_VARIABLE_UNITTEST(TestHostDeviceIdentical);

// Individual input keys should be permuted to output locations with uniform
// probability. Perform chi-squared test with confidence 99.9%.
template <typename Vector>
void TestShuffleKeyPosition() {
  typedef typename Vector::value_type T;
  size_t m = 20;
  size_t num_samples = 100;
  thrust::host_vector<size_t> index_sum(m, 0);
  thrust::host_vector<T> sequence(m);
  thrust::sequence(sequence.begin(), sequence.end(), T(0));

  for (size_t i = 0; i < num_samples; i++) {
    Vector shuffled(sequence.begin(), sequence.end());
    thrust::default_random_engine g(i);
    thrust::shuffle(shuffled.begin(), shuffled.end(), g);
    thrust::host_vector<T> tmp(shuffled.begin(), shuffled.end());

    for (auto j = 0ull; j < m; j++) {
      index_sum[tmp[j]] += j;
    }
  }
  double expected_average_position = static_cast<double>(m - 1) / 2;
  double chi_squared = 0.0;
  for (auto j = 0ull; j < m; j++) {
    double average_position = static_cast<double>(index_sum[j]) / num_samples;
    chi_squared += std::pow(expected_average_position - average_position, 2) /
                   expected_average_position;
  }
  // Tabulated chi-squared critical value for m-1=19 degrees of freedom
  // and 99.9% confidence
  double confidence_threshold = 43.82;
  ASSERT_LESS(chi_squared, confidence_threshold);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestShuffleKeyPosition);

struct vector_compare {
  template <typename VectorT>
  bool operator()(const VectorT& a, const VectorT& b) const {
    for (auto i = 0ull; i < a.size(); i++) {
      if (a[i] < b[i]) return true;
      if (a[i] > b[i]) return false;
    }
    return false;
  }
};

// Brute force check permutations are uniformly distributed on small input
// Uses a chi-squared test indicating 99% confidence the output is uniformly
// random
template <typename Vector>
void TestShuffleUniformPermutation() {
  typedef typename Vector::value_type T;

  size_t m = 5;
  size_t num_samples = 1000;
  size_t total_permutations = 1 * 2 * 3 * 4 * 5;
  std::map<thrust::host_vector<T>, size_t, vector_compare> permutation_counts;
  Vector sequence(m);
  thrust::sequence(sequence.begin(), sequence.end(), T(0));
  thrust::default_random_engine g(17);
  for (auto i = 0ull; i < num_samples; i++) {
    thrust::shuffle(sequence.begin(), sequence.end(), g);
    thrust::host_vector<T> tmp(sequence.begin(), sequence.end());
    permutation_counts[tmp]++;
  }

  ASSERT_EQUAL(permutation_counts.size(), total_permutations);

  double chi_squared = 0.0;
  double expected_count = static_cast<double>(num_samples) / total_permutations;
  for (auto kv : permutation_counts) {
    chi_squared += std::pow(expected_count - kv.second, 2) / expected_count;
  }
  // Tabulated chi-squared critical value for 119 degrees of freedom (5! - 1)
  // and 99% confidence
  double confidence_threshold = 157.8;
  ASSERT_LESS(chi_squared, confidence_threshold);
}
DECLARE_VECTOR_UNITTEST(TestShuffleUniformPermutation);
#endif
