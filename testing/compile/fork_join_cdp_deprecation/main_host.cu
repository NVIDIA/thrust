#include <thrust/device_vector.h>

#ifdef DIRECT_CALL // reduce directly uses THRUST_CNP_DISPATCH, etc
#include <thrust/reduce.h>
#else // equal indirectly uses THRUST_CNP_DISPATCH, etc
#include <thrust/equal.h>
#endif
#include <iostream>

int main()
{
  const thrust::device_vector<int> vec(20, 1);

#ifdef DIRECT_CALL
  const int result = thrust::reduce(EXEC_POLICY,
                                    vec.cbegin(),
                                    vec.cend(),
                                    3);
  const int expected = 23;
#else
  const bool result = thrust::equal(EXEC_POLICY,
                                    vec.cbegin(),
                                    vec.cend(),
                                    vec.cbegin());
  const bool expected = true;
#endif

  std::cout << "Result: " << result << std::endl;

  if (result != expected)
  {
    std::cerr << "Expected '" << expected << "'\n";
    return 1;
  }

  return 0;
}
