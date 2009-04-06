#include <komrade/host_vector.h>
#include <komrade/device_vector.h>
#include <komrade/generate.h>
#include <komrade/reduce.h>
#include <komrade/functional.h>
#include <cstdlib>

int main(void)
{
  // generate random data on the host
  komrade::host_vector<int> h_vec(100);
  komrade::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device and compute sum
  komrade::device_vector<int> d_vec = h_vec;
  int x = komrade::reduce(d_vec.begin(), d_vec.end(), (int) 0,
                          komrade::plus<int>());
  return 0;
}
