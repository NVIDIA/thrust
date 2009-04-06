#include <komrade/host_vector.h>
#include <komrade/device_vector.h>
#include <komrade/generate.h>
#include <komrade/sort.h>
#include <cstdlib>

int main(void)
{
  // generate random data on the host
  komrade::host_vector<int> h_vec(20);
  komrade::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device
  komrade::device_vector<int> d_vec = h_vec;

  // sort on device
  komrade::sort(d_vec.begin(), d_vec.end());

  return 0;
}
