#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/is_sorted.h>
#include <cstdlib>
#include <iostream>
#include <iterator>

// defines the function prototype
#include "device.h"

int main(void)
{
    // generate 20 random numbers on the host
    thrust::host_vector<int> h_vec(20);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);

    // interface to CUDA code
    sort_on_device(h_vec);

    // print sorted array
    thrust::copy(h_vec.begin(), h_vec.end(), std::ostream_iterator<int>(std::cout, "\n"));

    return 0;
}

