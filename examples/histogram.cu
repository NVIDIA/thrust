#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/adjacent_difference.h>

#include <iostream>
#include <iterator>
#include <cstdlib>

// This example compute the histogram [1] and cumulative
// histogram of an array of integer values.
//
// [1] http://en.wikipedia.org/wiki/Histogram


int main(void)
{
    const size_t N = 30;

    // generate random data on the host
    thrust::host_vector<int> h_data(N);
    for(size_t i = 0; i < N; i++)
        h_data[i] = rand() % 10;

    // transfer data to device
    thrust::device_vector<int> d_data(h_data);
    
    // print the initial data
    std::cout << "initial data" << std::endl;
    thrust::copy(d_data.begin(), d_data.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // sort data to bring equal elements together
    thrust::sort(d_data.begin(), d_data.end());
    
    // print the sorted data
    std::cout << "sorted data" << std::endl;
    thrust::copy(d_data.begin(), d_data.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // number of histogram bins is equal to the maximum value plus one
    const int num_bins = d_data.back() + 1;

    // allocate storage for the cumulative histogram and histogram
    thrust::device_vector<int> d_cumulative_histogram(num_bins);
    thrust::device_vector<int> d_histogram(num_bins);
    
    // find the end of each bin of values
    thrust::counting_iterator<int> search_begin(0);
    thrust::upper_bound(d_data.begin(),
                        d_data.end(),
                        search_begin,
                        search_begin + num_bins,
                        d_cumulative_histogram.begin());
    
    // print the sorted data
    std::cout << "sorted data" << std::endl;
    thrust::copy(d_data.begin(), d_data.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;


    // print the cumulative histogram
    std::cout << "cumulative histogram" << std::endl;
    thrust::copy(d_cumulative_histogram.begin(), d_cumulative_histogram.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(d_cumulative_histogram.begin(),
                                d_cumulative_histogram.end(),
                                d_histogram.begin());

    // print the histogram
    std::cout << "histogram" << std::endl;
    thrust::copy(d_histogram.begin(), d_histogram.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    
    return 0;
}

