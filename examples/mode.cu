#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>

#include <iostream>
#include <iterator>
#include <cstdlib>

// This example compute the mode [1] of a set of numbers.  If there
// are multiple modes, one with the smallest value it returned.
//
// [1] http://en.wikipedia.org/wiki/Mode_(statistics)

template <typename Tuple>
struct combine_counts : thrust::binary_function<Tuple,Tuple,Tuple>
{
    __host__ __device__
    Tuple operator()(Tuple a, Tuple b)
    {
        if(thrust::get<0>(a) == thrust::get<0>(b))
            return Tuple(thrust::get<0>(a), thrust::get<1>(a) + thrust::get<1>(b));
        else
            return Tuple(thrust::get<0>(b), thrust::get<1>(b));
    }
};


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

    // scan the values and counts together, adding the counts 
    // together when the two values are equal
    thrust::device_vector<unsigned int> d_counts(N, 1);
    thrust::inclusive_scan(thrust::make_zip_iterator(thrust::make_tuple(d_data.begin(), d_counts.begin())),
                           thrust::make_zip_iterator(thrust::make_tuple(d_data.end(),   d_counts.end())),
                           thrust::make_zip_iterator(thrust::make_tuple(d_data.begin(), d_counts.begin())),
                           combine_counts< thrust::tuple<int,unsigned int> >());
    
    // print the counts
    std::cout << "counts" << std::endl;
    thrust::copy(d_counts.begin(), d_counts.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // find the index of the maximum count
    thrust::device_vector<unsigned int>::iterator mode_iter;
    mode_iter = thrust::max_element(d_counts.begin(), d_counts.end());

    int mode = d_data[mode_iter - d_counts.begin()];
    unsigned int occurances = *mode_iter;
    
    std::cout << "Modal value " << mode << " occurs " << occurances << " times " << std::endl;
    
    return 0;
}

