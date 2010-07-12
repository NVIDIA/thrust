#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

#include <iostream>
#include <iterator>

// BinaryPredicate for the head flag segment representation
// equivalent to thrust::not2(thrust::project2nd<int,int>()));
template <typename HeadFlagType>
struct head_flag_predicate 
    : public thrust::binary_function<HeadFlagType,HeadFlagType,bool>
{
    __host__ __device__
    bool operator()(HeadFlagType left, HeadFlagType right) const
    {
        return !right;
    }
};

int main(void)
{
    int keys[]   = {0,0,0,1,1,2,2,2,2,3,4,4,5,5,5};  // segments represented with keys
    int flags[]  = {1,0,0,1,0,1,0,0,0,1,1,0,1,0,0};  // segments represented with head flags
    int values[] = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};  // values corresponding to each key

    int N = sizeof(keys) / sizeof(int); // number of elements

    // copy input data to device
    thrust::device_vector<int> d_keys(keys, keys + N);
    thrust::device_vector<int> d_flags(flags, flags + N);
    thrust::device_vector<int> d_values(values, values + N);
    
    std::cout << "input:      ";  thrust::copy(d_values.begin(), d_values.end(), std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;
    std::cout << "keys:       ";  thrust::copy(d_keys.begin(),   d_keys.end(),   std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;
    std::cout << "head flags: ";  thrust::copy(d_flags.begin(),  d_flags.end(),  std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;
    
    // allocate storage for output
    thrust::device_vector<int> d_output(N);

    // scan using keys
    thrust::inclusive_scan_by_key(d_values.begin(), d_values.end(),
                                  d_keys.begin(),
                                  d_output.begin());
    
    // scan using head flags
    thrust::inclusive_scan_by_key(d_values.begin(), d_values.end(),
                                  d_flags.begin(),
                                  d_output.begin(),
                                  head_flag_predicate<int>(),
                                  thrust::plus<int>());
    
    std::cout << "output:     ";  thrust::copy(d_output.begin(), d_output.end(), std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;


    return 0;
}

