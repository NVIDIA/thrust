#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>


// compute minimum and maximum values in a single reduction

// minmax_pair stores the minimum and maximum 
// values that have been encountered so far
template <typename T>
struct minmax_pair
{
    T min_val;
    T max_val;
};

// minmax_unary_op is a functor that takes in a value x and
// returns a minmax_pair whose minimum and maximum values
// are initialized to x.
template <typename T>
struct minmax_unary_op
{
    __host__ __device__
        minmax_pair<T> operator()(const T& x) const {
            minmax_pair<T> result;
            result.min_val = x;
            result.max_val = x;
            return result;
        }
};

// minmax_binary_op is a functor that accepts two minmax_pair 
// structs and returns a new minmax_pair whose minimum and 
// maximum values are the min() and max() respectively of 
// the minimums and maximums of the input pairs
template <typename T>
struct minmax_binary_op
{
    __host__ __device__
        minmax_pair<T> operator()(const minmax_pair<T>& x, const minmax_pair<T>& y) const {
            minmax_pair<T> result;
            result.min_val = thrust::min(x.min_val, y.min_val);
            result.max_val = thrust::max(x.max_val, y.max_val);
            return result;
        }
};


int main(void)
{
    // initialize host array
    int x[7] = {-1, 2, 7, -3, -4, 5};

    // transfer to device
    thrust::device_vector<float> d_x(x, x + 7);

    // setup arguments
    minmax_unary_op<int>  unary_op;
    minmax_binary_op<int> binary_op;
    minmax_pair<int> init = unary_op(d_x[0]);  // initialize with first element

    // compute minimum and maximum values
    minmax_pair<int> result = thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op);

    std::cout << result.min_val << std::endl;
    std::cout << result.max_val << std::endl;

    return 0;
}

