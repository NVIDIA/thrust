#include <thrust/host_vector.h>
#include <thrust/remove.h>

// This example generates random points in the 
// unit square [0,1)x[0,1) and then removes all 
// points where x^2 + y^2 > 1
//
// The x and y coordinates are stored in separate arrays
// and a zip_iterator is used to combine them together

template <typename T>
struct is_outside_circle
{
    template <typename Tuple>
    bool operator()(const Tuple& tuple) const
    {
        // unpack the tuple into x and y coordinates
        const T x = thrust::get<0>(tuple);
        const T y = thrust::get<1>(tuple);

        if (x*x + y*y > 1)
            return true;
        else
            return false;
    }
};

int main(void)
{
    const size_t N = 20;

    // generate random points in the unit square on the host
    thrust::host_vector<float> x(N);
    thrust::host_vector<float> y(N);
    for(size_t i = 0; i < N; i++)
    {
        x[i] = rand() / (RAND_MAX + 1.0f);
        y[i] = rand() / (RAND_MAX + 1.0f);
    }

    // print the initial points
    std::cout << "Generated " << N << " points" << std::endl;
    for(size_t i = 0; i < N; i++)
        std::cout << "(" << x[i] << "," << y[i] << ")" << std::endl;
    std::cout << std::endl;

    // remove points where x^2 + y^2 > 1 and determine new array sizes
    size_t new_size = thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end())),
                                        is_outside_circle<float>())
                      - thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin()));

    // resize the vectors (note: this does not free any memory)
    x.resize(new_size);
    y.resize(new_size);

    // print the filtered points
    std::cout << "After stream compaction, " << new_size << " points remain" << std::endl;
    for(size_t i = 0; i < new_size; i++)
        std::cout << "(" << x[i] << "," << y[i] << ")" << std::endl;

    return 0;
}

