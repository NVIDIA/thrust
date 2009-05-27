#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// this example shows how to implement a complicated
// transformation that requires 3 inputs and 1 output
// using for_each and counting_iterator

struct complicated_functor
{
    const float * A;
    const float * B;    
    const float * C;    
          float * D;

    complicated_functor(const float * _A, const float * _B, const float * _C, float * _D):
        A(_A), B(_B), C(_C), D(_D) {}

    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i){
        D[i] = 10 * A[i] + 20 * B[i] * C[i];
    }
};


int main(void)
{
    thrust::device_vector<float> A(10);
    thrust::device_vector<float> B(10);
    thrust::device_vector<float> C(10);
    thrust::device_vector<float> D(10);

    thrust::experimental::counting_iterator<int, thrust::random_access_device_iterator_tag> begin(0);
    thrust::experimental::counting_iterator<int, thrust::random_access_device_iterator_tag> end(10);

    complicated_functor op(thrust::raw_pointer_cast(&A[0]), 
                           thrust::raw_pointer_cast(&B[0]), 
                           thrust::raw_pointer_cast(&C[0]),
                           thrust::raw_pointer_cast(&D[0]));

    thrust::for_each(begin, end, op);
}
