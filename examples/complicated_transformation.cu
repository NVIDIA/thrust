#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/generate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <cstdlib>

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
        D[i] = A[i] + B[i] * C[i];
    }
};


int main(void)
{
    thrust::device_vector<float> A(5);
    thrust::device_vector<float> B(5);
    thrust::device_vector<float> C(5);
    thrust::device_vector<float> D(5);

    A[0] = 3;  B[0] = 6;  C[0] = 2; 
    A[1] = 4;  B[1] = 7;  C[1] = 5; 
    A[2] = 0;  B[2] = 2;  C[2] = 7; 
    A[3] = 8;  B[3] = 1;  C[3] = 4; 
    A[4] = 2;  B[4] = 8;  C[4] = 3; 

    thrust::experimental::counting_iterator<int, thrust::random_access_device_iterator_tag> begin(0);

    complicated_functor op(thrust::raw_pointer_cast(&A[0]), 
                           thrust::raw_pointer_cast(&B[0]), 
                           thrust::raw_pointer_cast(&C[0]),
                           thrust::raw_pointer_cast(&D[0]));

    thrust::for_each(begin, begin + 5, op);

    for(int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D[i] << std::endl;
}
