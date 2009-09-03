#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>

// This example shows how to implement an arbitrary transformation of
// the form output[i] = F(first[i], second[i], third[i], ... ).
// In this example, we use a function with 3 inputs and 1 output.
//
// Iterators for all four vectors (3 inputs + 1 output) are "zipped"
// into a single sequence of tuples with the zip_iterator.
//  
// The arbitrary_functor receives a tuple that contains four elements,
// which are references to values in each of the four sequences. When we
// access the tuple 't' with the get() function,
//      get<0>(t) returns a reference to A[i],
//      get<1>(t) returns a reference to B[i],
//      get<2>(t) returns a reference to C[i],
//      get<3>(t) returns a reference to D[i].
//
// In this example, we can implement the transformation,
//      D[i] = A[i] + B[i] * C[i];
// by invoking arbitrary_functor() on each of the tuples using for_each.
//
// Note that we could extend this example to implement functions with an
// arbitrary number of input arguments by zipping more sequence together.
// With the same approach we can have multiple *output* sequences, if we 
// wanted to implement something like
//      D[i] = A[i] + B[i] * C[i];
//      E[i] = A[i] + B[i] + C[i];
//
// The possibilities are endless! :)

struct arbitrary_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
    }
};


int main(void)
{
    // allocate storage
    thrust::device_vector<float> A(5);
    thrust::device_vector<float> B(5);
    thrust::device_vector<float> C(5);
    thrust::device_vector<float> D(5);

    // initialize input vectors
    A[0] = 3;  B[0] = 6;  C[0] = 2; 
    A[1] = 4;  B[1] = 7;  C[1] = 5; 
    A[2] = 0;  B[2] = 2;  C[2] = 7; 
    A[3] = 8;  B[3] = 1;  C[3] = 4; 
    A[4] = 2;  B[4] = 8;  C[4] = 3; 

    // apply the transformation
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end(),   C.end(),   D.end())),
                     arbitrary_functor());

    // print the output
    for(int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D[i] << std::endl;
}

