#include <komrade/device_ptr.h>
#include <komrade/fill.h>
#include <cuda.h>

int main(void)
{
    size_t N = 10;

    // raw pointer to device memory
    int * raw_ptr;
    cudaMalloc((void **) &raw_ptr, N * sizeof(int));

    // wrap raw pointer with a device_ptr 
    komrade::device_ptr<int> dev_ptr(raw_ptr);

    // use device_ptr in komrade algorithms
    komrade::fill(dev_ptr, dev_ptr + N, (int) 0);

    // access device memory through device_ptr
    dev_ptr[0] = 1;

    // free memory
    cudaFree(raw_ptr);

    return 0;
}
