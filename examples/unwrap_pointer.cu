#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <cuda.h>

int main(void)
{
    size_t N = 10;

    // create a device_ptr 
    thrust::device_ptr<int> dev_ptr = thrust::device_malloc<int>(N);
     
    // extract raw pointer from device_ptr
    int * raw_ptr = thrust::raw_pointer_cast(dev_ptr);

    // use raw_ptr in non-thrust functions
    cudaMemset(raw_ptr, 0, N * sizeof(int));

    // free memory
    thrust::device_free(dev_ptr);

    return 0;
}
