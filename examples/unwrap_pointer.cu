#include <komrade/device_ptr.h>
#include <komrade/device_malloc.h>
#include <komrade/device_free.h>
#include <cuda.h>

int main(void)
{
    size_t N = 10;

    // create a device_ptr 
    komrade::device_ptr<int> dev_ptr = komrade::device_malloc<int>(N);
     
    // extract raw pointer from device_ptr
    int * raw_ptr = komrade::raw_pointer_cast(dev_ptr);

    // use raw_ptr in non-komrade functions
    cudaMemset(raw_ptr, 0, N * sizeof(int));

    // free memory
    komrade::device_free(dev_ptr);

    return 0;
}
