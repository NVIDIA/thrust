#include <thrusttest/unittest.h>

#if defined(__CUDACC__)

#include <thrust/experimental/arch.h>

using namespace thrust::experimental::arch;

void set_compute_capability(cudaDeviceProp& properties, int major, int minor)
{
    properties.major = major;
    properties.minor = minor;
}

void set_G80(cudaDeviceProp& properties)
{
    set_compute_capability(properties, 1, 0);
    properties.multiProcessorCount = 16;
    properties.sharedMemPerBlock   = 16384;
    properties.regsPerBlock        = 8192;
    properties.warpSize            = 32;
    properties.maxThreadsPerBlock  = 512;
}

void set_G84(cudaDeviceProp& properties)
{
    set_compute_capability(properties, 1, 1);
    properties.multiProcessorCount = 4;
    properties.sharedMemPerBlock   = 16384;
    properties.regsPerBlock        = 8192;
    properties.warpSize            = 32;
    properties.maxThreadsPerBlock  = 512;
}

void set_GT200(cudaDeviceProp& properties)
{
    set_compute_capability(properties, 1, 3);
    properties.multiProcessorCount = 30;
    properties.sharedMemPerBlock   = 16384;
    properties.regsPerBlock        = 16384;
    properties.warpSize            = 32;
    properties.maxThreadsPerBlock  = 512;
}

void set_func_attributes(cudaFuncAttributes& attributes,
                         size_t constSizeBytes,           // Size of constant memory in bytes.
                         size_t localSizeBytes,           // Size of local memory in bytes.
                         int maxThreadsPerBlock,          // Maximum number of threads per block.
                         int numRegs,                     // Number of registers used.
                         size_t sharedSizeBytes)          // Size of shared memory in bytes.
{
    attributes.constSizeBytes     = constSizeBytes;
    attributes.localSizeBytes     = localSizeBytes;
    attributes.maxThreadsPerBlock = maxThreadsPerBlock; 
    attributes.numRegs            = numRegs;
    attributes.sharedSizeBytes    = sharedSizeBytes;
}

void TestMaxActiveThreads(void)
{
    cudaDeviceProp properties;

    set_compute_capability(properties, 1, 0);
    ASSERT_EQUAL(max_active_threads_per_multiprocessor(properties), 768);
    
    set_compute_capability(properties, 1, 1);
    ASSERT_EQUAL(max_active_threads_per_multiprocessor(properties), 768);
    
    set_compute_capability(properties, 1, 2);
    ASSERT_EQUAL(max_active_threads_per_multiprocessor(properties), 1024);
    
    set_compute_capability(properties, 1, 3);
    ASSERT_EQUAL(max_active_threads_per_multiprocessor(properties), 1024);
}
DECLARE_UNITTEST(TestMaxActiveThreads);


void TestMaxActiveBlocks(void)
{
    cudaDeviceProp properties;
    cudaFuncAttributes attributes;

    // Kernel #1 : Full Occupancy on all devices
    set_func_attributes(attributes, 0, 0, 512, 10, 2048);
    
    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 3);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 3);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 4);
    
    // Kernel #2 : 2/3rds Occupancy on G8x and 100% on GT200
    set_func_attributes(attributes, 0, 0, 512, 16, 2048);

    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 4);
    
    // Kernel #3 : 1/3rds Occupancy on G8x and 75% on GT200
    set_func_attributes(attributes, 0, 0, 512, 20, 2048);

    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 1);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 1);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 3);
    
    // Kernel #4 : 1/3rds Occupancy on G8x and 50% on GT200
    set_func_attributes(attributes, 0, 0, 512, 21, 2048);

    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 1);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 1);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    
    // Kernel #5 : 2/3rds Occupancy on G8x and 50% on GT200
    set_func_attributes(attributes, 0, 0, 512, 10, 8192);

    set_G80(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    set_G84(properties);   ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
    set_GT200(properties); ASSERT_EQUAL(max_active_blocks_per_multiprocessor(properties, attributes, 256, 0), 2);
}
DECLARE_UNITTEST(TestMaxActiveBlocks);

#endif // defined(__CUDACC__)

