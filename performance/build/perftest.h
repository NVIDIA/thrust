#include <unittest/unittest.h>
#include <build/timer.h>
#include <string>
#include <algorithm>


//#include <cuda_runtime.h>
//#include <cuda.h>

#define RECORD_RESULT(name, value, units)   { std::cout << "  <result  name=\"" << name << "\"  value=\"" << value  << "\"  units=\"" << units << "\"/>" << std::endl; }
#define RECORD_TIME()                       RECORD_RESULT("Time", best_time, "seconds")
#define RECORD_RATE(name, value, units)     RECORD_RESULT(name, (double(value)/best_time), units)
#define RECORD_BANDWIDTH(bytes)             RECORD_RATE("Bandwidth", double(bytes) / 1e9, "GBytes/s")
#define RECORD_THROUGHPUT(value)            RECORD_RATE("Throughput", double(value) / 1e9, "GOp/s")
#define RECORD_SORTING_RATE(size)           RECORD_RATE("Sorting", double(size) / 1e6, "MKeys/s")
#define RECORD_VARIABLE(name, value)        { std::cout << "  <variable  name=\"" << name << "\"  value=\"" << value << "\"/>" << std::endl; }
#define RECORD_TEST_STATUS(result, message) { std::cout << "  <status  result=\"" << result  << "\"  message=\"" << message << "\"/>" << std::endl; }
#define RECORD_TEST_SUCCESS()               RECORD_TEST_STATUS("Success",  "")
#define RECORD_TEST_FAILURE(message)        RECORD_TEST_STATUS("Failure",  message)
#define BEGIN_TEST(name)                    { std::cout << "<test name=\"" << name << "\">" << std::endl; }
#define END_TEST()                          { std::cout << "</test>" << std::endl; }
#define BEGIN_TESTSUITE(name)               { std::cout << "<?xml version=\"1.0\" ?>" << std::endl << "<testsuite  name=\"" << name << "\">" << std::endl; }
#define END_TESTSUITE()                     { std::cout << "</testsuite>" << std::endl; }


#if defined(__GNUC__)  // GCC
#define __HOST_COMPILER_NAME__ "GCC"
# if defined(__GNUC_PATCHLEVEL__)
#define __HOST_COMPILER_VERSION__ (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
# else
#define __HOST_COMPILER_VERSION__ (__GNUC__ * 10000 + __GNUC_MINOR__ * 100)
# endif
#elif defined(_MSC_VER) // Microsoft Visual C++
#define __HOST_COMPILER_NAME__ "MSVC"
#define __HOST_COMPILER_VERSION__  _MSC_VER
#elif defined(__INTEL_COMPILER) // Intel Compiler
#define __HOST_COMPILER_NAME__ "ICC"
#define __HOST_COMPILER_VERSION__  __INTEL_COMPILER 
#else // Unknown
#define __HOST_COMPILER_NAME__ "UNKNOWN"
#define __HOST_COMPILER_VERSION__ 0
#endif


inline void RECORD_PLATFORM_INFO(void)
{
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0){
        std::cerr << "There is no device supporting CUDA" << std::endl;
        exit(1);
    }

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if (dev == 0 && deviceProp.major == 9999 && deviceProp.minor == 9999){
        std::cerr << "There is no device supporting CUDA" << std::endl;
        exit(1);
    }

    std::cout << "<platform>" << std::endl;
    std::cout << "  <device name=\"" << deviceProp.name << "\">" << std::endl;
    std::cout << "    <property name=\"revision\"" << " " << "value=\"" << deviceProp.major << "." << deviceProp.minor << "\"/>" << std::endl;
    std::cout << "    <property name=\"global memory\"" << " " << "value=\"" << deviceProp.totalGlobalMem << "\"  units=\"bytes\"/>" << std::endl;
    std::cout << "    <property name=\"multiprocessors\"" << " " << "value=\"" << deviceProp.multiProcessorCount << "\"/>" << std::endl;
    std::cout << "    <property name=\"cores\"" << " " << "value=\"" << 8*deviceProp.multiProcessorCount << "\"/>" << std::endl;
    std::cout << "    <property name=\"constant memory\"" << " " << "value=\"" << deviceProp.totalConstMem << "\"  units=\"bytes\"/>" << std::endl;
    std::cout << "    <property name=\"shared memory per block\"" << " " << "value=\"" << deviceProp.sharedMemPerBlock << "\"  units=\"bytes\"/>" << std::endl;
    std::cout << "    <property name=\"warp size\"" << " " << "value=\"" << deviceProp.warpSize << "\"/>" << std::endl;
    std::cout << "    <property name=\"max threads per block\"" << " " << "value=\"" << deviceProp.maxThreadsPerBlock << "\"/>" << std::endl;
    std::cout << "    <property name=\"clock rate\"" << " " << "value=\"" << (deviceProp.clockRate * 1e-6f) << "\"  units=\"GHz\"/>" << std::endl;
    std::cout << "  </device>" << std::endl;
    std::cout << "  <compilation>" << std::endl;
    std::cout << "    <property name=\"CUDA_VERSION\" value=\"" << CUDA_VERSION << "\"/>" << std::endl;
    std::cout << "    <property name=\"host compiler\" value=\"" << __HOST_COMPILER_NAME__ << " " << __HOST_COMPILER_VERSION__ << "\"/>" << std::endl;
    std::cout << "    <property name=\"__DATE__\" value=\"" << __DATE__ << "\"/>" << std::endl;
    std::cout << "    <property name=\"__TIME__\" value=\"" << __TIME__ << "\"/>" << std::endl;
    std::cout << "  </compilation>" << std::endl;
    std::cout << "</platform>" << std::endl;
#endif
}


inline void PROCESS_ARGUMENTS(int argc, char **argv)
{
  for(int i = 1; i < argc; ++i)
  {
    if(std::string(argv[i]) == "--device")
    {
      ++i;
      if(i == argc)
      {
        std::cerr << "usage: --device n" << std::endl;
        exit(-1);
      }

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
      int device_index = atoi(argv[i]);
      cudaSetDevice(device_index);
#endif
    }
  }
}


