#include <thrust/version.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include "cuda_timer.h"
typedef cuda_timer device_timer;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
#include "tbb_timer.h"
typedef tbb_timer device_timer;
#else
#include "clock_timer.h"
typedef clock_timer device_timer;
#endif

