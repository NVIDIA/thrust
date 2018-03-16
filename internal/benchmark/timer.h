#pragma once

#include <cuda.h>

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

class timer
{
    cudaEvent_t _start;
    cudaEvent_t _end;

    public:
    timer()
    {
        CUDA_SAFE_CALL(cudaEventCreate(&_start));
        CUDA_SAFE_CALL(cudaEventCreate(&_end));
    }

    ~timer()
    {
        CUDA_SAFE_CALL(cudaEventDestroy(_start));
        CUDA_SAFE_CALL(cudaEventDestroy(_end));
    }

    void start()
    {
        CUDA_SAFE_CALL(cudaEventRecord(_start, 0));
    }

    void stop()
    {
        CUDA_SAFE_CALL(cudaEventRecord(_end, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(_end));
    }

    double milliseconds_elapsed()
    {
        float elapsed_time;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, _start, _end));
        return elapsed_time;
    }

    double seconds_elapsed()
    {
        return milliseconds_elapsed() / 1000.0;
    }
};


