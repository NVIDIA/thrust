#pragma once

#if defined(__NVCC__) && defined(__clang__) && __CUDACC_VER_MAJOR__ == 11 &&                       \
    __CUDACC_VER_MINOR__ <= 5

#if defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#  pragma nv_diag_suppress 3171
#else
#  pragma diag_suppress 3171
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wkeyword-compat"

// Clang has a builtin called `__is_signed`. Unfortunately, libstdc++ headers
// use this name as an identifier. Clang has a workaround for that, it checks 
// if `__is_signed` is `const static bool` as in libstdc++ headers and if so,
// disables the intrinsic for the rest of the TU:
// https://github.com/llvm/llvm-project/blob/f49b6afc231242dfee027d5da69734836097cd43/clang/lib/Parse/ParseDecl.cpp#L3552-L3566
const static bool __is_signed = false;

#pragma clang diagnostic pop
#endif // defined(__NVCC__) && defined(__clang__) && __CUDACC_VER_MAJOR__ == 11 &&
       //   __CUDACC_VER_MINOR__ <= 5
