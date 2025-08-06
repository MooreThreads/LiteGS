#pragma once
#include "musa_runtime.h"
void cuda_error_check(const char* file, const char* function);

//#define CUDA_DEBUG
#ifdef CUDA_DEBUG
    #define CUDA_CHECK_ERRORS cuda_error_check(__FILE__,__FUNCTION__)
#else
    #define CUDA_CHECK_ERRORS
#endif