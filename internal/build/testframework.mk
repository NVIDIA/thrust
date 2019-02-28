STATIC_LIBRARY := testframework

SRC_PATH := $(ROOTDIR)/thrust/testing/
BUILD_SRC := unittest/testframework.cu

CUSRC := unittest/cuda/testframework.cu
$(CUSRC).CUDACC_FLAGS    := -I$(ROOTDIR)/thrust/testing/cuda/
$(CUSRC).TARGET_BASENAME := testframework_cu
CU_FILES += $(CUSRC)

INCLUDES_ABSPATH += $(ROOTDIR)/thrust/testing

THRUST_TEST := 1

include $(ROOTDIR)/thrust/internal/build/common_detect.mk
include $(ROOTDIR)/thrust/internal/build/common_build.mk

