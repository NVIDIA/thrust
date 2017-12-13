STATIC_LIBRARY := testframework
BUILD_SRC      := $(ROOTDIR)/thrust/testing/testframework.cpp

CUTESTFRMWRK := $(ROOTDIR)/thrust/testing/backend/cuda/testframework.cu
$(CUTESTFRMWRK).CUDACC_FLAGS    := -I$(ROOTDIR)/thrust/testing/backend/cuda/
$(CUTESTFRMWRK).TARGET_BASENAME := testframework_cu

CU_FILES_ABSPATH += $(CUTESTFRMWRK)

INCLUDES_ABSPATH += $(ROOTDIR)/thrust/testing

THRUST_TEST := 1
include $(ROOTDIR)/thrust/internal/build/common_build.mk

