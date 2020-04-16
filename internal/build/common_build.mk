USE_NEW_PROJECT_MK := 1

CCCL_ENABLE_DEPRECATIONS := 1

ifeq ($(OS),Linux)
  LIBRARIES += m
endif

include $(ROOTDIR)/thrust/internal/build/common_compiler.mk

# Add /bigobj to Windows build flag to workaround building Thrust with debug
ifeq ($(OS),win32)
  CUDACC_FLAGS += -Xcompiler "/bigobj"
endif

# Add -mthumb for Linux on ARM to work around bug in arm cross compiler from p4
ifeq ($(TARGET_ARCH),ARMv7)
  ifneq ($(HOST_ARCH),ARMv7)
    ifeq ($(THRUST_TEST),1)
      CUDACC_FLAGS += -Xcompiler "-mthumb"
    endif
  endif
endif

# Make PGI statically link against its libraries.
ifeq ($(OS),$(filter $(OS),Linux Darwin))
  ifdef USEPGCXX
    NVCC_LDFLAGS += -Xcompiler "-Bstatic_pgi"
  endif
endif
ifeq ($(SRC_PATH),)
  SRC_PATH:=$(dir $(BUILD_SRC))
  BUILD_SRC:=$(notdir $(BUILD_SRC))
endif

BUILD_SRC_SUFFIX:=$(suffix $(BUILD_SRC))

ifeq ($(BUILD_SRC_SUFFIX),.cu)
  CU_FILES += $(BUILD_SRC)
else ifeq ($(BUILD_SRC_SUFFIX),.cpp)
  FILES += $(BUILD_SRC)
endif

ifndef BUILD_AGAINST_RELEASE
  # CUDA includes
  ifdef VULCAN
    INCLUDES_ABSPATH += $(VULCAN_INSTALL_DIR)/cuda/include
    INCLUDES_ABSPATH += $(VULCAN_INSTALL_DIR)/cuda/_internal/cudart
  else
    INCLUDES_ABSPATH += $(ROOTDIR)/cuda/inc
    INCLUDES_ABSPATH += $(ROOTDIR)/cuda/tools/cudart
  endif

  # Thrust includes
  ifdef VULCAN
    INCLUDES_ABSPATH += $(VULCAN_TOOLKIT_BASE)/thrust
  else
    INCLUDES_ABSPATH += $(ROOTDIR)/thrust
  endif

  # CUB includes
  ifdef VULCAN
    INCLUDES_ABSPATH += $(VULCAN_TOOLKIT_BASE)/cub
  else
    INCLUDES_ABSPATH += $(ROOTDIR)/cub
  endif
else
  # CUDA, CUB, and Thrust includes
  INCLUDES_ABSPATH += $(GPGPU_COMPILER_EXPORT)/include

  ifeq ($(TARGET_ARCH),ARMv7)
    LIBDIRS_ABSPATH += $(GPGPU_COMPILER_EXPORT)/lib32
  else
    LIBDIRS_ABSPATH += $(GPGPU_COMPILER_EXPORT)/lib64
  endif
endif

ifdef VULCAN
  LIBDIRS_ABSPATH  += $(VULCAN_BUILD_DIR)/bin/$(VULCAN_ARCH)_$(VULCAN_OS)$(VULCAN_ABI)_$(VULCAN_BUILD)
endif

USES_CUDA_DRIVER_HEADERS := 1

ifdef VULCAN_TOOLKIT_BASE
  include $(VULCAN_TOOLKIT_BASE)/build/common.mk
else
  include $(ROOTDIR)/build/common.mk
endif

