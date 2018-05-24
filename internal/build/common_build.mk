USE_NEW_PROJECT_MK := 1

ifeq ($(OS),Linux)
  LIBRARIES += m
endif

include $(ROOTDIR)/thrust/internal/build/common_warnings.mk

# Add /bigobj to Windows build flag to workaround building Thrust with debug
ifeq ($(OS), win32)
  CUDACC_FLAGS += -Xcompiler "/bigobj"
endif

ARCH_NEG_FILTER += 20 21
# Determine which SASS to generate
# if DVS (either per-CL or on-demand)
ifneq ($(or $(THRUST_DVS),$(THRUST_DVS_NIGHTLY)),)
  # DVS doesn't run Thrust on fermi so filter out SM 2.0/2.1
  # DVS doesn't run Thrust on mobile so filter those out as well
  # DVS doesn't have PASCAL configs at the moment
  ARCH_NEG_FILTER += 20 21 32 37 53 60
else
  # If building for ARMv7 (32-bit ARM), build only mobile SASS since no dGPU+ARM32 are supported anymore
  ifeq ($(TARGET_ARCH),ARMv7)
    ARCH_FILTER = 32 53 62
  endif
  # If its androideabi, we know its mobile, so can target specific SASS
  ifeq ($(OS),Linux)
    ifeq ($(ABITYPE), androideabi)
     ARCH_FILTER = 32 53 62
     ifeq ($(THRUST_TEST),1)
       NVCC_OPTIONS += -include "$(ROOTDIR)/cuda/tools/demangler/demangler.h"
       LIBRARIES += demangler
     endif
    endif
  endif
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

ifdef VULCAN
  LIBDIRS_ABSPATH  += $(VULCAN_BUILD_DIR)/bin/$(VULCAN_ARCH)_$(VULCAN_OS)$(VULCAN_ABI)_$(VULCAN_BUILD)
endif

ifdef VULCAN_TOOLKIT_BASE
  include $(VULCAN_TOOLKIT_BASE)/build/common.mk
else
  include $(ROOTDIR)/build/common.mk
endif

