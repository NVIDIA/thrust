I_AM_SLOPPY := 1
USE_NEW_PROJECT_MK := 1

ifeq ($(THRUST_TEST),1)
  include $(ROOTDIR)/build/config/DetectOS.mk
else
  ifdef VULCAN_TOOLKIT_BASE
    include $(VULCAN_TOOLKIT_BASE)/build/config/DetectOS.mk
  else
    include $(ROOTDIR)/build/config/DetectOS.mk
  endif  # VULCAN_TOOLKIT_BASE
endif  # THRUST_TEST

ifeq ($(OS),Linux)
LIBRARIES += m
endif

#
# Add /bigobj to Windows build flag to workaround building Thrust with debug
#
ifeq ($(OS), win32)
CUDACC_FLAGS += -Xcompiler /bigobj
endif

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
 # if its androideabi, we know its mobile, so can target specific SASS
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

#
# Add -mthumb for Linux on ARM to work around bug in arm cross compiler fom p4
#
ifeq ($(TARGET_ARCH),ARMv7)
ifneq ($(HOST_ARCH),ARMv7)
ifeq ($(THRUST_TEST),1)
CUDACC_FLAGS += -Xcompiler -mthumb
endif
endif
endif

BUILD_SRC_SUFFIX=$(suffix $(BUILD_SRC))
ifeq ($(BUILD_SRC_SUFFIX),.cu)
  CU_FILES_ABSPATH += $(BUILD_SRC)
else ifeq ($(BUILD_SRC_SUFFIX),.cpp)
  FILES_ABSPATH += $(BUILD_SRC)
endif
$(BUILD_SRC).CUDACC_FLAGS += $(BUILD_SRC_FLAGS)


# CUDA includes
ifdef VULCAN
INCLUDES_ABSPATH += $(VULCAN_INSTALL_DIR)/cuda/include/
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

ifdef ERIS_TEST_LEVELS
LIBDIRS_ABSPATH  += ${VULCAN_BUILD_DIR}/bin/${VULCAN_ARCH}_${VULCAN_OS}${VULCAN_ABI}_${VULCAN_BUILD}
endif

ifdef VULCAN_TOOLKIT_BASE
include $(VULCAN_TOOLKIT_BASE)/build/common.mk
else
include $(ROOTDIR)/build/common.mk
endif
