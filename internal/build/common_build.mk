I_AM_SLOPPY := 1
USE_NEW_PROJECT_MK := 1

ifeq ($(THRUST_TEST),1)
  include $(ROOTDIR)/build/getprofile.mk
  include $(ROOTDIR)/build/config/$(PROFILE).mk
else
  ifdef VULCAN_TOOLKIT_BASE
    include $(VULCAN_TOOLKIT_BASE)/build/getprofile.mk
    include $(VULCAN_TOOLKIT_BASE)/build/config/$(PROFILE).mk
  else
    include $(ROOTDIR)/build/getprofile.mk
    include $(ROOTDIR)/build/config/$(PROFILE).mk
  endif  # VULCAN_TOOLKIT_BASE
endif  # THRUST_TEST

ifeq ($(OS),Linux)
  LIBRARIES += m
endif

ifeq ($(OS),$(filter $(OS),Linux Darwin))
  ifndef USEPGCXX
    CUDACC_FLAGS += -Xcompiler "-Wall -Wextra -Werror"

    ifdef USEXLC
      # GCC does not warn about unused parameters in uninstantiated
      # template functions, but xlC does. This causes xlC to choke on the
      # OMP backend, which is mostly #ifdef'd out when you aren't using it.
      CUDACC_FLAGS += -Xcompiler "-Wno-unused-parameter"
    else # GCC, ICC or Clang AKA the sane ones.
      # XXX Enable -Wcast-align and -Wcast-qual.
      CUDACC_FLAGS += -Xcompiler "-Winit-self -Woverloaded-virtual -Wno-cast-align -Wno-long-long -Wno-variadic-macros"

      ifdef USE_CLANGLLVM
        IS_CLANG := 1
      endif

      ifeq ($(OS),Darwin)
        IS_CLANG := 1
      endif

      ifdef IS_CLANG 
        # GCC does not warn about unused parameters in uninstantiated
        # template functions, but Clang does. This causes Clang to choke on the
        # OMP backend, which is mostly #ifdef'd out when you aren't using it.
        CUDACC_FLAGS += -Xcompiler "-Wno-unused-parameter"

        # -Wunneeded-internal-declaration misfires in the unit test framework
        # on older versions of Clang.
        CUDACC_FLAGS += -Xcompiler "-Wno-unneeded-internal-declaration"
      else # GCC
        ifdef CCBIN
          GCC_VERSION = $(shell $(CCBIN) -dumpversion | sed -e 's/\.//g')
          ifeq ($(shell if test $(GCC_VERSION) -lt 420; then echo true; fi),true)
            # In GCC 4.1.2 and older, numeric conversion warnings are not
            # suppressable, so shut off -Wno-error.
            CUDACC_FLAGS += -Xcompiler "-Wno-error"
          endif
          ifeq ($(shell if test $(GCC_VERSION) -ge 450; then echo true; fi),true)
            # This isn't available until GCC 4.3, and misfires on TMP code until
            # GCC 4.5.
            CUDACC_FLAGS += -Xcompiler "-Wlogical-op"
          endif
          ifeq ($(shell if test $(GCC_VERSION) -ge 480; then echo true; fi),true)
            # XXX The mechanism for checking if compiler flags are supported
            # seems to be broken for the ARMv7 DVS builder, so the main CUDA
            # Makefiles accidentally add -Wno-unused-local-typedefs to older
            # GCC builds that don't support it.
            ifeq ($(TARGET_ARCH),ARMv7)
              C_WARNING_FLAGS_TMP := $(filter-out -Wno-unused-local-typedefs,$(C_WARNING_FLAGS))
              C_WARNING_FLAGS := $(C_WARNING_FLAGS_TMP)
            endif
          endif
        else
          $(error CCBIN is not defined)
        endif
      endif
    endif
  endif
else ifeq ($(OS),win32)
  # XXX Enable /Wall
  CUDACC_FLAGS += -Xcompiler "/WX"

  # Disabled loss-of-data conversion warnings.
  # XXX Re-enable.
  CUDACC_FLAGS += -Xcompiler "/wd4244 /wd4267"

  # Suppress numeric conversion-to-bool warnings.
  # XXX Re-enable.
  CUDACC_FLAGS += -Xcompiler "/wd4800"

  # Disable warning about applying unary - to unsigned type.
  CUDACC_FLAGS += -Xcompiler "/wd4146"
endif

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

# Add -mthumb for Linux on ARM to work around bug in arm cross compiler fom p4
ifeq ($(TARGET_ARCH),ARMv7)
  ifneq ($(HOST_ARCH),ARMv7)
    ifeq ($(THRUST_TEST),1)
      CUDACC_FLAGS += -Xcompiler "-mthumb"
    endif
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

ifdef VULCAN
  LIBDIRS_ABSPATH  += $(VULCAN_BUILD_DIR)/bin/$(VULCAN_ARCH)_$(VULCAN_OS)$(VULCAN_ABI)_$(VULCAN_BUILD)
endif

ifdef VULCAN_TOOLKIT_BASE
  include $(VULCAN_TOOLKIT_BASE)/build/common.mk
else
  include $(ROOTDIR)/build/common.mk
endif

