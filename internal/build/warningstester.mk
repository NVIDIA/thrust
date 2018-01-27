USE_NEW_PROJECT_MK := 1
EXECUTABLE        := warningstester
PROJ_DIR          := internal/build
#GENCODE           :=

ifndef PROFILE
ifdef VULCAN_TOOLKIT_BASE
include $(VULCAN_TOOLKIT_BASE)/build/getprofile.mk
include $(VULCAN_TOOLKIT_BASE)/build/config/$(PROFILE).mk
else
include $(ROOTDIR)/build/getprofile.mk
include $(ROOTDIR)/build/config/$(PROFILE).mk
endif
endif

ARCH_NEG_FILTER += 20 21

ifdef VULCAN_TOOLKIT_BASE
include $(VULCAN_TOOLKIT_BASE)/build/config/DetectOS.mk
else
include $(ROOTDIR)/build/config/DetectOS.mk
endif

FILES += ../test/warningstester.cpp

# Thrust includes (thrust/)
ifdef VULCAN
INCLUDES += $(VULCAN_INSTALL_DIR)/cuda/include/
INCLUDES += $(VULCAN_INSTALL_DIR)/cuda/_internal/cudart
else
INCLUDES += ../../
INCLUDES += ../../../cuda/tools/cudart
endif

# Location of generated include file that includes all Thrust public headers
GENERATED_SOURCES = $(BUILT_CWD)
CUDACC_FLAGS += -I$(GENERATED_SOURCES)

ifeq ($(OS),$(filter $(OS),Linux Darwin))
  ifndef USEPGCXX
    CUDACC_FLAGS += -Xcompiler "-pedantic -Wall -Wextra -Werror"

    ifdef USEXLC
      # GCC does not warn about unused parameters in uninstantiated
      # template functions, but xlC does. This causes xlC to choke on the
      # OMP backend, which is mostly #ifdef'd out when you aren't using it.
      CUDACC_FLAGS += -Xcompiler "-Wno-unused-parameter"
    else # GCC, ICC or Clang AKA the sane ones.
      # XXX Enable -Wcast-align.
      CUDACC_FLAGS += -Xcompiler "-Winit-self -Woverloaded-virtual -Wno-cast-align -Wcast-qual -Wno-long-long -Wno-variadic-macros"

      ifdef USE_CLANGLLVM
        IS_CLANG := 1
      endif

      ifeq ($(OS),Darwin)
        IS_CLANG := 1
      endif

      ifdef IS_CLANG 
        # -Wunneeded-internal-declaration misfires in the unit test framework
        # on older versions of Clang.
        CUDACC_FLAGS += -Xcompiler "-Wno-unneeded-internal-declaration"

        # GCC does not warn about unused parameters in uninstantiated
        # template functions, but Clang does. This causes Clang to choke on the
        # OMP backend, which is mostly #ifdef'd out when you aren't using it.
        CUDACC_FLAGS += -Xcompiler "-Wno-unused-parameter"
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

ifdef VULCAN_TOOLKIT_BASE
include $(VULCAN_TOOLKIT_BASE)/build/common.mk
else
include $(ROOTDIR)/build/common.mk
endif

warningstester$(OBJSUFFIX): $(GENERATED_SOURCES)/warningstester.h

$(GENERATED_SOURCES)/warningstester.h: FORCE
ifdef VULCAN
ifeq ($(TARGET_ARCH), ppc64le)
	$(PYTHON) $(SRC_CWD)/warningstester_create_uber_header.py $(VULCAN_INSTALL_DIR)/cuda/targets/ppc64le-linux/include > $@
else
	$(PYTHON) $(SRC_CWD)/warningstester_create_uber_header.py $(VULCAN_INSTALL_DIR)/cuda/include > $@
endif
else
	$(PYTHON) $(SRC_CWD)/warningstester_create_uber_header.py $(SRC_CWD)/../.. > $@
endif

FORCE:
