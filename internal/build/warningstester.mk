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

ifeq ($(OS),Linux)
    ifndef USEPGCXX
        CUDACC_FLAGS += -Xcompiler "-pedantic -Wall -Wextra -Winit-self -Woverloaded-virtual -Wcast-align -Wcast-qual -Wno-long-long -Wno-variadic-macros"

        GCC_VERSION = $(shell $(CC) -dumpversion | sed -e 's/\.//g')
        ifeq ($(shell if test $(GCC_VERSION) -ge 430; then echo true; fi),true)
            # These two were added in GCC 4.3
            CUDACC_FLAGS += -Xcompiler "-Wlogical-op -Wno-vla"
        endif
    endif
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
