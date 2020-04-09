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

CU_FILES += ../test/warningstester.cu

# Thrust includes
ifdef VULCAN
INCLUDES += $(VULCAN_INSTALL_DIR)/cuda/include
INCLUDES += $(VULCAN_INSTALL_DIR)/cuda/_internal/cudart
INCLUDES += $(VULCAN_TOOLKIT_BASE)/cub
else
INCLUDES += ../..
INCLUDES += ../../../cuda/tools/cudart
INCLUDES += ../../../cub
endif

# Location of generated include file that includes all Thrust public headers
GENERATED_SOURCES = $(BUILT_CWD)
CUDACC_FLAGS += -I$(GENERATED_SOURCES)

include $(ROOTDIR)/thrust/internal/build/common_compiler.mk

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
