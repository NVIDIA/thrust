#ifdef VULCAN_TOOLKIT_BASE

#ifndef PROFILE
#include $(ROOTDIR)/build/getprofile.mk
#include $(ROOTDIR)/build/config/$(PROFILE).mk
#endif
#include $(ROOTDIR)/build/config/DetectOS.mk

ifdef VULCAN_TOOLKIT_BASE
include $(VULCAN_TOOLKIT_BASE)/build/config/DetectOS.mk
else
include $(ROOTDIR)/build/config/DetectOS.mk
endif

ifndef PROFILE
ifdef VULCAN_TOOLKIT_BASE
include $(VULCAN_TOOLKIT_BASE)/build/getprofile.mk
include $(VULCAN_TOOLKIT_BASE)/build/config/$(PROFILE).mk
else
include $(ROOTDIR)/build/getprofile.mk
include $(ROOTDIR)/build/config/$(PROFILE).mk
endif
endif


USE_NEW_PROJECT_MK := 1
ARCH_NEG_FILTER += 20 21



ifdef ERIS_TEST_LEVELS
BINPATH=${VULCAN_BUILD_DIR}/bin/${VULCAN_ARCH}_${VULCAN_OS}${VULCAN_ABI}_${VULCAN_BUILD}

ifneq ($(MAKECMDGOALS),clean)
  res:=$(shell $(PYTHON) $(ROOTDIR)/thrust/generate_eris_vlct.py $(BINPATH) $(ERIS_TEST_LEVELS))
endif

endif  # ERIS_TEST_LEVELS

ifdef VULCAN_TOOLKIT_BASE
include $(VULCAN_TOOLKIT_BASE)/build/common.mk
else
include $(ROOTDIR)/build/common.mk
endif
