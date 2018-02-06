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

