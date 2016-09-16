USE_NEW_PROJECT_MK := 1
EXECUTABLE        := bench
PROJ_DIR          := internal/benchmark

include $(ROOTDIR)/build/config/DetectOS.mk

CU_FILES += bench.cu

# Thrust includes
INCLUDES += ../../

I_AM_SLOPPY = 1

CUDACC_FLAGS += -DNO_TBB
CUDACC_FLAGS += $(GENSASS_SM10PLUS)

ifeq ($(OS),Linux)
ifeq ($(ABITYPE), androideabi)
    override ALL_SASS_ARCHITECTURES := 32
    CUDACC_FLAGS += $(GENSASS_SM32)
endif
endif
ARCH_NEG_FILTER += 20 21

include $(ROOTDIR)/build/common.mk
