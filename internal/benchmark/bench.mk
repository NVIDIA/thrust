EXECUTABLE := bench
BUILD_SRC  := $(ROOTDIR)/thrust/internal/benchmark/bench.cu

BUILD_SRC_FLAGS += -DNO_TBB
BUILD_SRC_FLAGS += $(GENSASS_SM10PLUS)

LDFLAGS += -lm

ifeq ($(OS),Linux)
  ifeq ($(ABITYPE), androideabi)
    override ALL_SASS_ARCHITECTURES := 32
    BUILD_SRC_FLAGS += $(GENSASS_SM32)
  endif
endif

ARCH_NEG_FILTER += 20 21

include $(ROOTDIR)/thrust/internal/build/common_build.mk
