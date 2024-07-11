# Check for verbose flag
ifeq ($(V),1)
  Q :=
else
  Q := @
endif

CC := g++-9
NVCC := nvcc

# OpenCV configuration
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

# CUDA paths and flags
CUDA_PATH := /usr/local/cuda
CUDA_INCLUDE := -I$(CUDA_PATH)/include
CUDA_LIBS := -L$(CUDA_PATH)/lib64 -lcudart -lcuda

# NPP configuration
NPP_INCLUDE := -I$(CUDA_PATH)/include
NPP_LIBS := -L$(CUDA_PATH)/lib64 -lnppig -lnppc -lnppicc -lnppif

# Compiler flags
CXXFLAGS := -std=c++14 -O3 -fopenmp -w $(OPENCV_CFLAGS)
NVCCFLAGS := -std=c++14 -O3 -arch=sm_61 -w $(OPENCV_CFLAGS)

# Include directories
INCLUDES := -Iinclude $(CUDA_INCLUDE) $(NPP_INCLUDE)

# Source files
CUDA_SRCS := $(wildcard src/*.cu)
CPP_SRCS := $(wildcard src/*.cpp)

# Object files
CUDA_OBJS := $(CUDA_SRCS:src/%.cu=obj/%.o)
CPP_OBJS := $(CPP_SRCS:src/%.cpp=obj/%.o)
OBJS := $(CUDA_OBJS) $(CPP_OBJS)

# Executable name
TARGET := app

# Default target
all: $(TARGET)

# Link the target
$(TARGET): $(OBJS)
	$(Q)echo "Linking $@..."
	$(Q)$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@ $(CUDA_LIBS) $(NPP_LIBS) $(OPENCV_LIBS)

# Compile CUDA source files
obj/%.o: src/%.cu
	$(Q)echo "Compiling $<..."
	$(Q)$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
obj/%.o: src/%.cpp
	$(Q)echo "Compiling $<..."
	$(Q)$(CC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	$(Q)echo "Cleaning up..."
	$(Q)rm -f $(TARGET) $(OBJS)

.PHONY: all clean