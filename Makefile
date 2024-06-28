CC := g++-9
NVCC := nvcc

# OpenCV configuration
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

# CUDA paths and flags
CUDA_PATH := /usr/local/cuda
CUDA_INCLUDE := -I$(CUDA_PATH)/include
CUDA_LIBS := -L$(CUDA_PATH)/lib64 -lcudart -lcuda

# Compiler flags
CXXFLAGS := -std=c++14 -O3 -fopenmp -w $(OPENCV_CFLAGS)
NVCCFLAGS := -std=c++14 -O3 -arch=sm_61 -w $(OPENCV_CFLAGS)

# Include directories
INCLUDES := -I. $(CUDA_INCLUDE)

# Source files
CUDA_SRCS := $(wildcard *.cu)
CPP_SRCS := $(wildcard *.cpp)

# Object files
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
OBJS := $(CUDA_OBJS) $(CPP_OBJS)

# Executable name
TARGET := app

# Default target
all: $(TARGET)

# Link the target
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(CUDA_LIBS) $(OPENCV_LIBS)

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
