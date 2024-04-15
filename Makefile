CC=g++-9
NVCC=nvcc

# Use pkg-config to get OpenCV flags
OPENCV_CFLAGS=$(shell pkg-config --cflags opencv4)
OPENCV_LIBS=$(shell pkg-config --libs opencv4)

# CUDA runtime library
CUDA_LIBS=-lcudart

# Define C++ standard and any other compiler flags
CXXFLAGS=-std=c++14

# NVCC flags: specify the host compiler and pass the necessary OpenCV compiler flags
NVCCFLAGS=-ccbin $(CC) $(OPENCV_CFLAGS) -Xcompiler "$(CXXFLAGS)"

# Linker flags: Link both CUDA and OpenCV libraries
LDFLAGS=$(CUDA_LIBS) $(OPENCV_LIBS) -L/usr/local/cuda/lib64

TARGET=app
CUDA_SRC=imageprocessing.cu
CPP_SRC=segmentation.cpp  # Assuming the segmentation function is here

OBJ_FILES=imageprocessing.o segmentation.o

# Default target
all: $(TARGET)

# Target for the application
$(TARGET): $(OBJ_FILES)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile CUDA source to object files
imageprocessing.o: $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source to object files
segmentation.o: $(CPP_SRC)
	$(CC) $(OPENCV_CFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ_FILES)
