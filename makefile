# Specifying the compilers
CC=g++
NVCC=nvcc

# Dynamically fetch the OpenCV compilation and linking flags using pkg-config
OPENCV_CFLAGS=$(shell pkg-config --cflags opencv4)
OPENCV_LIBS=$(shell pkg-config --libs opencv4)

# CUDA runtime library and C++ standard
LIBS=-lcudart $(OPENCV_LIBS)
CXXFLAGS=-std=c++17
NVCCFLAGS=-ccbin g++-9 $(OPENCV_CFLAGS) -Xcompiler "$(CXXFLAGS)"

# Compiler flags for linking
LDFLAGS=$(LIBS)

# Target executable name
TARGET=ocr_app

# Source files
SRC = imageprocessing.cu

# Compile and link the program
$(TARGET):
	$(NVCC) $(SRC) -o $(TARGET) $(NVCCFLAGS) $(LDFLAGS)

# Clean objects in the directory
clean:
	rm -f $(TARGET)
