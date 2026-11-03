# CudaOCR container image: CUDA 13.4 + cuDNN 9 on Ubuntu 24.04, model included.
#   docker build -t cudaocr .
#   docker run --rm --gpus all -v "$PWD":/data cudaocr scan.pdf -o out/
# Needs the NVIDIA Container Toolkit on the host.
ARG CUDA_VERSION=13.4.3

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS build
ARG CUDA_ARCHS="75;80;86;89;90;100;120"
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake pkg-config libpoppler-cpp-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN scripts/fetch_models.sh \
    && cmake -B /build -DCMAKE_BUILD_TYPE=Release -DCUDAOCR_BUILD_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build /build -j"$(nproc)" \
    && DESTDIR=/stage cmake --install /build

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libpoppler-cpp0t64 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /stage/usr /usr
WORKDIR /data
ENTRYPOINT ["cudaocr"]
CMD ["--help"]
