#!/usr/bin/env bash
# Build release artifacts into dist/.
#
#   scripts/package.sh host     # .deb for this machine's Ubuntu release (native build)
#   scripts/package.sh noble    # .deb for Ubuntu 24.04, built in the CUDA 13.4 devel container
#   scripts/package.sh docker   # Docker image cudaocr:<version> (see Dockerfile)
#   scripts/package.sh all
#
# Packages are Release builds for every supported GPU generation (sm_75..sm_120),
# without tests, installing to /usr with the bundled ONNX Runtime and model.
# Depends: are computed from the binaries only, so CUDA/cuDNN versions are not
# pinned (they are a documented runtime requirement; cuDNN is a Recommends).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHS="75;80;86;89;90;100;120"
CUDA_IMAGE="nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04"
VERSION="$(sed -n 's/^project(cudaocr VERSION \([0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt")"
mkdir -p "$ROOT/dist"

# deb <build_dir> <distro_tag> [extra cmake args...] — runs where the toolchain is.
deb() {
    local bdir="$1" distro="$2"
    shift 2
    [ -f "$ROOT/models/rec.onnx" ] || "$ROOT/scripts/fetch_models.sh"
    cmake -S "$ROOT" -B "$bdir" -DCMAKE_BUILD_TYPE=Release -DCUDAOCR_BUILD_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
        -DCUDAOCR_DEB_DISTRO="$distro" "$@"
    cmake --build "$bdir" -j"$(nproc)"
    local stage
    stage="$(mktemp -d)"
    DESTDIR="$stage" cmake --install "$bdir" >/dev/null
    local libdir
    libdir="$(dirname "$(find "$stage" -name 'libonnxruntime.so.1' | head -1)")"
    mkdir -p "$stage/debian" && touch "$stage/debian/control"
    local deps
    deps="$(cd "$stage" && dpkg-shlibdeps -O --ignore-missing-info -l"$libdir" \
        usr/bin/cudaocr usr/bin/cudaocr-eval 2>/dev/null | sed 's/^shlibs:Depends=//')"
    rm -rf "$stage"
    echo "Depends: $deps"
    (cd "$bdir" && cpack -G DEB -D CPACK_DEBIAN_PACKAGE_DEPENDS="$deps" \
        -D CPACK_PACKAGE_DIRECTORY="$ROOT/dist" >/dev/null)
    ls -l "$ROOT"/dist/*"${distro}"*.deb
}

host() {
    . /etc/os-release
    deb "$ROOT/build-pkg-host" "ubuntu${VERSION_ID}" \
        ${ORT_ROOT:+-DCUDAOCR_ORT_ROOT="$ORT_ROOT"}
}

noble() {
    # Reuse the already-downloaded ONNX Runtime and toml++ sources if present.
    local ort="$ROOT/build/_deps/onnxruntime-src" toml="$ROOT/build/_deps/tomlplusplus-src"
    docker run --rm -v "$ROOT":/src -w /src -e HOME=/tmp \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$CUDA_IMAGE" bash -c "
        set -e
        apt-get update -qq && apt-get install -y -qq --no-install-recommends \
            cmake pkg-config libpoppler-cpp-dev dpkg-dev curl ca-certificates file >/dev/null
        source scripts/package.sh --source-only
        deb /tmp/build-noble ubuntu24.04 \
            $([ -d "$ort" ] && echo "-DCUDAOCR_ORT_ROOT=/src/build/_deps/onnxruntime-src") \
            $([ -d "$toml" ] && echo "-DFETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS=/src/build/_deps/tomlplusplus-src")
        chown \$HOST_UID:\$HOST_GID dist/*.deb"
}

docker_image() {
    docker build -t "cudaocr:${VERSION}" -t cudaocr:latest --build-arg CUDA_ARCHS="$ARCHS" "$ROOT"
}

[ "${1:-}" == "--source-only" ] && return 0
case "${1:-}" in
    host) host ;;
    noble) noble ;;
    docker) docker_image ;;
    all) host && noble && docker_image ;;
    *) echo "usage: scripts/package.sh host|noble|docker|all" >&2; exit 2 ;;
esac
