#include "cudaocr/version.hpp"

#include "cudaocr/core/pdf.hpp"

namespace cudaocr {

const char* version() {
    return CUDAOCR_VERSION;
}

std::string buildInfo(const char* program) {
    std::string s = std::string(program) + " " + version() + "\n";
#ifdef CUDAOCR_WITH_ORT
    s += "recognizer: onnxruntime " CUDAOCR_ORT_VERSION " (cuda and cpu execution providers)\n";
#else
    s += "recognizer: none (built without ONNX Runtime; detection only)\n";
#endif
    s += PdfDocument::supported() ? "pdf: poppler " CUDAOCR_POPPLER_VERSION "\n"
                                  : "pdf: none (built without poppler-cpp)\n";
    s += "cuda architectures: " CUDAOCR_CUDA_ARCHS "\n";
    return s;
}

}  // namespace cudaocr
