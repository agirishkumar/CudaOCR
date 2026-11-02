// M3.4: ONNX Runtime is linked and both execution providers work on this machine.
// Model: tests/fixtures/onnx/add.onnx (C = A + B, float32[1,3]).
#include <gtest/gtest.h>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace {

const std::string kModel = std::string(CUDAOCR_ONNX_FIXTURE_DIR) + "/add.onnx";

Ort::Env& env() {
    static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "cudaocr-test");
    return e;
}

std::vector<float> runAdd(Ort::Session& session) {
    std::array<float, 3> a{1.0f, 2.0f, 3.0f}, b{10.0f, 20.0f, 30.0f};
    const std::array<int64_t, 2> shape{1, 3};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(mem, a.data(), a.size(), shape.data(), shape.size()),
        Ort::Value::CreateTensor<float>(mem, b.data(), b.size(), shape.data(), shape.size())};
    const char* inNames[] = {"A", "B"};
    const char* outNames[] = {"C"};
    auto out = session.Run(Ort::RunOptions{nullptr}, inNames, inputs.data(), 2, outNames, 1);
    const float* c = out[0].GetTensorData<float>();
    return {c, c + 3};
}

TEST(OnnxRuntime, CudaAndCpuProvidersAreAvailable) {
    const auto providers = Ort::GetAvailableProviders();
    for (const char* p : {"CPUExecutionProvider", "CUDAExecutionProvider"})
        EXPECT_NE(std::find(providers.begin(), providers.end(), p), providers.end()) << p;
}

TEST(OnnxRuntime, RunsAddOnCpu) {
    Ort::SessionOptions opts;
    Ort::Session session(env(), kModel.c_str(), opts);
    EXPECT_EQ(runAdd(session), (std::vector<float>{11.0f, 22.0f, 33.0f}));
}

TEST(OnnxRuntime, RunsAddOnCudaWithoutCpuFallback) {
    // The Env (ORT's default logger) must exist before any provider-options call;
    // CreateCUDAProviderOptions throws "DefaultLogger ... none has been registered"
    // otherwise. Only visible when this test runs alone (as under ctest).
    (void)env();
    Ort::SessionOptions opts;
    // Fail session creation instead of silently placing nodes on the CPU.
    opts.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    OrtCUDAProviderOptionsV2* cuda = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda));
    opts.AppendExecutionProvider_CUDA_V2(*cuda);
    Ort::GetApi().ReleaseCUDAProviderOptions(cuda);
    try {
        Ort::Session session(env(), kModel.c_str(), opts);
        EXPECT_EQ(runAdd(session), (std::vector<float>{11.0f, 22.0f, 33.0f}));
    } catch (const Ort::Exception& e) {
        FAIL() << "CUDA execution provider failed: " << e.what();
    }
}

}  // namespace
