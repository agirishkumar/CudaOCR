#pragma once

// Name -> factory tables, one per component interface. Pipeline::create resolves
// every ComponentSpec in its config through a registry, so adding an
// implementation is one class plus one add<>() call — no Pipeline edits.

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "cudaocr/components/decoder.hpp"
#include "cudaocr/components/deskewer.hpp"
#include "cudaocr/components/formatter.hpp"
#include "cudaocr/components/image_stage.hpp"
#include "cudaocr/components/line_recognizer.hpp"
#include "cudaocr/components/params.hpp"
#include "cudaocr/components/text_detector.hpp"

namespace cudaocr {

struct ComponentSpec {
    std::string name;
    Params params;
};

class ComponentRegistry {
  public:
    template <typename I>
    using Factory = std::function<Result<std::unique_ptr<I>>(const Params&)>;

    // Registry pre-populated with every built-in component (architecture.md §6.1).
    static ComponentRegistry withBuiltins();

    // InvalidArgument if `name` is already registered for interface I.
    template <typename I>
    Status add(const std::string& name, Factory<I> factory) {
        auto& t = table<I>();
        if (t.count(name))
            return Status::Error(ErrorCode::InvalidArgument,
                                 "component '" + name + "' is already registered");
        t.emplace(name, std::move(factory));
        return Status::Good();
    }

    // Unknown name -> InvalidArgument listing the registered choices; factory
    // errors are prefixed with the component name.
    template <typename I>
    Result<std::unique_ptr<I>> make(const ComponentSpec& spec) const {
        const auto& t = table<I>();
        auto it = t.find(spec.name);
        if (it == t.end()) {
            std::string known;
            for (const auto& [n, f] : t) known += (known.empty() ? "" : ", ") + n;
            return Status::Error(ErrorCode::InvalidArgument,
                                 "unknown component '" + spec.name + "' (known: " + known + ")");
        }
        auto made = it->second(spec.params);
        if (!made.ok()) {
            const std::string& msg = made.status().message;
            if (msg.rfind(spec.name, 0) == 0) return made.status();  // already prefixed
            return Status::Error(made.status().code, spec.name + ": " + msg);
        }
        return made;
    }

    template <typename I>
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (const auto& [n, f] : table<I>()) out.push_back(n);
        return out;
    }

  private:
    template <typename I>
    std::map<std::string, Factory<I>>& table() {
        return const_cast<std::map<std::string, Factory<I>>&>(
            static_cast<const ComponentRegistry*>(this)->table<I>());
    }
    template <typename I>
    const std::map<std::string, Factory<I>>& table() const {
        if constexpr (std::is_same_v<I, ImageStage>)
            return imageStages_;
        else if constexpr (std::is_same_v<I, Deskewer>)
            return deskewers_;
        else if constexpr (std::is_same_v<I, TextDetector>)
            return detectors_;
        else if constexpr (std::is_same_v<I, LineRecognizer>)
            return recognizers_;
        else if constexpr (std::is_same_v<I, Decoder>)
            return decoders_;
        else {
            static_assert(std::is_same_v<I, Formatter>, "not a component interface");
            return formatters_;
        }
    }

    std::map<std::string, Factory<ImageStage>> imageStages_;
    std::map<std::string, Factory<Deskewer>> deskewers_;
    std::map<std::string, Factory<TextDetector>> detectors_;
    std::map<std::string, Factory<LineRecognizer>> recognizers_;
    std::map<std::string, Factory<Decoder>> decoders_;
    std::map<std::string, Factory<Formatter>> formatters_;
};

}  // namespace cudaocr
