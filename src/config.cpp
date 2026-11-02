#include "cudaocr/config.hpp"

#include <toml++/toml.hpp>

#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <sstream>

namespace cudaocr {
namespace {

Status err(std::string_view src, const toml::source_region& at, const std::string& msg) {
    return Status::Error(ErrorCode::InvalidArgument,
                         std::string(src) + ":" + std::to_string(at.begin.line) + ":" +
                             std::to_string(at.begin.column) + ": " + msg);
}

// TOML scalar -> Params text (Params parses text on read).
Result<std::string> scalarText(const toml::node& v) {
    if (auto i = v.as_integer()) return std::to_string(i->get());
    if (auto d = v.as_floating_point()) {
        char buf[64];
        auto res = std::to_chars(buf, buf + sizeof buf, d->get());
        return std::string(buf, res.ptr);
    }
    if (auto b = v.as_boolean()) return std::string(b->get() ? "true" : "false");
    if (auto s = v.as_string()) return s->get();
    return Status::Error(ErrorCode::InvalidArgument, "");
}

Result<ComponentSpec> parseSpec(std::string_view src, const toml::node& n,
                                const std::string& where) {
    const toml::table* t = n.as_table();
    if (!t) return err(src, n.source(), where + " must be a table like { name = \"...\" }");
    ComponentSpec spec;
    bool hasName = false;
    for (auto&& [key, value] : *t) {
        const std::string k(key.str());
        if (k == "name") {
            if (!value.is_string())
                return err(src, value.source(), where + ".name must be a string");
            spec.name = value.as_string()->get();
            hasName = true;
            continue;
        }
        auto text = scalarText(value);
        if (!text.ok())
            return err(src, value.source(), where + "." + k + ": use a number, string or boolean");
        spec.params.set(k, text.value());
    }
    if (!hasName) return err(src, n.source(), where + " needs a name = \"...\"");
    return spec;
}

Status parseSpecList(std::string_view src, const toml::node& n, const std::string& where,
                     std::vector<ComponentSpec>& out) {
    const toml::array* a = n.as_array();
    if (!a) return err(src, n.source(), where + " must be an array of { name = ... } tables");
    out.clear();
    for (size_t i = 0; i < a->size(); ++i) {
        auto spec = parseSpec(src, (*a)[i], where + "[" + std::to_string(i) + "]");
        if (!spec.ok()) return spec.status();
        out.push_back(spec.take());
    }
    return Status::Good();
}

// Optional slot: a component table, or `false` to disable it.
Status parseOptional(std::string_view src, const toml::node& n, const std::string& where,
                     std::optional<ComponentSpec>& out) {
    if (auto b = n.as_boolean()) {
        if (b->get()) return err(src, n.source(), where + " = true is ambiguous; give a table");
        out.reset();
        return Status::Good();
    }
    auto spec = parseSpec(src, n, where);
    if (!spec.ok()) return spec.status();
    out = spec.take();
    return Status::Good();
}

// Params text -> TOML literal: bare when it is an int/float/bool literal (parses
// back to the same text), otherwise a quoted string.
std::string literal(const std::string& v) {
    if (v == "true" || v == "false") return v;
    if (!v.empty()) {
        long long i;
        auto [pi, ei] = std::from_chars(v.data(), v.data() + v.size(), i);
        if (ei == std::errc() && pi == v.data() + v.size() && std::to_string(i) == v) return v;
        double d;
        auto [pd, ed] = std::from_chars(v.data(), v.data() + v.size(), d);
        if (ed == std::errc() && pd == v.data() + v.size() && std::isfinite(d)) {
            char buf[64];
            auto res = std::to_chars(buf, buf + sizeof buf, d);
            if (std::string(buf, res.ptr) == v && v.find_first_of(".e") != std::string::npos)
                return v;
        }
    }
    std::string q = "\"";
    for (char c : v) {
        switch (c) {
            case '"':
                q += "\\\"";
                break;
            case '\\':
                q += "\\\\";
                break;
            case '\n':
                q += "\\n";
                break;
            case '\t':
                q += "\\t";
                break;
            default:
                q += c;
        }
    }
    return q + "\"";
}

// Bare key when TOML allows it, else a quoted key.
std::string key(const std::string& k) {
    bool bare = !k.empty();
    for (char c : k)
        bare = bare && (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-');
    if (bare) return k;
    std::string q = "\"";
    for (char c : k) q += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
    return q + "\"";
}

std::string spec(const ComponentSpec& s) {
    std::string out = "{ name = " + literal(s.name);
    for (const auto& [k, v] : s.params.raw()) out += ", " + key(k) + " = " + literal(v);
    return out + " }";
}

std::string specList(const std::vector<ComponentSpec>& list) {
    if (list.empty()) return "[]";
    std::string out = "[\n";
    for (const auto& s : list) out += "  " + spec(s) + ",\n";
    return out + "]";
}

}  // namespace

Result<PipelineConfig> parsePipelineConfig(std::string_view text, std::string_view src,
                                           const PipelineConfig& base) {
    toml::table tbl;
    try {
        tbl = toml::parse(text, src);
    } catch (const toml::parse_error& e) {
        return err(src, e.source(), std::string(e.description()));
    }
    PipelineConfig cfg = base;
    for (auto&& [k, v] : tbl) {
        const std::string name(k.str());
        Status s = Status::Good();
        if (name == "toGray") {
            auto spec = parseSpec(src, v, name);
            if (spec.ok()) cfg.toGray = spec.take();
            s = spec.status();
        } else if (name == "preprocess") {
            s = parseSpecList(src, v, name, cfg.preprocess);
        } else if (name == "binarizePrep") {
            s = parseSpecList(src, v, name, cfg.binarizePrep);
        } else if (name == "binarizer" || name == "detector" || name == "decoder") {
            auto spec = parseSpec(src, v, name);
            if (spec.ok())
                (name == "binarizer"  ? cfg.binarizer
                 : name == "detector" ? cfg.detector
                                      : cfg.decoder) = spec.take();
            s = spec.status();
        } else if (name == "deskewer") {
            s = parseOptional(src, v, name, cfg.deskewer);
        } else if (name == "recognizer") {
            s = parseOptional(src, v, name, cfg.recognizer);
        } else {
            s = err(src, k.source(),
                    "unknown key '" + name +
                        "' (known: toGray, preprocess, binarizePrep, binarizer, deskewer, "
                        "detector, recognizer, decoder)");
        }
        if (!s.ok()) return s;
    }
    return cfg;
}

Result<PipelineConfig> loadPipelineConfig(const std::string& path, const PipelineConfig& base) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status::Error(ErrorCode::IoError, "cannot read config " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return parsePipelineConfig(ss.str(), path, base);
}

std::string toToml(const PipelineConfig& cfg) {
    std::string t = "# cudaocr pipeline config (cudaocr --print-config)\n";
    t += "toGray = " + spec(cfg.toGray) + "\n";
    t += "preprocess = " + specList(cfg.preprocess) + "\n";
    t += "binarizePrep = " + specList(cfg.binarizePrep) + "\n";
    t += "binarizer = " + spec(cfg.binarizer) + "\n";
    t += "deskewer = " + (cfg.deskewer ? spec(*cfg.deskewer) : std::string("false")) + "\n";
    t += "detector = " + spec(cfg.detector) + "\n";
    t += "recognizer = " + (cfg.recognizer ? spec(*cfg.recognizer) : std::string("false")) + "\n";
    t += "decoder = " + spec(cfg.decoder) + "\n";
    return t;
}

}  // namespace cudaocr
