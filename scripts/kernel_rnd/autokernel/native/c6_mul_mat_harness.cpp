// Standalone C6 witness harness for governed ggml MUL_MAT candidates.
//
// This file intentionally lives outside llama.cpp.  It uses only the public
// ggml API and is compiled against the exact candidate build.  The evaluator
// owns every argument and independently validates the emitted JSON sidecar.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct options {
    std::string backend;
    std::string type_a;
    std::string sidecar;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    uint64_t seed = 0;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

int64_t parse_i64(const char * raw, const char * label) {
    char * end = nullptr;
    errno = 0;
    const long long value = std::strtoll(raw, &end, 10);
    if (errno || end == raw || *end != '\0' || value < 1) {
        fail(std::string("invalid ") + label);
    }
    return static_cast<int64_t>(value);
}

uint64_t parse_u64(const char * raw, const char * label) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (errno || end == raw || *end != '\0') {
        fail(std::string("invalid ") + label);
    }
    return static_cast<uint64_t>(value);
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) fail("every option requires one value");
        const std::string key = argv[i++];
        const char * value = argv[i];
        if (key == "--backend") result.backend = value;
        else if (key == "--type-a") result.type_a = value;
        else if (key == "--m") result.m = parse_i64(value, "m");
        else if (key == "--n") result.n = parse_i64(value, "n");
        else if (key == "--k") result.k = parse_i64(value, "k");
        else if (key == "--seed") result.seed = parse_u64(value, "seed");
        else if (key == "--sidecar") result.sidecar = value;
        else fail("unknown option: " + key);
    }
    if (result.backend.empty() || result.type_a.empty() || result.sidecar.empty()
            || result.m < 1 || result.n < 1 || result.k < 1) {
        fail("backend, type-a, dimensions, seed, and sidecar are required");
    }
    return result;
}

ggml_type parse_type(const std::string & value) {
    for (int raw = 0; raw < GGML_TYPE_COUNT; ++raw) {
        const auto type = static_cast<ggml_type>(raw);
        const char * name = ggml_type_name(type);
        if (name != nullptr && value == name) return type;
    }
    fail("unknown ggml type: " + value);
}

uint64_t splitmix64(uint64_t & state) {
    uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

float seeded_float(uint64_t & state) {
    const uint32_t bits = static_cast<uint32_t>(splitmix64(state) >> 40);
    return static_cast<float>(bits) / static_cast<float>(UINT32_C(0x00ffffff))
        * 2.0f - 1.0f;
}

std::string hex_bytes(const std::vector<uint8_t> & bytes) {
    static const char digits[] = "0123456789abcdef";
    std::string result(bytes.size() * 2, '0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        result[2*i] = digits[bytes[i] >> 4];
        result[2*i + 1] = digits[bytes[i] & 15];
    }
    return result;
}

std::string hex_floats(const std::vector<float> & values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return hex_bytes(bytes);
}

struct inputs {
    std::vector<uint8_t> weights;
    std::vector<float> activations;
};

inputs make_inputs(const options & opts, ggml_type type) {
    if (!ggml_is_quantized(type)) fail("type-a must be quantized");
    if (opts.k % ggml_blck_size(type) != 0) fail("k is not type block aligned");
    uint64_t state = opts.seed;
    std::vector<float> source(static_cast<size_t>(opts.k * opts.m));
    std::vector<float> activations(static_cast<size_t>(opts.k * opts.n));
    for (float & value : source) value = seeded_float(state);
    for (float & value : activations) value = seeded_float(state);
    std::vector<uint8_t> quantized(ggml_row_size(type, opts.k) * opts.m);
    std::vector<float> imatrix(static_cast<size_t>(opts.k), 1.0f);
    const size_t written = ggml_quantize_chunk(
        type, source.data(), quantized.data(), 0, opts.m, opts.k,
        ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr);
    if (written != quantized.size()) fail("quantization size mismatch");
    return {std::move(quantized), std::move(activations)};
}

ggml_backend_ptr new_backend(const std::string & name, bool reference) {
    if (reference) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        if (!backend) fail("could not initialize CPU reference backend");
        ggml_backend_cpu_set_n_threads(backend.get(), 1);
        ggml_backend_cpu_set_use_ref(backend.get(), true);
        return backend;
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (name == ggml_backend_dev_name(device)) {
            ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
            if (!backend) fail("could not initialize candidate backend");
            return backend;
        }
    }
    fail("candidate backend was not found: " + name);
}

std::vector<float> run_once(const options & opts, ggml_type type,
                            const inputs & data, bool reference) {
    const ggml_init_params params = {
        ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) fail("could not create graph context");
    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), type, opts.k, opts.m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, opts.k, opts.n);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), a, b);
    ggml_set_name(a, "c6_weights");
    ggml_set_name(b, "c6_activations");
    ggml_set_name(out, "c6_output");
    if (out->type != GGML_TYPE_F32) fail("ggml MUL_MAT output is not f32");
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    ggml_backend_ptr backend = new_backend(opts.backend, reference);
    if (!ggml_backend_supports_op(backend.get(), out)) {
        fail("backend does not support governed MUL_MAT");
    }
    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) fail("could not allocate governed graph");
    ggml_backend_tensor_set(a, data.weights.data(), 0, data.weights.size());
    ggml_backend_tensor_set(b, data.activations.data(), 0,
                            data.activations.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        fail("governed graph compute failed");
    }
    std::vector<float> result(static_cast<size_t>(opts.m * opts.n));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

void write_sidecar(const options & opts, ggml_type type, const inputs & data,
                   const std::vector<float> & reference,
                   const std::vector<std::vector<float>> & candidates) {
    std::ofstream stream(opts.sidecar, std::ios::binary | std::ios::trunc);
    if (!stream) fail("could not open sidecar");
    stream << "{\"schema\":\"epyc.autokernel.c6_native_mul_mat_sidecar.v1\""
           << ",\"sequence\":[\"reference\",\"candidate-1\",\"candidate-2\",\"candidate-3\"]"
           << ",\"backend\":\"" << opts.backend << "\""
           << ",\"operation\":\"MUL_MAT\""
           << ",\"type_a\":\"" << ggml_type_name(type) << "\""
           << ",\"type_b\":\"f32\",\"output_dtype\":\"f32\""
           << ",\"m\":" << opts.m << ",\"n\":" << opts.n
           << ",\"k\":" << opts.k << ",\"seed\":" << opts.seed
           << ",\"input_witness\":{\"weights_hex\":\""
           << hex_bytes(data.weights) << "\",\"activations_f32le_hex\":\""
           << hex_floats(data.activations) << "\"}"
           << ",\"reference_output_f32le_hex\":\"" << hex_floats(reference) << "\""
           << ",\"candidate_outputs_f32le_hex\":[";
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i) stream << ',';
        stream << '\"' << hex_floats(candidates[i]) << '\"';
    }
    stream << "],\"candidate_clone_ids\":[\"candidate-1\",\"candidate-2\",\"candidate-3\"]}\n";
    stream.close();
    if (!stream) fail("could not durably write sidecar");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        ggml_backend_load_all();
        const ggml_type type = parse_type(opts.type_a);
        const inputs data = make_inputs(opts, type);
        const std::vector<float> reference = run_once(opts, type, data, true);
        std::vector<std::vector<float>> candidates;
        for (int i = 0; i < 3; ++i) {
            candidates.push_back(run_once(opts, type, data, false));
        }
        write_sidecar(opts, type, data, reference, candidates);
        ggml_quantize_free();
        return 0;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "c6-mul-mat-harness: %s\n", exc.what());
        return 1;
    }
}
