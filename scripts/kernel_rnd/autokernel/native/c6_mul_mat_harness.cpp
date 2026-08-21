// Standalone C6 witness harness for governed ggml GPU candidates.
//
// This file intentionally lives outside llama.cpp.  It uses only the public
// ggml API and is compiled against the exact candidate build.  The evaluator
// owns every argument and independently validates the emitted JSON sidecar.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct options {
    std::string operation;
    std::string backend;
    std::string type_a;
    std::string sidecar;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    uint64_t seed = 0;
};

struct input_blob {
    std::string name;
    std::vector<uint8_t> bytes;
};

using input_set = std::vector<input_blob>;

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
    if (errno || end == raw || *end != '\0') fail(std::string("invalid ") + label);
    return static_cast<uint64_t>(value);
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) fail("every option requires one value");
        const std::string key = argv[i++];
        const char * value = argv[i];
        if (key == "--operation") result.operation = value;
        else if (key == "--backend") result.backend = value;
        else if (key == "--type-a") result.type_a = value;
        else if (key == "--m") result.m = parse_i64(value, "m");
        else if (key == "--n") result.n = parse_i64(value, "n");
        else if (key == "--k") result.k = parse_i64(value, "k");
        else if (key == "--seed") result.seed = parse_u64(value, "seed");
        else if (key == "--sidecar") result.sidecar = value;
        else fail("unknown option: " + key);
    }
    if ((result.operation != "MUL_MAT" && result.operation != "RMS_NORM"
             && result.operation != "FLASH_ATTN_EXT")
            || result.backend.empty() || result.type_a.empty() || result.sidecar.empty()
            || result.m < 1 || result.n < 1 || result.k < 1) {
        fail("canonical operation, backend, type-a, dimensions, seed, and sidecar are required");
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

template <typename T>
std::vector<uint8_t> raw_bytes(const std::vector<T> & values) {
    std::vector<uint8_t> result(values.size() * sizeof(T));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

std::vector<uint8_t> seeded_f32_bytes(size_t count, uint64_t & state) {
    std::vector<float> values(count);
    for (float & value : values) value = seeded_float(state);
    return raw_bytes(values);
}

std::vector<uint8_t> seeded_f16_bytes(size_t count, uint64_t & state) {
    std::vector<float> source(count);
    for (float & value : source) value = seeded_float(state);
    std::vector<ggml_fp16_t> values(count);
    ggml_fp32_to_fp16_row(source.data(), values.data(), static_cast<int64_t>(count));
    return raw_bytes(values);
}

input_set make_inputs(const options & opts, ggml_type type) {
    uint64_t state = opts.seed;
    if (opts.operation == "MUL_MAT") {
        if (!ggml_is_quantized(type)) fail("MUL_MAT type-a must be quantized");
        if (opts.k % ggml_blck_size(type) != 0) fail("MUL_MAT k is not block aligned");
        std::vector<float> source(static_cast<size_t>(opts.k * opts.m));
        for (float & value : source) value = seeded_float(state);
        std::vector<uint8_t> quantized(ggml_row_size(type, opts.k) * opts.m);
        std::vector<float> imatrix(static_cast<size_t>(opts.k), 1.0f);
        const size_t written = ggml_quantize_chunk(
            type, source.data(), quantized.data(), 0, opts.m, opts.k,
            ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr);
        if (written != quantized.size()) fail("quantization size mismatch");
        return {{"weights", std::move(quantized)},
                {"activations_f32le", seeded_f32_bytes(
                    static_cast<size_t>(opts.k * opts.n), state)}};
    }
    if (opts.operation == "RMS_NORM") {
        if (type != GGML_TYPE_F32 || opts.k != 1) fail("RMS_NORM requires f32 and k=1");
        return {{"activations_f32le", seeded_f32_bytes(
                    static_cast<size_t>(opts.m * opts.n), state)},
                {"scale_f32le", seeded_f32_bytes(static_cast<size_t>(opts.m), state)}};
    }
    if (type != GGML_TYPE_F16 || opts.m != 64 || opts.n != 1) {
        fail("FLASH_ATTN_EXT requires f16 K/V, D64, and Q1");
    }
    constexpr int64_t kv_heads = 2;
    constexpr int64_t q_heads = 14;
    return {{"query_f32le", seeded_f32_bytes(
                static_cast<size_t>(opts.m * opts.n * q_heads), state)},
            {"key_f16le", seeded_f16_bytes(
                static_cast<size_t>(opts.m * opts.k * kv_heads), state)},
            {"value_f16le", seeded_f16_bytes(
                static_cast<size_t>(opts.m * opts.k * kv_heads), state)}};
}

const std::vector<uint8_t> & input(const input_set & inputs, const char * name) {
    for (const auto & item : inputs) if (item.name == name) return item.bytes;
    fail(std::string("missing input witness: ") + name);
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
                            const input_set & data, bool reference) {
    const ggml_init_params params = {
        ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) fail("could not create graph context");
    std::vector<std::pair<ggml_tensor *, const std::vector<uint8_t> *>> bindings;
    ggml_tensor * out = nullptr;
    if (opts.operation == "MUL_MAT") {
        ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), type, opts.k, opts.m);
        ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, opts.k, opts.n);
        out = ggml_mul_mat(ctx.get(), a, b);
        bindings = {{a, &input(data, "weights")},
                    {b, &input(data, "activations_f32le")}};
    } else if (opts.operation == "RMS_NORM") {
        ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, opts.m, opts.n);
        ggml_tensor * scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, opts.m);
        out = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), a, 1.0e-6f), scale);
        bindings = {{a, &input(data, "activations_f32le")},
                    {scale, &input(data, "scale_f32le")}};
    } else {
        constexpr int64_t kv_heads = 2;
        constexpr int64_t q_heads = 14;
        ggml_tensor * q = ggml_new_tensor_4d(
            ctx.get(), GGML_TYPE_F32, opts.m, opts.n, q_heads, 1);
        ggml_tensor * key = ggml_new_tensor_4d(
            ctx.get(), GGML_TYPE_F16, opts.m, opts.k, kv_heads, 1);
        ggml_tensor * value = ggml_new_tensor_4d(
            ctx.get(), GGML_TYPE_F16, opts.m, opts.k, kv_heads, 1);
        out = ggml_flash_attn_ext(ctx.get(), q, key, value, nullptr,
                                  1.0f/std::sqrt(static_cast<float>(opts.m)),
                                  0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        bindings = {{q, &input(data, "query_f32le")},
                    {key, &input(data, "key_f16le")},
                    {value, &input(data, "value_f16le")}};
    }
    ggml_set_name(out, "c6_output");
    if (out->type != GGML_TYPE_F32) fail("governed output is not f32");
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    ggml_backend_ptr backend = new_backend(opts.backend, reference);
    if (!ggml_backend_supports_op(backend.get(), out)) {
        fail("backend does not support governed operation");
    }
    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) fail("could not allocate governed graph");
    for (const auto & binding : bindings) {
        ggml_backend_tensor_set(binding.first, binding.second->data(), 0,
                                binding.second->size());
    }
    std::vector<uint8_t> initialized_output(
        static_cast<size_t>(ggml_nbytes(out)), 0);
    ggml_backend_tensor_set(
        out, initialized_output.data(), 0, initialized_output.size());
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        fail("governed graph compute failed");
    }
    std::vector<float> result(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

std::vector<float> dequantized_weights(const options & opts, ggml_type type,
                                       const input_set & data) {
    const ggml_init_params params = {
        ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) fail("could not create float64-oracle dequant context");
    ggml_tensor * weights = ggml_new_tensor_2d(ctx.get(), type, opts.k, opts.m);
    ggml_tensor * cast = ggml_cast(ctx.get(), weights, GGML_TYPE_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, cast);
    ggml_backend_ptr backend = new_backend(opts.backend, true);
    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) fail("could not allocate float64-oracle dequant graph");
    const auto & raw = input(data, "weights");
    ggml_backend_tensor_set(weights, raw.data(), 0, raw.size());
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        fail("float64-oracle dequant graph failed");
    }
    std::vector<float> result(static_cast<size_t>(ggml_nelements(cast)));
    ggml_backend_tensor_get(cast, result.data(), 0, result.size()*sizeof(float));
    return result;
}

template <typename T>
const T * typed_input(const input_set & data, const char * name);

std::vector<float> quantized_dequantized_activations(
        const options & opts, ggml_type weight_type, const input_set & data) {
    const ggml_type activation_type =
        weight_type == GGML_TYPE_Q5_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
    const float * source = typed_input<float>(data, "activations_f32le");
    std::vector<uint8_t> quantized(
        ggml_row_size(activation_type, opts.k)*opts.n);
    const size_t written = ggml_quantize_chunk(
        activation_type, source, quantized.data(), 0, opts.n, opts.k, nullptr);
    if (written != quantized.size()) {
        fail("float64-oracle activation quantization size mismatch");
    }
    const ggml_init_params params = {
        ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) fail("could not create float64-oracle activation context");
    ggml_tensor * activation = ggml_new_tensor_2d(
        ctx.get(), activation_type, opts.k, opts.n);
    ggml_tensor * cast = ggml_cast(ctx.get(), activation, GGML_TYPE_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, cast);
    ggml_backend_ptr backend = new_backend(opts.backend, true);
    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) fail("could not allocate float64-oracle activation graph");
    ggml_backend_tensor_set(activation, quantized.data(), 0, quantized.size());
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        fail("float64-oracle activation dequant graph failed");
    }
    std::vector<float> result(static_cast<size_t>(opts.k*opts.n));
    ggml_backend_tensor_get(cast, result.data(), 0, result.size()*sizeof(float));
    return result;
}

template <typename T>
const T * typed_input(const input_set & data, const char * name) {
    const auto & raw = input(data, name);
    if (raw.size() % sizeof(T)) fail(std::string("misaligned input: ") + name);
    return reinterpret_cast<const T *>(raw.data());
}

std::vector<double> float64_oracle(const options & opts, ggml_type type,
                                   const input_set & data) {
    if (opts.operation == "MUL_MAT") {
        const std::vector<float> weights = dequantized_weights(opts, type, data);
        const std::vector<float> activations =
            quantized_dequantized_activations(opts, type, data);
        std::vector<double> out(static_cast<size_t>(opts.m*opts.n), 0.0);
        for (int64_t col = 0; col < opts.n; ++col) {
            for (int64_t row = 0; row < opts.m; ++row) {
                double sum = 0.0;
                for (int64_t d = 0; d < opts.k; ++d) {
                    sum += static_cast<double>(weights[row*opts.k + d])
                         * static_cast<double>(activations[col*opts.k + d]);
                }
                out[col*opts.m + row] = sum;
            }
        }
        return out;
    }
    if (opts.operation == "RMS_NORM") {
        const float * activations = typed_input<float>(data, "activations_f32le");
        const float * scale = typed_input<float>(data, "scale_f32le");
        std::vector<double> out(static_cast<size_t>(opts.m*opts.n), 0.0);
        for (int64_t col = 0; col < opts.n; ++col) {
            double sumsq = 0.0;
            for (int64_t row = 0; row < opts.m; ++row) {
                const double value = activations[col*opts.m + row];
                sumsq += value*value;
            }
            const double inverse_rms = 1.0/std::sqrt(
                sumsq/static_cast<double>(opts.m) + 1.0e-6);
            for (int64_t row = 0; row < opts.m; ++row) {
                out[col*opts.m + row] =
                    static_cast<double>(activations[col*opts.m + row])
                    * inverse_rms * static_cast<double>(scale[row]);
            }
        }
        return out;
    }
    constexpr int64_t kv_heads = 2;
    constexpr int64_t q_heads = 14;
    const float * query = typed_input<float>(data, "query_f32le");
    const ggml_fp16_t * key16 = typed_input<ggml_fp16_t>(data, "key_f16le");
    const ggml_fp16_t * value16 = typed_input<ggml_fp16_t>(data, "value_f16le");
    std::vector<float> key(static_cast<size_t>(opts.m*opts.k*kv_heads));
    std::vector<float> value(static_cast<size_t>(opts.m*opts.k*kv_heads));
    ggml_fp16_to_fp32_row(key16, key.data(), static_cast<int64_t>(key.size()));
    ggml_fp16_to_fp32_row(value16, value.data(), static_cast<int64_t>(value.size()));
    std::vector<double> out(static_cast<size_t>(opts.m*q_heads), 0.0);
    const double scale = 1.0/std::sqrt(static_cast<double>(opts.m));
    std::vector<double> logits(static_cast<size_t>(opts.k));
    for (int64_t head = 0; head < q_heads; ++head) {
        const int64_t kv_head = head/(q_heads/kv_heads);
        double maximum = -INFINITY;
        for (int64_t position = 0; position < opts.k; ++position) {
            double dot = 0.0;
            for (int64_t d = 0; d < opts.m; ++d) {
                dot += static_cast<double>(query[head*opts.m + d])
                     * static_cast<double>(key[(kv_head*opts.k + position)*opts.m + d]);
            }
            logits[position] = dot*scale;
            maximum = std::max(maximum, logits[position]);
        }
        double denominator = 0.0;
        for (double & logit : logits) {
            logit = std::exp(logit - maximum);
            denominator += logit;
        }
        for (int64_t d = 0; d < opts.m; ++d) {
            double numerator = 0.0;
            for (int64_t position = 0; position < opts.k; ++position) {
                numerator += logits[position]
                    * static_cast<double>(value[(kv_head*opts.k + position)*opts.m + d]);
            }
            out[head*opts.m + d] = numerator/denominator;
        }
    }
    return out;
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
    return hex_bytes(raw_bytes(values));
}

std::string hex_doubles(const std::vector<double> & values) {
    return hex_bytes(raw_bytes(values));
}

void write_sidecar(const options & opts, ggml_type type, const input_set & data,
                   const std::vector<float> & reference,
                   const std::vector<double> & reference_float64,
                   const std::vector<std::vector<float>> & candidates) {
    std::ofstream stream(opts.sidecar, std::ios::binary | std::ios::trunc);
    if (!stream) fail("could not open sidecar");
    stream << "{\"schema\":\"epyc.autokernel.c6_native_operator_sidecar.v3\""
           << ",\"sequence\":[\"reference\",\"candidate-1\",\"candidate-2\",\"candidate-3\"]"
           << ",\"backend\":\"" << opts.backend << "\""
           << ",\"operation\":\"" << opts.operation << "\""
           << ",\"type_a\":\"" << ggml_type_name(type) << "\""
           << ",\"type_b\":\"f32\",\"output_dtype\":\"f32\""
           << ",\"m\":" << opts.m << ",\"n\":" << opts.n
           << ",\"k\":" << opts.k << ",\"seed\":" << opts.seed
           << ",\"output_elements\":" << reference.size()
           << ",\"input_witness\":{";
    for (size_t i = 0; i < data.size(); ++i) {
        if (i) stream << ',';
        stream << '\"' << data[i].name << "_hex\":\""
               << hex_bytes(data[i].bytes) << '\"';
    }
    stream << "},\"reference_output_f32le_hex\":\"" << hex_floats(reference) << "\""
           << ",\"reference_output_f64le_hex\":\"" << hex_doubles(reference_float64) << "\""
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
        const input_set data = make_inputs(opts, type);
        const std::vector<float> reference = run_once(opts, type, data, true);
        const std::vector<double> reference_float64 = float64_oracle(opts, type, data);
        std::vector<std::vector<float>> candidates;
        for (int i = 0; i < 3; ++i) candidates.push_back(run_once(opts, type, data, false));
        write_sidecar(opts, type, data, reference, reference_float64, candidates);
        ggml_quantize_free();
        return 0;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "c6-native-operator-harness: %s\n", exc.what());
        return 1;
    }
}
