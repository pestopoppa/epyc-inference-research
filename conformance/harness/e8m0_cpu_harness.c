// CPU conformance harness — calls the REAL ggml decoders on the vector codes.
//
// WHY THIS IS NOT A TRANSCRIPTION
//   It #includes ggml-impl.h from the FROZEN production tree and calls
//   ggml_e8m0_to_fp32 / ggml_e8m0_to_fp32_half directly. Nothing here reimplements
//   or copies their logic, so if the frozen tree's behaviour ever changes, this
//   harness changes with it and the pinned vectors disagree. That disagreement is
//   the whole point: a harness that restates the implementation can never fail.
//
// READ-ONLY on the production tree. Including a header does not modify it, and
// nothing here is compiled into or linked against any production artifact.
//
// Build (see run_backend_conformance.sh):
//   cc -I<llama>/ggml/include -I<llama>/ggml/src -o e8m0_cpu_harness e8m0_cpu_harness.c
#include "ggml-impl.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static const uint8_t CODES[] = {0, 1, 2, 126, 127, 128, 253, 254, 255};
static const int NCODES = (int)(sizeof(CODES) / sizeof(CODES[0]));

static uint32_t bits_of(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

int main(void) {
    printf("{\n  \"backend\": \"cpu\",\n");
    printf("  \"source\": \"ggml/src/ggml-impl.h (frozen production tree, included not copied)\",\n");

    printf("  \"e8m0_ggml_full\": [\n");
    for (int i = 0; i < NCODES; ++i) {
        printf("    {\"code\": %u, \"bits\": \"0x%08x\"}%s\n",
               CODES[i], bits_of(ggml_e8m0_to_fp32(CODES[i])), i + 1 < NCODES ? "," : "");
    }
    printf("  ],\n");

    printf("  \"e8m0_ggml_half\": [\n");
    for (int i = 0; i < NCODES; ++i) {
        printf("    {\"code\": %u, \"bits\": \"0x%08x\"}%s\n",
               CODES[i], bits_of(ggml_e8m0_to_fp32_half(CODES[i])), i + 1 < NCODES ? "," : "");
    }
    printf("  ]\n}\n");
    return 0;
}
