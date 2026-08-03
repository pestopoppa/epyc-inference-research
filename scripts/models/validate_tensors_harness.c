// Load a GGUF with check_tensors=true and report whether validation accepted it.
//
// WHY NOT llama-cli --check-tensors
//   Because I tried that first and it does not work for this purpose. With a
//   non-TTY stdin, llama-cli enters an interactive loop and emits "> " forever --
//   observed at 312 MILLION lines and 895 MB of log before the timeout fired, with
//   `-no-cnv` set and stdin redirected from /dev/null. That is a CLI-semantics
//   problem, and fighting it to obtain a boolean is the wrong shape.
//
//   The loader API gives exactly the semantic we want and nothing else:
//   llama_model_load_from_file returns NULL when validation rejects the tensors.
//   No token generation, no interactive mode, no unbounded output.
//
// EXIT CODES — three states, deliberately
//   0  PASS   loaded, validation accepted the tensor data
//   4  FAIL   loader returned NULL: validation REJECTED this model
//   5  ERROR  could not run (bad args, missing file). Says nothing about the model.
//
//   FAIL and ERROR must never collapse: reporting "invalid model" because the tool
//   could not run would send someone to re-download a good 38 GB file.
//
// Build/run via validate_model_tensors.sh; it links libllama from the frozen
// production build, read-only, and nothing here is linked into production.
#include "llama.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 5;
    }

    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.check_tensors = true;   // the whole point
    mp.n_gpu_layers  = 0;      // CPU only: validation is CPU-side, and this host serves
    mp.use_mmap      = true;   // validation faults the pages in either way
    mp.vocab_only    = false;  // must read tensor data for validation to mean anything

    struct llama_model* m = llama_model_load_from_file(argv[1], mp);

    if (m == NULL) {
        // The loader rejects on validation failure AND on genuinely malformed files.
        // Both mean "do not serve this"; the caller greps the log to tell them apart.
        printf("{\"result\": \"FAIL\", \"reason\": \"llama_model_load_from_file returned NULL "
               "with check_tensors=true\"}\n");
        llama_backend_free();
        return 4;
    }

    printf("{\"result\": \"PASS\", \"reason\": \"loaded with check_tensors=true\"}\n");
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
