// C5-ISA-2 GCN-Gym hipModuleLaunchKernel replay driver (host code only; no device code in this file).
//
// Replays ONE captured launch of ONE kernel from a code object (the original extracted from libggml-hip.so, or an
// LLM-edited .hsaco) and reports per-launch hipEvent timings. Inputs come from the capture's ".pre" blobs (or a
// deterministic fill); after one correctness launch every allocation is written to <outdir>/a<j>.out so the host side
// (gcn_replay.py) can compare it with the original kernel's in-situ ".post" blob or with another code object's replay.
//
// Build (inside a GPU row only):  /opt/rocm/bin/hipcc -O2 -o gcn_replay_driver gcn_replay_driver.cpp
// Run:    gcn_replay_driver <plan.txt> <outdir> [warmup=5] [iters=50]
// Plan (written by gcn_replay.write_plan; one directive per line, '#' comments):
//   co <path>                       code object to hipModuleLoad
//   kernel <mangled name>
//   grid <x> <y> <z>                in workgroups
//   block <x> <y> <z>
//   lds <bytes>                     dynamic LDS
//   kernarg <hex>                   explicit kernarg segment as captured (pointers re-patched below)
//   arg <offset> <size>             explicit args in order (hipModuleLaunchKernel kernelParams)
//   alloc <j> <bytes> <prefile|-> <seed>
//   ptr <kernarg offset> <j> <delta>   kernarg[offset..+8] := device_base(j) + delta
// Stdout: one JSON line {"ok":true,"times_ms":[...],...}; rc 0 ok, 2 bad plan/IO, 3 HIP error.
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Alloc { size_t bytes = 0; std::string pre; uint64_t seed = 0; void *dev = nullptr; std::vector<unsigned char> init; };
struct Ptr { unsigned off; int j; unsigned long long delta; };

static void fail(int rc, const char *what, const char *detail = "") {
  printf("{\"ok\":false,\"rc\":%d,\"err\":\"%s %s\"}\n", rc, what, detail);
  fflush(stdout);
  exit(rc);
}
#define HIPCHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) fail(3, #x, hipGetErrorString(e_)); } while (0)

static void fill(std::vector<unsigned char> &v, uint64_t seed) {
  // deterministic xorshift64*; values as f32 in [-1, 1) per 4 bytes (fallback when no .pre blob exists)
  uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ull;
  for (size_t i = 0; i + 4 <= v.size(); i += 4) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    float f = (float)((s * 2685821657736338717ull) >> 40) / (float)(1u << 24) * 2.f - 1.f;
    memcpy(&v[i], &f, 4);
  }
}

int main(int argc, char **argv) {
  if (argc < 3) fail(2, "usage", "gcn_replay_driver <plan> <outdir> [warmup] [iters]");
  int warmup = argc > 3 ? atoi(argv[3]) : 5, iters = argc > 4 ? atoi(argv[4]) : 50;
  FILE *pf = fopen(argv[1], "r");
  if (!pf) fail(2, "open plan", argv[1]);
  std::string co, kname; unsigned g[3] = {0, 0, 0}, b[3] = {0, 0, 0}; unsigned lds = 0;
  std::vector<unsigned char> ka; std::vector<std::pair<unsigned, unsigned>> args; std::vector<Alloc> allocs;
  std::vector<Ptr> ptrs;
  char *line = nullptr; size_t cap = 0;
  while (getline(&line, &cap, pf) > 0) {
    char tag[32] = {0};
    if (line[0] == '#' || sscanf(line, "%31s", tag) != 1) continue;
    std::string t(tag); const char *rest = line + strlen(tag);
    if (t == "co" || t == "kernel") {
      char v[8192]; if (sscanf(rest, "%8191s", v) != 1) fail(2, "plan", tag);
      (t == "co" ? co : kname) = v;
    } else if (t == "grid") { if (sscanf(rest, "%u %u %u", &g[0], &g[1], &g[2]) != 3) fail(2, "plan", "grid"); }
    else if (t == "block") { if (sscanf(rest, "%u %u %u", &b[0], &b[1], &b[2]) != 3) fail(2, "plan", "block"); }
    else if (t == "lds") { if (sscanf(rest, "%u", &lds) != 1) fail(2, "plan", "lds"); }
    else if (t == "kernarg") {
      std::string h; for (const char *c = rest; *c; c++) if (*c != ' ' && *c != '\n') h += *c;
      if (h.size() % 2) fail(2, "plan", "odd kernarg hex");
      ka.resize(h.size() / 2);
      for (size_t i = 0; i < ka.size(); i++) ka[i] = (unsigned char)strtoul(h.substr(2 * i, 2).c_str(), nullptr, 16);
    } else if (t == "arg") {
      unsigned o, s; if (sscanf(rest, "%u %u", &o, &s) != 2) fail(2, "plan", "arg"); args.push_back({o, s});
    } else if (t == "alloc") {
      int j; unsigned long long n, seed; char pre[8192];
      if (sscanf(rest, "%d %llu %8191s %llu", &j, &n, pre, &seed) != 4 || j != (int)allocs.size()) fail(2, "plan", "alloc");
      Alloc a; a.bytes = n; a.pre = pre; a.seed = seed; allocs.push_back(a);
    } else if (t == "ptr") {
      Ptr p; if (sscanf(rest, "%u %d %llu", &p.off, &p.j, &p.delta) != 3) fail(2, "plan", "ptr"); ptrs.push_back(p);
    } else fail(2, "plan: unknown directive", tag);
  }
  free(line); fclose(pf);
  if (co.empty() || kname.empty() || !g[0] || !b[0]) fail(2, "plan", "co/kernel/grid/block missing");
  for (auto &a : args) if (a.first + a.second > ka.size()) fail(2, "plan", "arg beyond kernarg");
  for (auto &p : ptrs) {
    if (p.j < 0 || p.j >= (int)allocs.size() || p.off + 8 > ka.size()) fail(2, "plan", "ptr out of range");
    if (p.delta >= allocs[p.j].bytes) fail(2, "plan", "ptr delta beyond allocation");
  }

  // inputs (host side first, so a missing blob is a plan error, not a HIP one)
  for (auto &a : allocs) {
    a.init.assign(a.bytes, 0);
    if (a.pre != "-") {
      FILE *f = fopen(a.pre.c_str(), "rb");
      if (!f) fail(2, "open pre", a.pre.c_str());
      size_t got = fread(a.init.data(), 1, a.bytes, f); int extra = fgetc(f); fclose(f);
      if (got != a.bytes || extra != EOF) fail(2, "pre size mismatch", a.pre.c_str());
    } else fill(a.init, a.seed);
  }

  HIPCHK(hipSetDevice(0));
  hipModule_t mod; hipFunction_t fn;
  HIPCHK(hipModuleLoad(&mod, co.c_str()));
  HIPCHK(hipModuleGetFunction(&fn, mod, kname.c_str()));
  for (auto &a : allocs) { HIPCHK(hipMalloc(&a.dev, a.bytes)); }
  for (auto &p : ptrs) { uint64_t v = (uint64_t)(uintptr_t)allocs[p.j].dev + p.delta; memcpy(&ka[p.off], &v, 8); }
  std::vector<void *> params; for (auto &a : args) params.push_back(&ka[a.first]);
  hipStream_t st; HIPCHK(hipStreamCreate(&st));
  auto upload = [&]() { for (auto &a : allocs) HIPCHK(hipMemcpy(a.dev, a.init.data(), a.bytes, hipMemcpyHostToDevice)); };
  auto launch = [&]() {
    HIPCHK(hipModuleLaunchKernel(fn, g[0], g[1], g[2], b[0], b[1], b[2], lds, st, params.data(), nullptr));
  };

  // 1) correctness: fresh inputs, one launch, dump every allocation
  upload(); launch(); HIPCHK(hipStreamSynchronize(st));
  std::vector<unsigned char> h;
  for (size_t j = 0; j < allocs.size(); j++) {
    h.resize(allocs[j].bytes);
    HIPCHK(hipMemcpy(h.data(), allocs[j].dev, h.size(), hipMemcpyDeviceToHost));
    std::string p = std::string(argv[2]) + "/a" + std::to_string(j) + ".out";
    FILE *f = fopen(p.c_str(), "wb");
    if (!f || fwrite(h.data(), 1, h.size(), f) != h.size()) fail(2, "write out", p.c_str());
    fclose(f);
  }
  // 2) timing: inputs restored, warmup, then one event pair per launch (same measurement as gcn_capture.c in situ)
  upload();
  for (int i = 0; i < warmup; i++) launch();
  HIPCHK(hipStreamSynchronize(st));
  hipEvent_t e0, e1; HIPCHK(hipEventCreate(&e0)); HIPCHK(hipEventCreate(&e1));
  std::vector<float> ms(iters > 0 ? iters : 0);
  for (int i = 0; i < iters; i++) {
    HIPCHK(hipEventRecord(e0, st)); launch(); HIPCHK(hipEventRecord(e1, st));
    HIPCHK(hipEventSynchronize(e1)); HIPCHK(hipEventElapsedTime(&ms[i], e0, e1));
  }
  printf("{\"ok\":true,\"kernel\":\"%s\",\"n_allocs\":%zu,\"warmup\":%d,\"times_ms\":[", kname.c_str(), allocs.size(), warmup);
  for (int i = 0; i < iters; i++) printf("%s%.6f", i ? "," : "", ms[i]);
  printf("]}\n");
  fflush(stdout);
  for (auto &a : allocs) (void)hipFree(a.dev);
  (void)hipModuleUnload(mod);
  return 0;
}
