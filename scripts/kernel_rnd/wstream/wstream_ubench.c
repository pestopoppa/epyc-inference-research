// WSTREAM-DECODE-UBENCH-CPU: single-core lossless decode throughput for Q8_0 int8 weight-code streams.
// Schemes: static (scalar LUT Huffman, one stream), split (raw low nibble + 32-lane Huffman high nibble),
// tans (table-driven tANS, 4 interleaved states, scalar), lanes32 (32-lane interleaved Huffman, AVX-512 gathers).
// Synthetic i.i.d. streams drawn from a measured 256-bin code histogram (real Q8_0 tensors).
// Decode writes into a 64 KiB L2-resident chunk (models a fused decode->dot); input streams from DRAM.
// Also measures: an AVX-512 VNNI Q8_0-style dot over an L2 chunk (cycles/byte), and the N-thread DRAM read
// bandwidth of the cores in the affinity mask (the per-core decode-bandwidth share).
// Build: gcc -O3 -march=znver5 -mavx512f -mavx512bw -mavx512vnni -pthread wstream_ubench.c -o wstream_ubench
// Usage: wstream_ubench <hist.txt> <raw_MiB> <round_index> <bw_threads>   -> one JSON object on stdout
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#define CH 65536          // symbols per chunk
#define MAXL 11
#define RL 11             // tANS table log
#define TL (1 << RL)
#define PAD 64

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline uint64_t be64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return __builtin_bswap64(v); }

// ---------------- bit writer (MSB-first) ----------------
typedef struct { uint8_t *p; size_t n; uint64_t acc; int nb; } BW;
static inline void bw_put(BW *w, uint32_t v, int k) {
  if (!k) return;
  w->acc = (w->acc << k) | v; w->nb += k;
  while (w->nb >= 8) { w->nb -= 8; w->p[w->n++] = (uint8_t)(w->acc >> w->nb); }
}
static void bw_flush(BW *w) { if (w->nb) { w->p[w->n++] = (uint8_t)(w->acc << (8 - w->nb)); w->nb = 0; } }

// ---------------- Huffman (length-limited, canonical) ----------------
typedef struct { uint8_t len[256]; uint32_t code[256]; uint32_t lut[1 << MAXL]; } Huff;
static void huff_build(Huff *h, const double *p, int ns) {
  int n = 0, idx[256]; double w[512]; int par[512];
  for (int i = 0; i < ns; i++) if (p[i] > 0) idx[n++] = i;
  memset(h->len, 0, sizeof h->len);
  if (n == 1) { h->len[idx[0]] = 1; goto canon; }
  {
    int alive[512], na = n, tot = n;
    for (int i = 0; i < n; i++) { w[i] = p[idx[i]]; alive[i] = i; par[i] = -1; }
    while (na > 1) {
      int a = -1, b = -1;
      for (int k = 0; k < na; k++) { int x = alive[k]; if (a < 0 || w[x] < w[a]) { b = a; a = x; } else if (b < 0 || w[x] < w[b]) b = x; }
      w[tot] = w[a] + w[b]; par[tot] = -1; par[a] = par[b] = tot;
      int m = 0; for (int k = 0; k < na; k++) if (alive[k] != a && alive[k] != b) alive[m++] = alive[k];
      alive[m++] = tot++; na = m;
    }
    for (int i = 0; i < n; i++) { int d = 0, x = i; while (par[x] >= 0) { x = par[x]; d++; } h->len[idx[i]] = d; }
    long K = 0;
    for (int i = 0; i < n; i++) { if (h->len[idx[i]] > MAXL) h->len[idx[i]] = MAXL; K += 1L << (MAXL - h->len[idx[i]]); }
    while (K > (1L << MAXL)) {   // lengthen the longest-but-not-max codes (lowest p first)
      int best = -1;
      for (int i = 0; i < n; i++) { int s = idx[i]; if (h->len[s] < MAXL && (best < 0 || h->len[s] > h->len[best] || (h->len[s] == h->len[best] && p[s] < p[best]))) best = s; }
      K -= 1L << (MAXL - h->len[best] - 1); h->len[best]++;
    }
  }
canon:;
  uint32_t c = 0;
  for (int L = 1; L <= MAXL; L++) { for (int s = 0; s < ns; s++) if (h->len[s] == L) { h->code[s] = c++; } c <<= 1; }
  for (int s = 0; s < ns; s++) if (h->len[s]) {
    int L = h->len[s]; uint32_t base = h->code[s] << (MAXL - L);
    for (uint32_t j = 0; j < (1u << (MAXL - L)); j++) h->lut[base + j] = (uint32_t)s | ((uint32_t)L << 8);
  }
}

// ---------------- tANS ----------------
typedef struct { uint16_t f[256]; uint16_t occ_start[256]; uint16_t occ[TL]; struct { uint16_t nbase; uint8_t sym, nb; } T[TL]; } Tans;
static void tans_build(Tans *t, const double *p) {
  int sum = 0, big = 0;
  for (int s = 0; s < 256; s++) { int f = p[s] > 0 ? (int)(p[s] * TL + 0.5) : 0; if (p[s] > 0 && f < 1) f = 1; t->f[s] = f; sum += f; if (f > t->f[big]) big = s; }
  t->f[big] += TL - sum;
  uint8_t spread[TL]; int pos = 0, step = (TL >> 1) + (TL >> 3) + 3;
  for (int s = 0; s < 256; s++) for (int k = 0; k < t->f[s]; k++) { spread[pos] = s; pos = (pos + step) & (TL - 1); }
  int st = 0; for (int s = 0; s < 256; s++) { t->occ_start[s] = st; st += t->f[s]; }
  int cnt[256] = {0}; int nxt[256];
  for (int s = 0; s < 256; s++) nxt[s] = t->f[s];
  for (int x = 0; x < TL; x++) {
    int s = spread[x]; t->occ[t->occ_start[s] + cnt[s]++] = x;
    int n = nxt[s]++; int nb = RL - (31 - __builtin_clz(n));
    t->T[x].sym = s; t->T[x].nb = nb; t->T[x].nbase = (uint16_t)((n << nb) - TL);
  }
}

// ---------------- encoders (per chunk) ----------------
typedef struct { uint8_t *buf; size_t *off; size_t total; } Enc;   // off[c] = chunk start; lane offsets inside chunk header
static size_t enc_static(const Huff *h, const uint8_t *s, int n, uint8_t *o) {
  BW w = {o, 0, 0, 0}; for (int i = 0; i < n; i++) bw_put(&w, h->code[s[i]], h->len[s[i]]); bw_flush(&w); return w.n;
}
// lanes: header 32 x u32 byte offsets (from chunk start), then lane streams; symbol i -> lane i&31
static size_t enc_lanes(const Huff *h, const uint8_t *s, int n, uint8_t *o, int shift) {
  uint32_t *hdr = (uint32_t *)o; size_t at = 128;
  for (int L = 0; L < 32; L++) {
    hdr[L] = (uint32_t)at; BW w = {o + at, 0, 0, 0};
    for (int i = L; i < n; i += 32) { int v = s[i] >> shift; bw_put(&w, h->code[v], h->len[v]); }
    bw_flush(&w); at += w.n + 4;
  }
  return at;
}
static size_t enc_split(const Huff *h, const uint8_t *s, int n, uint8_t *o) {
  for (int k = 0; k < n / 2; k++) o[k] = (s[2 * k] & 15) | ((s[2 * k + 1] & 15) << 4);
  return n / 2 + enc_lanes(h, s, n, o + n / 2, 4);
}
static size_t enc_tans(const Tans *t, const uint8_t *s, int n, uint8_t *o, uint32_t *pb, uint8_t *pk) {
  uint32_t X[4] = {TL, TL, TL, TL};
  for (int i = n - 1; i >= 0; i--) {
    int sym = s[i], l = i & 3; uint32_t x = X[l], f = t->f[sym]; int k = 0;
    while ((x >> k) >= 2 * f) k++;
    pb[i] = x & ((1u << k) - 1); pk[i] = k;
    X[l] = TL + t->occ[t->occ_start[sym] + ((x >> k) - f)];
  }
  BW w = {o, 0, 0, 0};
  for (int l = 0; l < 4; l++) bw_put(&w, X[l] - TL, RL);
  for (int i = 0; i < n; i++) bw_put(&w, pb[i], pk[i]);
  bw_flush(&w); return w.n;
}

// ---------------- decoders ----------------
static void dec_static(const Huff *h, const uint8_t *in, uint8_t *out, int n) {
  uint64_t pos = 0;
  for (int i = 0; i < n; i++) {
    uint32_t e = h->lut[(be64(in + (pos >> 3)) << (pos & 7)) >> (64 - MAXL)];
    out[i] = (uint8_t)e; pos += e >> 8;
  }
}
static const uint8_t BSW[64] = {3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,
                                3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12};
// decode n symbols from 32 lanes; out gets 8-bit symbols (lane-interleaved order restored)
static inline void dec_lanes_core(const uint32_t *lut, const uint8_t *chunk, uint8_t *out, int n, int store_shift) {
  const uint32_t *hdr = (const uint32_t *)chunk;
  __m512i bsw = _mm512_loadu_si512(BSW);
  __m512i p0 = _mm512_slli_epi32(_mm512_loadu_si512(hdr), 3), p1 = _mm512_slli_epi32(_mm512_loadu_si512(hdr + 16), 3);
  __m512i m7 = _mm512_set1_epi32(7), mff = _mm512_set1_epi32(0xff);
  for (int i = 0; i < n; i += 32) {
    __m512i w0 = _mm512_i32gather_epi32(_mm512_srli_epi32(p0, 3), chunk, 1);
    __m512i w1 = _mm512_i32gather_epi32(_mm512_srli_epi32(p1, 3), chunk, 1);
    w0 = _mm512_sllv_epi32(_mm512_shuffle_epi8(w0, bsw), _mm512_and_si512(p0, m7));
    w1 = _mm512_sllv_epi32(_mm512_shuffle_epi8(w1, bsw), _mm512_and_si512(p1, m7));
    __m512i e0 = _mm512_i32gather_epi32(_mm512_srli_epi32(w0, 32 - MAXL), lut, 4);
    __m512i e1 = _mm512_i32gather_epi32(_mm512_srli_epi32(w1, 32 - MAXL), lut, 4);
    p0 = _mm512_add_epi32(p0, _mm512_srli_epi32(e0, 8)); p1 = _mm512_add_epi32(p1, _mm512_srli_epi32(e1, 8));
    __m512i s0 = _mm512_and_si512(e0, mff), s1 = _mm512_and_si512(e1, mff);
    if (store_shift) { s0 = _mm512_slli_epi32(s0, 4); s1 = _mm512_slli_epi32(s1, 4); }
    _mm_storeu_si128((__m128i *)(out + i), _mm512_cvtepi32_epi8(s0));
    _mm_storeu_si128((__m128i *)(out + i + 16), _mm512_cvtepi32_epi8(s1));
  }
}
static void dec_lanes(const Huff *h, const uint8_t *chunk, uint8_t *out, int n) { dec_lanes_core(h->lut, chunk, out, n, 0); }
static void dec_split(const Huff *h, const uint8_t *chunk, uint8_t *out, int n) {
  dec_lanes_core(h->lut, chunk + n / 2, out, n, 1);   // out = hi<<4
  __m128i m15 = _mm_set1_epi8(15);
  for (int i = 0; i < n; i += 32) {
    __m128i b = _mm_loadu_si128((const __m128i *)(chunk + i / 2));
    __m128i lo = _mm_and_si128(b, m15), hi = _mm_and_si128(_mm_srli_epi16(b, 4), m15);
    __m256i nib = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
    __m256i o = _mm256_loadu_si256((const __m256i *)(out + i));
    _mm256_storeu_si256((__m256i *)(out + i), _mm256_or_si256(o, nib));
  }
}
static void dec_tans(const Tans *t, const uint8_t *in, uint8_t *out, int n) {
  uint64_t pos = 0; uint32_t x[4];
  for (int l = 0; l < 4; l++) { x[l] = (uint32_t)((be64(in + (pos >> 3)) << (pos & 7)) >> (64 - RL)); pos += RL; }
  for (int i = 0; i < n; i += 4) {
    for (int l = 0; l < 4; l++) {
      uint32_t s = x[l]; int nb = t->T[s].nb; out[i + l] = t->T[s].sym;
      uint64_t v = be64(in + (pos >> 3)) << (pos & 7);
      x[l] = t->T[s].nbase + (uint32_t)((v >> 1) >> (63 - nb)); pos += nb;
    }
  }
}


// ---------------- scales_only: raw int8 codes + 32-lane Huffman-coded 8-bit scale index (per-tensor fp16 table) ----
// chunk layout: [lane-coded scale indices for CH/32 blocks][CH raw code bytes]. Decoded output per chunk = CH codes
// (+ CH/32 fp16 scales into a side array). Throughput counts CH*34/32 decoded bytes per chunk (codes + 2-byte scales).
#define NSB (CH / 32)
static uint16_t SCTAB[256];
static size_t enc_scales(const Huff *h, const uint8_t *s, const uint8_t *sidx, uint8_t *o) {
  size_t n = enc_lanes(h, sidx, NSB, o, 0); n = (n + 63) & ~(size_t)63;
  memcpy(o + n, s, CH); return n + CH;
}
static uint16_t SCOUT[NSB];
static void dec_scales(const Huff *h, const uint8_t *chunk, uint8_t *out, size_t codes_at) {
  uint8_t idx[NSB]; dec_lanes_core(h->lut, chunk, idx, NSB, 0);
  for (int b = 0; b < NSB; b++) SCOUT[b] = SCTAB[idx[b]];
  memcpy(out, chunk + codes_at, CH);
}

// ---------------- Q8_0 dot baseline (34-byte blocks, L2-resident) ----------------
static float dot_q8(const uint8_t *w, const int8_t *a, int nblk) {
  __m512 acc = _mm512_setzero_ps();
  for (int b = 0; b + 2 <= nblk; b += 2) {   // two 32-code blocks per 64-byte vector
    __m256i w0 = _mm256_loadu_si256((const __m256i *)(w + 34 * b + 2)), w1 = _mm256_loadu_si256((const __m256i *)(w + 34 * b + 36));
    __m512i wv = _mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1);
    __m512i av = _mm512_loadu_si512(a + 32 * b);
    __m512i aw = _mm512_abs_epi8(wv);
    __mmask64 neg = _mm512_movepi8_mask(wv);
    __m512i sa = _mm512_mask_sub_epi8(av, neg, _mm512_setzero_si512(), av);
    __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sa);
    uint16_t s0, s1; memcpy(&s0, w + 34 * b, 2); memcpy(&s1, w + 34 * b + 34, 2);
    __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(_cvtsh_ss(s0)), _mm512_set1_ps(_cvtsh_ss(s1)));
    acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), sc, acc);
  }
  return _mm512_reduce_add_ps(acc);
}

// ---------------- DRAM read bandwidth (N threads, one per CPU in mask) ----------------
typedef struct { const uint8_t *p; size_t n; int cpu; double sec; uint64_t sink; } BWT;
static void *bw_thr(void *v) {
  BWT *t = v; cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(t->cpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
  __m512i a = _mm512_setzero_si512(); double t0 = now();
  for (int r = 0; r < 3; r++) for (size_t i = 0; i < t->n; i += 256) {
    a = _mm512_xor_si512(a, _mm512_load_si512(t->p + i)); a = _mm512_xor_si512(a, _mm512_load_si512(t->p + i + 64));
    a = _mm512_xor_si512(a, _mm512_load_si512(t->p + i + 128)); a = _mm512_xor_si512(a, _mm512_load_si512(t->p + i + 192));
  }
  t->sec = now() - t0; t->sink = _mm512_reduce_add_epi64(a); return 0;
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s hist.txt raw_MiB round bw_threads\n", argv[0]); return 64; }
  double hist[256], tot = 0; FILE *f = fopen(argv[1], "r"); if (!f) return 66;
  for (int i = 0; i < 256; i++) { if (fscanf(f, "%lf", &hist[i]) != 1) return 65; tot += hist[i]; }
  fclose(f);
  double p[256], phi[16] = {0}, H = 0;
  for (int i = 0; i < 256; i++) { p[i] = hist[i] / tot; phi[i >> 4] += p[i]; if (p[i] > 0) H -= p[i] * __builtin_log2(p[i]); }
  size_t raw = (size_t)atol(argv[2]) << 20; int round = atoi(argv[3]), bwn = atoi(argv[4]);
  size_t nch = raw / CH; raw = nch * CH;
  cpu_set_t cs; sched_getaffinity(0, sizeof cs, &cs); int cpus[1024], ncpu = 0;
  for (int c = 0; c < 1024; c++) if (CPU_ISSET(c, &cs)) cpus[ncpu++] = c;
  { cpu_set_t one; CPU_ZERO(&one); CPU_SET(cpus[0], &one); sched_setaffinity(0, sizeof one, &one); }

  // synthetic stream
  uint8_t *src = aligned_alloc(64, raw); double cdf[256]; uint32_t lutc[1 << 16];
  { double c = 0; for (int i = 0; i < 256; i++) { c += p[i]; cdf[i] = c; } int s = 0;
    for (int j = 0; j < 65536; j++) { double u = (j + 0.5) / 65536; while (s < 255 && cdf[s] < u) s++; lutc[j] = s; } }
  uint64_t xs = 0x9e3779b97f4a7c15ull ^ (uint64_t)round;
  for (size_t i = 0; i < raw; i++) { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; src[i] = (uint8_t)lutc[xs >> 48]; }

  static Huff h8, h4; static Tans tt; huff_build(&h8, p, 256); huff_build(&h4, phi, 16); tans_build(&tt, p);
  const char *names[5] = {"static", "split", "tans", "lanes32", "scales_only"};
  for (int i = 0; i < 256; i++) SCTAB[i] = (uint16_t)(0x2000 + 37 * i);
  size_t *cat = malloc((nch + 1) * sizeof(size_t));
  uint8_t *enc[5]; size_t *off[5], etot[5];
  uint32_t *pb = malloc(CH * 4); uint8_t *pk = malloc(CH);
  for (int k = 0; k < 5; k++) {
    enc[k] = aligned_alloc(64, raw + raw / 4 + nch * 512 + 4096); off[k] = malloc((nch + 1) * sizeof(size_t)); size_t at = 0;
    for (size_t c = 0; c < nch; c++) {
      off[k][c] = at; const uint8_t *s = src + c * CH; uint8_t *o = enc[k] + at; size_t n;
      if (k == 0) n = enc_static(&h8, s, CH, o); else if (k == 1) n = enc_split(&h4, s, CH, o);
      else if (k == 2) n = enc_tans(&tt, s, CH, o, pb, pk); else if (k == 3) n = enc_lanes(&h8, s, CH, o, 0);
      else { uint8_t si[NSB]; for (int b = 0; b < NSB; b++) si[b] = src[(c * CH + b * 32 + 17) % raw]; n = enc_scales(&h8, s, si, o); cat[c] = n - CH; }
      at += (n + PAD + 63) & ~(size_t)63;
    }
    off[k][nch] = at; etot[k] = 0;
    for (size_t c = 0; c < nch; c++) etot[k] += off[k][c + 1] - off[k][c] - PAD;  // approx payload (pad/alignment excluded)
  }
  { etot[4] = raw; for (size_t c = 0; c < nch; c++) etot[4] += cat[c]; }  // coded scales (incl. lane hdr) + raw codes
  uint8_t *outb = aligned_alloc(64, CH + 64);
  // lossless check on every chunk (untimed), then timed pass
  int ok[5];
  for (int k = 0; k < 5; k++) {
    ok[k] = 1;
    for (size_t c = 0; c < nch && ok[k]; c++) {
      const uint8_t *in = enc[k] + off[k][c];
      if (k == 0) dec_static(&h8, in, outb, CH); else if (k == 1) dec_split(&h4, in, outb, CH);
      else if (k == 2) dec_tans(&tt, in, outb, CH); else if (k == 3) dec_lanes(&h8, in, outb, CH);
      else { dec_scales(&h8, in, outb, cat[c]); for (int b = 0; b < NSB; b++) if (SCOUT[b] != SCTAB[src[(c * CH + b * 32 + 17) % raw]]) ok[k] = 0; }
      if (memcmp(outb, src + c * CH, CH)) ok[k] = 0;
    }
  }
  // flush caches between schemes by touching src (raw bytes > L3 share)
  double gbps[5], cyc[5]; uint64_t sink = 0;
  int order[5]; for (int k = 0; k < 5; k++) order[k] = (k + round) % 5;   // rotate scheme order per round
  for (int oi = 0; oi < 5; oi++) {
    int k = order[oi]; for (size_t i = 0; i < raw; i += 64) sink += src[i];
    double t0 = now(); uint64_t c0 = __rdtsc();
    for (size_t c = 0; c < nch; c++) {
      const uint8_t *in = enc[k] + off[k][c];
      if (k == 0) dec_static(&h8, in, outb, CH); else if (k == 1) dec_split(&h4, in, outb, CH);
      else if (k == 2) dec_tans(&tt, in, outb, CH); else if (k == 3) dec_lanes(&h8, in, outb, CH);
      else { dec_scales(&h8, in, outb, cat[c]); sink += SCOUT[c & (NSB - 1)]; }
      sink += outb[c & (CH - 1)];
    }
    uint64_t c1 = __rdtsc(); double t1 = now();
    double rb = k == 4 ? raw * 34.0 / 32 : raw; gbps[k] = rb / (t1 - t0) / 1e9; cyc[k] = (double)(c1 - c0) / rb;
  }
  // Q8_0 dot baseline over an L2-resident 34-byte-block buffer (1088 blocks = 36 KiB)
  int nb = 1088; uint8_t *wq = aligned_alloc(64, nb * 34 + 64); int8_t *av = aligned_alloc(64, nb * 32 + 64);
  for (int i = 0; i < nb * 34; i++) wq[i] = src[i]; for (int i = 0; i < nb * 32; i++) av[i] = (int8_t)src[i + 7];
  for (int b = 0; b < nb; b++) { uint16_t one = 0x3c00; memcpy(wq + 34 * b, &one, 2); }
  float fs = 0; int reps = 20000; double d0 = now(); uint64_t dc0 = __rdtsc();
  for (int r = 0; r < reps; r++) { fs += dot_q8(wq, av, nb); __asm__ volatile("" ::: "memory"); }
  uint64_t dc1 = __rdtsc(); double d1 = now();
  double dot_gbps = (double)reps * nb * 34 / (d1 - d0) / 1e9, dot_cyc = (double)(dc1 - dc0) / ((double)reps * nb * 34);
  double tsc_ghz = (double)(dc1 - dc0) / (d1 - d0) / 1e9;
  // DRAM read bandwidth on the first bwn CPUs of the mask (physical cores expected)
  if (bwn > ncpu) bwn = ncpu;
  size_t per = (size_t)256 << 20; uint8_t *bwbuf = aligned_alloc(64, per * bwn);
  BWT th[1024]; pthread_t tid[1024];
  for (int i = 0; i < bwn; i++) { memset(bwbuf + per * i, i, per); th[i] = (BWT){bwbuf + per * i, per, cpus[i], 0, 0}; }
  double b0 = now(); for (int i = 0; i < bwn; i++) pthread_create(&tid[i], 0, bw_thr, &th[i]);
  for (int i = 0; i < bwn; i++) { pthread_join(tid[i], 0); sink += th[i].sink; }
  double bw_gbs = 3.0 * per * bwn / (now() - b0) / 1e9;

  printf("{\"round\":%d,\"raw_bytes\":%zu,\"entropy_bits\":%.4f,\"tsc_ghz\":%.3f,\"cpu0\":%d,\"bw_threads\":%d,\"dram_read_gbs\":%.2f,"
         "\"per_core_share_gbs\":%.3f,\"dot_q8_gbps\":%.3f,\"dot_q8_tsc_cyc_per_byte\":%.4f,\"schemes\":{",
         round, raw, H, tsc_ghz, cpus[0], bwn, bw_gbs, bw_gbs / bwn, dot_gbps, dot_cyc);
  for (int k = 0; k < 5; k++)
    printf("%s\"%s\":{\"ok\":%s,\"decoded_gbps\":%.3f,\"tsc_cyc_per_byte\":%.4f,\"enc_bytes\":%zu,\"ratio\":%.5f,\"saving\":%.5f}",
           k ? "," : "", names[k], ok[k] ? "true" : "false", gbps[k], cyc[k], etot[k], (double)etot[k] / (k == 4 ? raw * 34.0 / 32 : raw), 1 - (double)etot[k] / (k == 4 ? raw * 34.0 / 32 : raw));
  printf("},\"sink\":%llu,\"fs\":%.1f}\n", (unsigned long long)(sink & 1), fs * 0);
  return 0;
}
