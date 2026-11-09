// AutoKernel native Ghost Replay no-op interposer.
//
// This is evaluator-owned negative-control code, never candidate code.  Build
// it as a shared library and inject it only into the fresh no-op replay process.
// The parent creates AUTOKERNEL_GHOST_EVENT_FD as a write-only inherited pipe,
// reads fixed-size LaunchEvent records, and binds the resulting library/source
// SHA-256 plus the complete runtime map into NativeGhostReplayPlan.
//
// There is deliberately no dlsym fallback.  Every supported native launch
// surface is neutralized and recorded.  A missing recorder terminates the
// replay rather than silently running an unobserved negative control.
#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>

namespace {

constexpr std::uint32_t kMagic = 0x414b4752;  // "AKGR"
enum Symbol : std::uint32_t {
    kHipLaunchKernel = 1,
    kHipLaunchCooperativeKernel = 2,
    kHipGraphLaunch = 3,
};

struct LaunchEvent {
    std::uint32_t magic;
    std::uint32_t symbol;
    std::uint64_t ordinal;
};

std::atomic<std::uint64_t> g_ordinal{0};

int event_fd() {
    const char* raw = std::getenv("AUTOKERNEL_GHOST_EVENT_FD");
    if (raw == nullptr || *raw == '\0') {
        _exit(126);
    }
    char* end = nullptr;
    errno = 0;
    long parsed = std::strtol(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
        _exit(126);
    }
    return static_cast<int>(parsed);
}

void record(Symbol symbol) {
    LaunchEvent event{
        kMagic,
        static_cast<std::uint32_t>(symbol),
        g_ordinal.fetch_add(1, std::memory_order_relaxed),
    };
    const ssize_t written = ::write(event_fd(), &event, sizeof(event));
    if (written != static_cast<ssize_t>(sizeof(event))) {
        _exit(126);
    }
}

}  // namespace

extern "C" hipError_t hipLaunchKernel(const void*, dim3, dim3, void**, size_t,
                                       hipStream_t) {
    record(kHipLaunchKernel);
    return hipSuccess;
}

extern "C" hipError_t hipLaunchCooperativeKernel(const void*, dim3, dim3, void**,
                                                  unsigned int, hipStream_t) {
    record(kHipLaunchCooperativeKernel);
    return hipSuccess;
}

extern "C" hipError_t hipGraphLaunch(hipGraphExec_t, hipStream_t) {
    record(kHipGraphLaunch);
    return hipSuccess;
}
