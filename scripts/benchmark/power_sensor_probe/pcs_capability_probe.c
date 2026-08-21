// RVP-C4-10: what PC-sampling configurations does the gfx90a agent ACTUALLY report?
// Queries rocprofiler-sdk 0.4.0 (ROCm 6.2) directly — the CLI flag does not exist at 6.2,
// but the library exports the API. Observations only (MEASUREMENT.md).
#include <stdio.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/fwd.h>
const char* rocprofiler_get_status_string(rocprofiler_status_t);
#include <rocprofiler-sdk/pc_sampling.h>

static const char* method_str(int m) {
    switch (m) {
        case ROCPROFILER_PC_SAMPLING_METHOD_NONE: return "NONE";
        case ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC: return "STOCHASTIC";
        case ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP: return "HOST_TRAP";
        default: return "?";
    }
}
static const char* unit_str(int u) {
    switch (u) {
        case ROCPROFILER_PC_SAMPLING_UNIT_NONE: return "NONE";
        case ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS: return "INSTRUCTIONS";
        case ROCPROFILER_PC_SAMPLING_UNIT_CYCLES: return "CYCLES";
        case ROCPROFILER_PC_SAMPLING_UNIT_TIME: return "TIME";
        default: return "?";
    }
}

static rocprofiler_status_t config_cb(const rocprofiler_pc_sampling_configuration_t* configs,
                                      unsigned long num_config, void* user_data) {
    (void)user_data;
    printf("  configurations reported: %lu\n", num_config);
    for (unsigned long i = 0; i < num_config; i++) {
        printf("  [%lu] method=%s unit=%s min_interval=%lu max_interval=%lu flags=0x%lx\n",
               i, method_str(configs[i].method), unit_str(configs[i].unit),
               (unsigned long)configs[i].min_interval, (unsigned long)configs[i].max_interval,
               (unsigned long)configs[i].flags);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

static rocprofiler_status_t agent_cb(rocprofiler_agent_version_t version, const void** agents,
                                     unsigned long num_agents, void* user_data) {
    (void)version; (void)user_data;
    for (unsigned long i = 0; i < num_agents; i++) {
        const rocprofiler_agent_t* a = (const rocprofiler_agent_t*)agents[i];
        if (a->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
        printf("GPU agent: %s (%s)\n", a->name, a->product_name ? a->product_name : "-");
        rocprofiler_status_t st =
            rocprofiler_query_pc_sampling_agent_configurations(a->id, config_cb, (void*)1);
        printf("  query status: %d (%s)\n", (int)st, rocprofiler_get_status_string(st));
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int main(void) {
    rocprofiler_status_t st = rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0, agent_cb, sizeof(rocprofiler_agent_t), NULL);
    printf("agent query status: %d (%s)\n", (int)st, rocprofiler_get_status_string(st));
    return 0;
}
