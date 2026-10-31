#include "llama.h"

#include <stdio.h>

int main(void) {
    const struct llama_model_params params = llama_model_default_params();
    struct llama_expert_cache_stats stats = {0};
    if (params.expert_cache_bytes != 0 || stats.gpu || llama_model_get_expert_cache_stats(NULL, &stats) ||
            llama_model_get_expert_cache_stats(NULL, NULL)) {
        fprintf(stderr, "expert cache C API/defaults failed\n");
        return 1;
    }
    return 0;
}
