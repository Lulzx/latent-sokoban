#include "sokoban_env.h"
#define OBS_SIZE SK_OBS_BYTES      /* 2 * 64 * 64 * 3: observation then goal */
#define NUM_ATNS 1
#define ACT_SIZES {4}              /* 0 up, 1 down, 2 left, 3 right */
#define OBS_TENSOR_T ByteTensor

#define Env SokobanEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->board_size = dict_get(kwargs, "board_size")->value;
    env->n_crates = dict_get(kwargs, "n_crates")->value;
    env->pool_size = dict_get(kwargs, "pool_size")->value;
    env->max_steps = dict_get(kwargs, "max_steps")->value;
    env->min_len = dict_get(kwargs, "min_len")->value;
    env->max_len = dict_get(kwargs, "max_len")->value;
    env->density = (float)dict_get(kwargs, "wall_density")->value / 100.0f;
    env->max_tries = 20000;
    /* Each env instance gets its own pool so the worker threads never share
     * mutable state; seed by rng so instances see different levels. */
    sk_build_pool(env, (uint64_t)env->rng * 2654435761u + 1u);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "deadlocks", log->deadlocks);
}
