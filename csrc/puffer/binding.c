/* PufferLib 3.0 Ocean binding.
 *
 * Note the version: 3.0 uses env_binding.h with PyObject-based my_init/my_log.
 * The 4.0 tree on GitHub uses vecenv.h with OBS_SIZE/ACT_SIZES macros and
 * float* actions. The two are not interchangeable, and the wrong one compiles
 * against nothing at all -- check `pufferlib.__version__` before editing.
 *
 * Observation and action spaces are declared in sokoban.py, not here.
 */

#include "sokoban_env.h"

#define Env SokobanEnv
#include "env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_agents = 1;
    env->board_size = unpack(kwargs, "board_size");
    env->n_crates = unpack(kwargs, "n_crates");
    env->pool_size = unpack(kwargs, "pool_size");
    env->max_steps = unpack(kwargs, "max_steps");
    env->min_len = unpack(kwargs, "min_len");
    env->max_len = unpack(kwargs, "max_len");
    /* unpack yields an integer, so density arrives as a percentage */
    env->density = (float)unpack(kwargs, "wall_density") / 100.0f;
    env->max_tries = 20000;
    /* Each instance owns its pool, so worker threads share no mutable state.
     * Seeded off the per-env rng so instances do not all draw one level set. */
    sk_build_pool(env, (uint64_t)env->rng * 2654435761u + 1u);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "deadlocks", log->deadlocks);
    return 0;
}
