/* Latent Sokoban as a PufferLib Ocean environment.
 *
 * Observation is the pair the agent protocol serves: the current 64x64x3
 * render followed by the goal render, 24576 bytes total. Actions are the
 * same four the server accepts. Reward is +1 on solve and 0 otherwise,
 * matching how the leaderboard scores -- deliberately sparse, because
 * shaping it here would train against a different objective than the one
 * being measured.
 *
 * LEVELS COME FROM A POOL BUILT ONCE AT INIT. Generation is rejection
 * sampling with a BFS solve inside the loop and runs at about 6 levels/s
 * even in C (see csrc/solver.h) -- generating one per reset would cap the
 * env at 6 episodes/s and throw away the 105x the step loop buys. The pool
 * size is the knob trading level diversity against startup time.
 */

#include <stdlib.h>
#include <string.h>

#include "../sokoban.h"
#include "../solver.h"

#define SK_OBS_BYTES (2 * IMG_SIZE * IMG_SIZE * 3)

/* Required struct. Only use floats! */
typedef struct {
    float perf;             /* solved fraction, the leaderboard's metric */
    float score;
    float episode_return;
    float episode_length;
    float deadlocks;        /* crate wedged in a corner off-goal */
    float n;                /* Required as the last field */
} Log;

typedef struct {
    Log log;                        /* Required */
    /* Types must match pufferlib 3.0's env_binding.h: int* for discrete
     * actions, unsigned char* for terminals. Getting these wrong compiles
     * fine and misreads the buffers at runtime. */
    unsigned char* observations;    /* Required */
    int* actions;                   /* Required (discrete) */
    float* rewards;                 /* Required */
    unsigned char* terminals;       /* Required */
    int num_agents;

    /* level pool, shared shape but per-env storage */
    Level* pool;
    int pool_size;
    int board_size;
    int n_crates;
    int max_tries;
    int min_len, max_len;
    float density;

    Level level;
    State state;
    int max_steps;
    int tick;
    int pushed_any;
    unsigned int rng;
} SokobanEnv;

/* Only the first half changes within an episode; the goal is written once at
 * reset. Rendering both every step halved throughput for nothing. */
static inline void sk_write_obs(SokobanEnv* env) {
    sk_render(&env->level, &env->state, &DEFAULT_THEME, 1, env->observations);
}

static inline void sk_write_goal(SokobanEnv* env) {
    sk_render_goal(&env->level, &DEFAULT_THEME,
                   env->observations + IMG_SIZE * IMG_SIZE * 3);
}

static inline int sk_env_deadlocked(SokobanEnv* env) {
    if (!env->pushed_any) return 0;
    for (int r = 0; r < env->level.h; r++)
        for (int c = 0; c < env->level.w; c++)
            if (env->state.boxes[r][c] && sk_corner_at(&env->level, r, c))
                return 1;
    return 0;
}

void add_log(SokobanEnv* env) {
    int solved = sk_solved(&env->level, &env->state);
    env->log.perf += solved ? 1.0f : 0.0f;
    env->log.score += solved ? 1.0f : 0.0f;
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.deadlocks += sk_env_deadlocked(env) ? 1.0f : 0.0f;
    env->log.n++;
}

/* Build the level pool. Called once; not on the hot path. */
static inline void sk_build_pool(SokobanEnv* env, uint64_t seed) {
    env->pool = (Level*)calloc(env->pool_size, sizeof(Level));
    Search* s = (Search*)calloc(1, sizeof(Search));
    search_alloc(s, 12);
    Rng rng;
    rng_seed(&rng, seed);
    uint8_t sol[MAX_SOL];
    for (int i = 0; i < env->pool_size; ) {
        Level lv;
        int n = sk_generate_level(&rng, s, &lv, env->board_size, env->n_crates,
                                  env->density, env->min_len, env->max_len,
                                  env->max_tries, sol);
        if (n < 0) continue;
        env->pool[i] = lv;
        i++;
    }
    search_free(s);
    free(s);
}

/* Required function */
void c_reset(SokobanEnv* env) {
    int idx = rand_r(&env->rng) % env->pool_size;
    env->level = env->pool[idx];
    sk_reset(&env->level, &env->state);
    /* Same budget rule the server uses: three times optimal, floor 30. The
     * pool stores no optimal length, so the configured max_steps stands in. */
    env->tick = 0;
    env->pushed_any = 0;
    /* Deliberately does NOT clear rewards/terminals. c_step calls c_reset
     * after setting them for the episode that just ended, so clearing here
     * would erase the only reward signal the learner ever sees. */
    sk_write_obs(env);
    sk_write_goal(env);
}

/* Required function */
void c_step(SokobanEnv* env) {
    env->tick += 1;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    int action = env->actions[0];
    if (action < 0) action = 0;
    if (action >= N_ACTIONS) action = N_ACTIONS - 1;

    int pushed = 0;
    sk_step(&env->level, &env->state, action, &pushed);
    if (pushed) env->pushed_any = 1;

    if (sk_solved(&env->level, &env->state)) {
        env->rewards[0] = 1.0f;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }
    if (env->tick >= env->max_steps) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }
    sk_write_obs(env);
}

/* Required function. Should handle creating the client on first call.
 * Raylib is only present in a PufferLib build, so the standalone demo in
 * sokoban_env.c compiles without it. */
#ifdef PUFFER_RAYLIB
#include "raylib.h"
void c_render(SokobanEnv* env) {
    if (!IsWindowReady()) {
        InitWindow(512, 512, "Latent Sokoban");
        SetTargetFPS(10);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    int px = 512 / IMG_SIZE;
    for (int y = 0; y < IMG_SIZE; y++) {
        for (int x = 0; x < IMG_SIZE; x++) {
            unsigned char* p = env->observations + (y * IMG_SIZE + x) * 3;
            DrawRectangle(x * px, y * px, px, px,
                          (Color){p[0], p[1], p[2], 255});
        }
    }
    EndDrawing();
}
#else
void c_render(SokobanEnv* env) { (void)env; }
#endif

/* Required function. Do not free observations, actions, rewards, terminals. */
void c_close(SokobanEnv* env) {
    free(env->pool);
    env->pool = NULL;
#ifdef PUFFER_RAYLIB
    if (IsWindowReady()) CloseWindow();
#endif
}
