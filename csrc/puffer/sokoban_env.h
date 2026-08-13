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

/* Reward design. The original reward (+1 on solve, 0 otherwise) is sparse,
 * and a random policy on 8x8 Sokoban solves ~0% of episodes -- so PPO has no
 * gradient to learn from, which is why the distilled warm start drifted to
 * random (commit 4a4480f). The fix is potential-based reward shaping:
 * reward = Phi(s') - Phi(s) + (per-step penalty) + (solve bonus), where Phi
 * is the negative sum of each off-goal crate's free-grid distance to its
 * nearest goal, with a large penalty for a crate wedged in a corner.
 *
 * Potential-based shaping preserves the optimal policy for ANY Phi (the
 * result is exact for F = gamma*Phi(s') - Phi(s); gamma ~ 1 makes the
 * correction negligible), so this does not "train against a different
 * objective" -- it gives the learner a dense signal down the same objective.
 * The solver is used here at training time only, which docs/RULES.md permits.
 *
 * SK_SHAPED=0 restores the sparse reward for A/B comparison. */
#ifndef SK_SHAPED
#define SK_SHAPED 1
#endif
#ifndef SK_STEP_PENALTY
#define SK_STEP_PENALTY 0.01f
#endif
#ifndef SK_DEADLOCK_PENALTY
#define SK_DEADLOCK_PENALTY 50.0f
#endif

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

    /* reward shaping: free-grid distance from each cell to the nearest goal
     * (computed once per reset) and the potential of the current state */
    uint8_t goal_dist[MAX_CELLS];
    float pot;
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

/* Multi-source BFS from every goal over the free grid, giving each cell the
 * shortest walk to a goal IGNORING crates and the player. This is the
 * standard admissible relaxation for Sokoban: cheap (<= a few hundred cell
 * visits), dense (defined for every cell), and monotone in crate movement.
 * Cells unreachable from any goal stay 255 (walls; ignored by the caller). */
static inline void sk_goal_dist(const Level* lv, uint8_t* out) {
    int w = lv->w;
    int q[MAX_CELLS], qh = 0, qt = 0;
    for (int i = 0; i < MAX_CELLS; i++) out[i] = 255;
    for (int r = 0; r < lv->h; r++)
        for (int c = 0; c < lv->w; c++)
            if (lv->goals[r][c]) {
                int cell = r * w + c;
                out[cell] = 0;
                q[qt++] = cell;
            }
    while (qh < qt) {
        int cell = q[qh++];
        int r = cell / w, c = cell % w;
        uint8_t d = (uint8_t)(out[cell] + 1);
        for (int a = 0; a < N_ACTIONS; a++) {
            int nr = r + ACT_DR[a], nc = c + ACT_DC[a];
            if (sk_is_wall(lv, nr, nc)) continue;
            int n = nr * w + nc;
            if (out[n] == 255) { out[n] = d; q[qt++] = n; }
        }
    }
}

/* Phi(s) = -sum over off-goal crates of goal_dist[crate], with a large extra
 * penalty for a crate wedged in a corner (irreversible). 0 at the goal. */
static inline float sk_potential(const Level* lv, const State* st,
                                 const uint8_t* gd) {
    float p = 0.0f;
    for (int r = 0; r < lv->h; r++)
        for (int c = 0; c < lv->w; c++)
            if (st->boxes[r][c] && !lv->goals[r][c]) {
                int cell = r * lv->w + c;
                if (gd[cell] != 255) p -= gd[cell];
                if (sk_corner_at(lv, r, c)) p -= SK_DEADLOCK_PENALTY;
            }
    return p;
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
    sk_goal_dist(&env->level, env->goal_dist);
    env->pot = sk_potential(&env->level, &env->state, env->goal_dist);
    /* Deliberately does NOT clear rewards/terminals. c_step calls c_reset
     * after setting them for the episode that just ended, so clearing here
     * would erase the only reward signal the learner ever sees. */
    sk_write_obs(env);
    sk_write_goal(env);
}

/* Required function */
void c_step(SokobanEnv* env) {
    env->tick += 1;
    env->rewards[0] = -SK_STEP_PENALTY;
    env->terminals[0] = 0;

    int action = env->actions[0];
    if (action < 0) action = 0;
    if (action >= N_ACTIONS) action = N_ACTIONS - 1;

    int pushed = 0;
    sk_step(&env->level, &env->state, action, &pushed);
    if (pushed) env->pushed_any = 1;

    if (SK_SHAPED) {
        float new_pot = sk_potential(&env->level, &env->state, env->goal_dist);
        env->rewards[0] += new_pot - env->pot;
        env->pot = new_pot;
    }

    if (sk_solved(&env->level, &env->state)) {
        env->rewards[0] += 1.0f;
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
