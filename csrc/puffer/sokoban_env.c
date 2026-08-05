/* Standalone demo and smoke test for the Ocean env.
 *
 * Compiles without PufferLib or raylib:
 *     cc -O2 -o sokoban_demo csrc/puffer/sokoban_env.c && ./sokoban_demo
 *
 * Exercises the same c_reset/c_step the vectorised binding calls, so the env
 * logic is verifiable without a PufferLib build present.
 */

#include <stdio.h>
#include <time.h>

#include "sokoban_env.h"

int main(int argc, char** argv) {
    int steps = (argc > 1) ? atoi(argv[1]) : 200000;

    unsigned char observations[SK_OBS_BYTES];
    int actions[1];
    float rewards[1];
    unsigned char terminals[1];

    SokobanEnv env = {0};
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.board_size = 8;
    env.n_crates = 1;
    env.pool_size = 64;
    env.max_steps = 30;
    env.min_len = 4;
    env.max_len = 20;
    env.density = 0.10f;
    env.max_tries = 20000;
    env.rng = 7;

    clock_t t0 = clock();
    sk_build_pool(&env, 13);
    double gen = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("pool: %d levels in %.2fs (%.0f levels/s)\n",
           env.pool_size, gen, env.pool_size / gen);

    c_reset(&env);
    unsigned int rng = 42;
    int episodes = 0, solved = 0;

    t0 = clock();
    for (int i = 0; i < steps; i++) {
        actions[0] = rand_r(&rng) % N_ACTIONS;
        c_step(&env);
        if (terminals[0]) {
            episodes++;
            if (rewards[0] > 0.5f) solved++;
        }
    }
    double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("%d steps in %.2fs -> %.0f steps/s\n", steps, dt, steps / dt);
    printf("%d episodes, %d solved (%.1f%%) by a random policy\n",
           episodes, solved, 100.0 * solved / (episodes ? episodes : 1));
    printf("log: perf=%.3f episode_length=%.1f deadlocks=%.3f n=%.0f\n",
           env.log.perf / (env.log.n ? env.log.n : 1),
           env.log.episode_length / (env.log.n ? env.log.n : 1),
           env.log.deadlocks / (env.log.n ? env.log.n : 1), env.log.n);

    c_close(&env);
    return 0;
}
