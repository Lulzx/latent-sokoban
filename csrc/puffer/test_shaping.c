/* Verify the shaped reward over a known optimal solution.
 * cc -O2 -o /tmp/test_shaping csrc/puffer/test_shaping.c && /tmp/test_shaping */
#include <stdio.h>
#include <stdlib.h>
#include "sokoban_env.h"

int main(void) {
    Rng rng;
    rng_seed(&rng, 13);
    Search* s = (Search*)calloc(1, sizeof(Search));
    search_alloc(s, 12);
    Level lv;
    uint8_t sol[MAX_SOL];
    int n = sk_generate_level(&rng, s, &lv, 8, 1, 0.10, 4, 20, 20000, sol);
    if (n < 0) { printf("no level\n"); return 1; }
    printf("solution len %d\n", n);

    SokobanEnv env = {0};
    env.level = lv;
    sk_reset(&lv, &env.state);
    sk_goal_dist(&lv, env.goal_dist);
    float start = sk_potential(&lv, &env.state, env.goal_dist);
    printf("start potential %.1f (negative total crate->goal distance)\n", start);

    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        int pushed = 0;
        float old_pot = sk_potential(&lv, &env.state, env.goal_dist);
        sk_step(&lv, &env.state, sol[i], &pushed);
        float new_pot = sk_potential(&lv, &env.state, env.goal_dist);
        float r = -SK_STEP_PENALTY + (new_pot - old_pot);
        if (sk_solved(&lv, &env.state)) r += 1.0f;
        total += r;
        printf("  step %2d a=%d push=%d  pot %+6.1f -> %+6.1f  r=%+7.2f\n",
               i, sol[i], pushed, old_pot, new_pot, r);
    }
    printf("total reward over optimal solve: %+.2f  "
           "(dense, positive; sparse was a single +1)\n", total);
    search_free(s);
    free(s);
    return 0;
}
