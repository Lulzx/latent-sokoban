/* ctypes shim: exposes the C env/renderer so tests/test_cport.py can diff it
 * against the Python implementation state by state. Not the PufferLib
 * binding -- this exists purely so the port is verifiable. */

#include "sokoban.h"

/* Render a board given as ASCII. Returns 0 on success. */
int c_render_ascii(const char *ascii, int show_player, uint8_t *out) {
    Level lv;
    if (sk_from_ascii(&lv, ascii)) return -1;
    State st;
    sk_reset(&lv, &st);
    sk_render(&lv, &st, &DEFAULT_THEME, show_player, out);
    return 0;
}

int c_render_goal_ascii(const char *ascii, uint8_t *out) {
    Level lv;
    if (sk_from_ascii(&lv, ascii)) return -1;
    sk_render_goal(&lv, &DEFAULT_THEME, out);
    return 0;
}

/* Apply `n` actions from the level's initial state, then render.
 * Writes the resulting player position and step count back through the
 * pointers so the test can compare dynamics as well as pixels. */
int c_play_and_render(const char *ascii, const int *actions, int n,
                      uint8_t *out, int *pr, int *pc, int *steps,
                      int *solved, int *pushed_any) {
    Level lv;
    if (sk_from_ascii(&lv, ascii)) return -1;
    State st;
    sk_reset(&lv, &st);
    *pushed_any = 0;
    for (int i = 0; i < n; i++) {
        int pushed = 0;
        sk_step(&lv, &st, actions[i], &pushed);
        if (pushed) *pushed_any = 1;
    }
    sk_render(&lv, &st, &DEFAULT_THEME, 1, out);
    *pr = st.pr; *pc = st.pc; *steps = st.steps;
    *solved = sk_solved(&lv, &st);
    return 0;
}

/* ---- solver / generator, for tests/test_cport.py and benchmarking ---- */
#include "solver.h"
#include <stdlib.h>

static Search *g_search = NULL;

/* Optimal solution length for an ASCII board, or -1. */
int c_solve_ascii(const char *ascii) {
    Level lv;
    if (sk_from_ascii(&lv, ascii)) return -2;
    if (!g_search) { g_search = (Search *)calloc(1, sizeof(Search));
                     search_alloc(g_search, 12); }
    return sk_solve(&lv, g_search, 200000, NULL);
}

/* Generate `count` levels; writes each board's ASCII into `out_ascii`
 * (stride bytes apart) and its optimal length into out_len. Returns how
 * many were produced. */
int c_generate(uint64_t seed, int count, int size, int n_boxes,
               double density, int min_len, int max_len, int max_tries,
               char *out_ascii, int stride, int *out_len) {
    Rng rng; rng_seed(&rng, seed);
    if (!g_search) { g_search = (Search *)calloc(1, sizeof(Search));
                     search_alloc(g_search, 12); }
    Level lv;
    uint8_t sol[MAX_SOL];
    int made = 0;
    while (made < count) {
        int n = sk_generate_level(&rng, g_search, &lv, size, n_boxes, density,
                                  min_len, max_len, max_tries, sol);
        if (n < 0) continue;
        char *dst = out_ascii + (size_t)made * stride;
        int p = 0;
        for (int r = 0; r < lv.h; r++) {
            for (int c = 0; c < lv.w; c++) {
                char ch = ' ';
                int box = lv.boxes[r][c], goal = lv.goals[r][c];
                int ply = (r == lv.player_r && c == lv.player_c);
                if (lv.walls[r][c]) ch = '#';
                else if (box && goal) ch = '*';
                else if (box) ch = '$';
                else if (ply && goal) ch = '+';
                else if (ply) ch = '@';
                else if (goal) ch = '.';
                dst[p++] = ch;
            }
            if (r < lv.h - 1) dst[p++] = '\n';
        }
        dst[p] = '\0';
        out_len[made] = n;
        made++;
    }
    return made;
}

/* Throughput probe: N random steps with a render after each, which is what an
 * RL rollout actually costs. */
double c_bench_steps(const char *ascii, int n_steps, uint64_t seed) {
    Level lv;
    if (sk_from_ascii(&lv, ascii)) return -1;
    State st; sk_reset(&lv, &st);
    Rng rng; rng_seed(&rng, seed);
    static uint8_t frame[IMG_SIZE * IMG_SIZE * 3];
    int pushed;
    for (int i = 0; i < n_steps; i++) {
        sk_step(&lv, &st, (int)rng_below(&rng, 4), &pushed);
        sk_render(&lv, &st, &DEFAULT_THEME, 1, frame);
        if (st.steps > 1000) sk_reset(&lv, &st);
    }
    return (double)frame[0];
}
