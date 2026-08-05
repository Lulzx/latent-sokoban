/* BFS solver and level generator in C.
 *
 * Port of latent_sokoban/solver.py:bfs_solve and levels.py:generate_level.
 * This is the part that actually buys throughput: generation is rejection
 * sampling with a BFS solve inside the loop, so most of the cost is solving
 * layouts that get thrown away. Measured in Python, high crate counts fall to
 * about one level per second.
 *
 * Unlike the renderer, this does NOT need to be bit-identical to Python. The
 * renderer must match because the evaluation server produces observations
 * with the Python code. Generated levels are only ever training data, and the
 * hidden set is generated server-side -- so what has to match here is the
 * DISTRIBUTION (board size, crate count, wall density, solution-length band),
 * not the particular boards. tests/test_cport.py checks that the C solver
 * agrees with Python on optimal solution LENGTH for shared boards, which is
 * the property the band filter depends on.
 */

#ifndef SOKOBAN_SOLVER_H
#define SOKOBAN_SOLVER_H

#include "sokoban.h"
#include <stdlib.h>

#define MAX_CELLS (MAX_DIM * MAX_DIM)
#define BOX_WORDS 4                 /* 256 cells of bitboard */
#define MAX_SOL 256

typedef struct {
    uint64_t boxes[BOX_WORDS];
    uint16_t player;
} Key;

static inline void key_set(Key *k, int cell) {
    k->boxes[cell >> 6] |= 1ULL << (cell & 63);
}
static inline void key_clear(Key *k, int cell) {
    k->boxes[cell >> 6] &= ~(1ULL << (cell & 63));
}
static inline int key_has(const Key *k, int cell) {
    return (k->boxes[cell >> 6] >> (cell & 63)) & 1ULL;
}
static inline int key_eq(const Key *a, const Key *b) {
    if (a->player != b->player) return 0;
    for (int i = 0; i < BOX_WORDS; i++)
        if (a->boxes[i] != b->boxes[i]) return 0;
    return 1;
}
static inline uint64_t key_hash(const Key *k) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < BOX_WORDS; i++) {
        h ^= k->boxes[i];
        h *= 1099511628211ULL;
    }
    h ^= k->player;
    h *= 1099511628211ULL;
    return h;
}

/* Open-addressed visited set + BFS frontier, sized for the 200k node cap the
 * Python solver uses. */
/* Sized to the search, not to the node cap. Measured: 89% of sampled layouts
 * are unsolvable and exhaust in ~1000 states, so the table holds ~1e3 entries
 * while a 2^19 table spans 21 MB -- every probe a cache miss into cold memory.
 * Overridable so the sweep in the commit message can be reproduced. */
#ifndef TABLE_BITS
#define TABLE_BITS 14
#endif
#define TABLE_SIZE (1 << TABLE_BITS)

/* Grown on demand rather than fixed at the worst case. A fixed table sized
 * for the 200k node cap is a cache disaster for the typical search, which
 * holds about a thousand states: every probe misses into multiple megabytes,
 * and the C port measured SLOWER than Python's dict, which is sized to its
 * contents. Capping instead of growing is worse than slow -- it silently
 * rejects levels whose search is large, which skewed generated solution
 * lengths from a mean of 20.7 down to 14.7. */
typedef struct {
    Key *keys;
    int32_t *parent;
    uint8_t *action;
    /* Occupancy by generation stamp rather than a flag byte. Clearing a flag
     * array costs a 0.5 MB memset per search, and generation runs a fresh
     * search per REJECTED layout -- so the clear dominated the actual solving
     * and the first version of this port was only 2x faster than Python. */
    uint32_t *stamp;
    int32_t *queue;
    int bits, size, qhead, qtail, filled;
    uint32_t gen;
} Search;

static inline void search_alloc(Search *s, int bits) {
    s->bits = bits;
    s->size = 1 << bits;
    s->keys = (Key *)malloc(sizeof(Key) * s->size);
    s->parent = (int32_t *)malloc(sizeof(int32_t) * s->size);
    s->action = (uint8_t *)malloc(s->size);
    s->stamp = (uint32_t *)calloc(s->size, sizeof(uint32_t));
    s->queue = (int32_t *)malloc(sizeof(int32_t) * s->size);
    s->gen = 0;
}

static inline void search_free(Search *s) {
    free(s->keys); free(s->parent); free(s->action);
    free(s->stamp); free(s->queue);
}

static inline void search_grow(Search *s) {
    search_free(s);
    search_alloc(s, s->bits + 2);
}

/* Open addressing has no natural failure mode when full: the probe loop
 * simply never finds a free slot and spins. Give up past this load factor
 * and report the search as unresolved, exactly as the node cap does. */
#define SEARCH_LOAD(s) (((s)->size * 3) / 4)
#define SEARCH_MAX_BITS 20

/* Only the crate that just moved can be newly wedged: its parent state was
 * already screened before being queued. The Python version rescans every
 * crate, which is equivalent but O(cells) per expanded node. */
static inline int sk_corner_at(const Level *lv, int r, int c) {
    if (lv->goals[r][c]) return 0;
    int up = sk_is_wall(lv, r - 1, c), down = sk_is_wall(lv, r + 1, c);
    int left = sk_is_wall(lv, r, c - 1), right = sk_is_wall(lv, r, c + 1);
    return (up || down) && (left || right);
}

static inline int sk_at_goal_bits(const Key *k, const uint64_t *goal_bits) {
    for (int i = 0; i < BOX_WORDS; i++)
        if (k->boxes[i] != goal_bits[i]) return 0;
    return 1;
}

/* Returns solution length, or -1 if unsolvable / node budget exhausted.
 * When `out` is non-NULL the action sequence is written there. */
static inline int sk_bfs(const Level *lv, Search *s, int max_nodes,
                         uint8_t *out) {
    if (++s->gen == 0) {            /* wrapped: clear once, then resume */
        memset(s->stamp, 0, sizeof(uint32_t) * s->size);
        s->gen = 1;
    }
    s->qhead = s->qtail = s->filled = 0;

    uint64_t goal_bits[BOX_WORDS] = {0};
    for (int r = 0; r < lv->h; r++)
        for (int c = 0; c < lv->w; c++)
            if (lv->goals[r][c]) {
                int cell = r * lv->w + c;
                goal_bits[cell >> 6] |= 1ULL << (cell & 63);
            }

    Key start;
    memset(&start, 0, sizeof(start));
    for (int r = 0; r < lv->h; r++)
        for (int c = 0; c < lv->w; c++)
            if (lv->boxes[r][c]) key_set(&start, r * lv->w + c);
    start.player = (uint16_t)(lv->player_r * lv->w + lv->player_c);

    if (sk_at_goal_bits(&start, goal_bits)) return 0;

    uint64_t h = key_hash(&start) & (s->size - 1);
    while (s->stamp[h] == s->gen) h = (h + 1) & (s->size - 1);
    s->filled++;
    s->stamp[h] = s->gen; s->keys[h] = start;
    s->parent[h] = -1; s->action[h] = 0;
    s->queue[s->qtail++] = (int32_t)h;

    int nodes = 0;
    while (s->qhead < s->qtail) {
        int32_t cur = s->queue[s->qhead++];
        if (++nodes > max_nodes) return -1;
        Key key = s->keys[cur];
        int pr = key.player / lv->w, pc = key.player % lv->w;

        for (int a = 0; a < N_ACTIONS; a++) {
            int nr = pr + ACT_DR[a], nc = pc + ACT_DC[a];
            if (sk_is_wall(lv, nr, nc)) continue;
            Key nxt = key;
            int ncell = nr * lv->w + nc;
            int pushed_r = -1, pushed_c = -1;
            if (key_has(&key, ncell)) {
                int br = nr + ACT_DR[a], bc = nc + ACT_DC[a];
                if (sk_is_wall(lv, br, bc)) continue;
                int bcell = br * lv->w + bc;
                if (key_has(&key, bcell)) continue;
                key_clear(&nxt, ncell);
                key_set(&nxt, bcell);
                pushed_r = br; pushed_c = bc;
            }
            nxt.player = (uint16_t)ncell;

            uint64_t slot = key_hash(&nxt) & (s->size - 1);
            int seen = 0;
            while (s->stamp[slot] == s->gen) {
                if (key_eq(&s->keys[slot], &nxt)) { seen = 1; break; }
                slot = (slot + 1) & (s->size - 1);
            }
            if (seen) continue;
            if (++s->filled > SEARCH_LOAD(s)) return -2;  /* grow and retry */

            s->stamp[slot] = s->gen; s->keys[slot] = nxt;
            s->parent[slot] = cur; s->action[slot] = (uint8_t)a;

            if (sk_at_goal_bits(&nxt, goal_bits)) {
                int n = 0;
                int32_t t = (int32_t)slot;
                uint8_t tmp[MAX_SOL];
                while (s->parent[t] != -1 && n < MAX_SOL) {
                    tmp[n++] = s->action[t];
                    t = s->parent[t];
                }
                if (out)
                    for (int i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
                return n;
            }
            /* same pruning as the Python solver: a box off-goal wedged into a
             * corner can never be recovered, so that branch is dead */
            if (pushed_r < 0 || !sk_corner_at(lv, pushed_r, pushed_c)) {
                if (s->qtail >= s->size) return -2;
                s->queue[s->qtail++] = (int32_t)slot;
            }
        }
    }
    return -1;
}

/* Retries with a larger table when the search outgrows it, so a big search is
 * slow rather than silently reported unsolvable. */
static inline int sk_solve(const Level *lv, Search *s, int max_nodes,
                           uint8_t *out) {
    for (;;) {
        int n = sk_bfs(lv, s, max_nodes, out);
        if (n != -2) return n;
        if (s->bits >= SEARCH_MAX_BITS) return -1;
        search_grow(s);
    }
}

/* ------------------------------------------------------------- sampling -- */

/* xoshiro256++ -- reproducible from a seed, and fast enough that generation
 * cost stays in the solver where it belongs. */
typedef struct { uint64_t s[4]; } Rng;

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(Rng *r) {
    uint64_t res = rotl(r->s[0] + r->s[3], 23) + r->s[0];
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = rotl(r->s[3], 45);
    return res;
}

static inline void rng_seed(Rng *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {          /* splitmix64 */
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}

static inline uint32_t rng_below(Rng *r, uint32_t n) {
    return (uint32_t)((rng_next(r) >> 32) * (uint64_t)n >> 32);
}

static inline double rng_double(Rng *r) {
    return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* binomial(n, p) by inversion; n is at most 196 here so counting up is fine */
static inline int rng_binomial(Rng *r, int n, double p) {
    int k = 0;
    for (int i = 0; i < n; i++) if (rng_double(r) < p) k++;
    return k;
}

/* Mirrors levels.py:_sample_layout. Returns 0 on success, -1 if the layout
 * left too few free cells. */
static inline int sk_sample_layout(Rng *rng, Level *lv, int size, int n_boxes,
                                   double wall_density) {
    memset(lv, 0, sizeof(*lv));
    lv->h = lv->w = size;
    for (int i = 0; i < size; i++) {
        lv->walls[0][i] = lv->walls[size - 1][i] = 1;
        lv->walls[i][0] = lv->walls[i][size - 1] = 1;
    }

    int interior[MAX_CELLS], n_int = 0;
    for (int r = 1; r < size - 1; r++)
        for (int c = 1; c < size - 1; c++)
            interior[n_int++] = r * size + c;

    int n_walls = rng_binomial(rng, n_int, wall_density);
    /* partial Fisher-Yates: choice(..., replace=False) */
    int pool[MAX_CELLS];
    memcpy(pool, interior, sizeof(int) * n_int);
    for (int i = 0; i < n_walls; i++) {
        int j = i + (int)rng_below(rng, (uint32_t)(n_int - i));
        int t = pool[i]; pool[i] = pool[j]; pool[j] = t;
        lv->walls[pool[i] / size][pool[i] % size] = 1;
    }

    int freec[MAX_CELLS], n_free = 0;
    for (int i = 0; i < n_int; i++) {
        int cell = interior[i];
        if (!lv->walls[cell / size][cell % size]) freec[n_free++] = cell;
    }
    int need = 2 * n_boxes + 1;
    if (n_free < need) return -1;

    for (int i = 0; i < need; i++) {
        int j = i + (int)rng_below(rng, (uint32_t)(n_free - i));
        int t = freec[i]; freec[i] = freec[j]; freec[j] = t;
    }
    for (int i = 0; i < n_boxes; i++)
        lv->boxes[freec[i] / size][freec[i] % size] = 1;
    for (int i = n_boxes; i < 2 * n_boxes; i++)
        lv->goals[freec[i] / size][freec[i] % size] = 1;
    lv->player_r = freec[2 * n_boxes] / size;
    lv->player_c = freec[2 * n_boxes] % size;
    lv->n_goals = n_boxes;
    return 0;
}

/* Mirrors levels.py:generate_level. Returns solution length, or -1 if no
 * level satisfied the band within max_tries. */
static inline int sk_generate_level(Rng *rng, Search *s, Level *lv, int size,
                                    int n_boxes, double wall_density,
                                    int min_len, int max_len, int max_tries,
                                    uint8_t *sol) {
    for (int t = 0; t < max_tries; t++) {
        if (sk_sample_layout(rng, lv, size, n_boxes, wall_density)) continue;
        int n = sk_solve(lv, s, 200000, sol);
        if (n < 0) continue;
        if (n >= min_len && n <= max_len) return n;
    }
    return -1;
}

#endif /* SOKOBAN_SOLVER_H */
