/* Fast distance-to-goal labelling for wm/generate.py, exposed via ctypes.
 *
 * One reverse BFS from the goal states per level fills the distance-to-goal
 * for every solvable state, after which each query is an O(1) lookup. The
 * first C version stored states in a hash table and was cache-hostile at
 * 3-4 crates; the per-state forward BFS was worse (one BFS per queried state
 * plus one per successor).
 *
 * This version stores distances in a DENSE array indexed by a combinadic rank
 * of the crate positions over the level's FREE cells, times the player cell.
 * For an 8x8 board with ~30 free cells the whole 4-crate state space is
 * C(30,4) * 30 ~= 820k entries (1.6 MB as uint16), so the BFS does direct
 * indexed writes instead of hash probes and runs in single-digit milliseconds
 * per level at every crate count.
 *
 * Assumes an 8x8 board (<=64 cells), which is everything the benchmark and the
 * world model use.
 *
 * Build:
 *     cc -O2 -shared -o csrc/liblabelsokoban.dylib csrc/label_impl.c
 *
 * ABI (C-order; boxes is BOX_WORDS uint64 bitboards, player is a cell index):
 *     void* fill_dist(walls, goals, h, w, max_nodes)   -> opaque handle
 *     int    lookup_dist(handle, boxes, player)        -> dist or -1 (dead)
 *     void   free_dist(handle)
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sokoban.h"

#define MAX_CRATES 4
#define UNVISITED 0xFFFFu

typedef struct {
    int f, k, w;
    uint8_t cell_to_free[64];    /* cell -> free index, 0xFF if wall */
    uint8_t free_to_cell[64];    /* free index -> cell */
    uint64_t comb[65][MAX_CRATES + 1];
    uint64_t n_states;           /* C(f, k) * f */
    uint16_t* dist;
    uint32_t* queue;
    int qhead, qtail;
} DenseMap;

static void build_level(Level* lv, const uint8_t* walls, const uint8_t* goals,
                        int h, int w) {
    memset(lv, 0, sizeof(*lv));
    lv->h = h;
    lv->w = w;
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
            uint8_t gl = goals[r * w + c];
            lv->walls[r][c] = walls[r * w + c];
            lv->goals[r][c] = gl;
            if (gl) lv->n_goals++;
        }
}

static void comb_init(DenseMap* m) {
    m->comb[0][0] = 1;
    for (int r = 1; r <= MAX_CRATES; r++) m->comb[0][r] = 0;
    for (int n = 1; n <= 64; n++) {
        m->comb[n][0] = 1;
        for (int r = 1; r <= MAX_CRATES; r++)
            m->comb[n][r] = m->comb[n - 1][r - 1]
                            + (r <= n - 1 ? m->comb[n - 1][r] : 0);
    }
}

static inline void sort4(uint8_t* c, int k) {
    for (int i = 1; i < k; i++) {
        uint8_t t = c[i];
        int j = i - 1;
        while (j >= 0 && c[j] > t) { c[j + 1] = c[j]; j--; }
        c[j + 1] = t;
    }
}

/* Combinadic rank of the sorted free-index combination c[0..k-1]. */
static inline uint64_t comb_rank(const DenseMap* m, const uint8_t* c, int k) {
    uint64_t r = 0;
    for (int j = 0; j < k; j++) r += m->comb[c[j]][j + 1];
    return r;
}

static void dense_push(DenseMap* m, const uint8_t* cr, int p, uint16_t d) {
    uint64_t idx = comb_rank(m, cr, m->k) * m->f + p;
    if (m->dist[idx] != UNVISITED) return;
    m->dist[idx] = d + 1;
    uint32_t st = (uint32_t)p;
    for (int j = 0; j < m->k; j++) st |= ((uint32_t)cr[j] << (6 * (j + 1)));
    m->queue[m->qtail++] = st;
}

static void dense_fill(DenseMap* m, const Level* lv, int max_nodes) {
    for (int i = 0; i < 64; i++) m->cell_to_free[i] = 0xFF;
    m->f = 0;
    for (int r = 0; r < lv->h; r++)
        for (int c = 0; c < lv->w; c++) {
            int cell = r * lv->w + c;
            if (!lv->walls[r][c]) {
                m->cell_to_free[cell] = (uint8_t)m->f;
                m->free_to_cell[m->f] = (uint8_t)cell;
                m->f++;
            }
        }
    m->k = lv->n_goals;
    m->w = lv->w;
    m->n_states = m->comb[m->f][m->k] * m->f;
    m->dist = (uint16_t*)malloc(m->n_states * sizeof(uint16_t));
    m->queue = (uint32_t*)malloc(m->n_states * sizeof(uint32_t));
    for (uint64_t i = 0; i < m->n_states; i++) m->dist[i] = UNVISITED;
    m->qhead = m->qtail = 0;

    /* seed every goal state: crates on goals, player on a free non-goal cell */
    uint8_t gc[MAX_CRATES];
    int ng = 0;
    for (int r = 0; r < lv->h; r++)
        for (int c = 0; c < lv->w; c++)
            if (lv->goals[r][c]) gc[ng++] = m->cell_to_free[r * lv->w + c];
    sort4(gc, ng);
    uint64_t grank = comb_rank(m, gc, ng);
    for (int p = 0; p < m->f; p++) {
        int cell = m->free_to_cell[p];
        if (lv->goals[cell / lv->w][cell % lv->w]) continue;
        m->dist[grank * m->f + p] = 0;
        uint32_t st = (uint32_t)p;
        for (int j = 0; j < ng; j++) st |= ((uint32_t)gc[j] << (6 * (j + 1)));
        m->queue[m->qtail++] = st;
    }

    int nodes = 0;
    while (m->qhead < m->qtail) {
        if (++nodes > max_nodes) break;
        uint32_t st = m->queue[m->qhead++];
        int p = st & 63;
        uint8_t cr[MAX_CRATES];
        for (int j = 0; j < m->k; j++) cr[j] = (uint8_t)((st >> (6 * (j + 1))) & 63);
        uint16_t d = m->dist[comb_rank(m, cr, m->k) * m->f + p];
        int pr = m->free_to_cell[p];
        int rr = pr / m->w, cc = pr % m->w;

        for (int a = 0; a < N_ACTIONS; a++) {
            int br = rr - ACT_DR[a], bc = cc - ACT_DC[a];
            if (sk_is_wall(lv, br, bc)) continue;
            int bfree = m->cell_to_free[br * m->w + bc];
            int crate_behind = 0;
            for (int j = 0; j < m->k; j++) if (cr[j] == bfree) crate_behind = 1;
            if (crate_behind) continue;

            /* reverse walk: player steps back */
            dense_push(m, cr, bfree, d);

            /* reverse pull: a crate ahead is dragged onto the player's cell */
            int ar = rr + ACT_DR[a], ac = cc + ACT_DC[a];
            if (sk_is_wall(lv, ar, ac)) continue;
            int afree = m->cell_to_free[ar * m->w + ac];
            int crate_ahead = -1;
            for (int j = 0; j < m->k; j++) if (cr[j] == afree) crate_ahead = j;
            if (crate_ahead >= 0) {
                uint8_t ncr[MAX_CRATES];
                memcpy(ncr, cr, m->k);
                ncr[crate_ahead] = (uint8_t)p;
                sort4(ncr, m->k);
                dense_push(m, ncr, bfree, d);
            }
        }
    }
}

static int dense_lookup(const DenseMap* m, const uint64_t* boxes,
                        int32_t player) {
    uint8_t cr[MAX_CRATES];
    int nc = 0;
    for (int cell = 0; cell < 64 && nc < m->k; cell++)
        if (boxes[cell >> 6] & (1ULL << (cell & 63)))
            cr[nc++] = m->cell_to_free[cell];
    sort4(cr, nc);
    uint64_t rank = comb_rank(m, cr, nc);
    int p = m->cell_to_free[player];
    uint16_t d = m->dist[rank * m->f + p];
    return (d == UNVISITED) ? -1 : (int)d;
}

void* fill_dist(const uint8_t* walls, const uint8_t* goals, int h, int w,
                int max_nodes) {
    Level lv;
    build_level(&lv, walls, goals, h, w);
    DenseMap* m = (DenseMap*)calloc(1, sizeof(DenseMap));
    comb_init(m);
    dense_fill(m, &lv, max_nodes);
    return m;
}

int lookup_dist(void* handle, const uint64_t* boxes, int32_t player) {
    return dense_lookup((DenseMap*)handle, boxes, player);
}

void free_dist(void* handle) {
    DenseMap* m = (DenseMap*)handle;
    free(m->dist);
    free(m->queue);
    free(m);
}
