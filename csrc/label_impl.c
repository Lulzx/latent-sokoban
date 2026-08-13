/* Fast distance-to-goal labelling for wm/generate.py, exposed via ctypes.
 *
 * wm/generate.py's Python label() runs a BFS per state (plus one per
 * successor, so up to five), which dominates generation at 3-4 crates
 * (measured 0.5 and 0.16 levels/s). A per-state forward BFS in C was not the
 * fix: the hash table grows to a power of two far larger than the search and
 * thrashes cache, and it was SLOWER than CPython's dict at 4 crates.
 *
 * The right answer is one REVERSE BFS from the goal states per level
 * (csrc/solver.h, sk_fill_dist): the reverse of a push is a pull, so the
 * reversed edges are exactly the reversed forward edges and reverse distance
 * from the goal equals forward distance to the goal. One BFS labels every
 * non-dead state; lookups are then O(1).
 *
 * Build:
 *     cc -O2 -shared -o csrc/liblabelsokoban.dylib csrc/label_impl.c
 *
 * ABI (C-order arrays; boxes is n*BOX_WORDS uint64 bitboards, players are
 * cell indices r*w+c):
 *     void* fill_dist(walls, goals, h, w, max_nodes)   -> opaque DistMap*
 *     int    lookup_dist(handle, boxes, player)        -> dist or -1 (dead)
 *     void   free_dist(handle)
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sokoban.h"
#include "solver.h"

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

void* fill_dist(const uint8_t* walls, const uint8_t* goals, int h, int w,
                int max_nodes) {
    Level lv;
    build_level(&lv, walls, goals, h, w);
    DistMap* m = (DistMap*)calloc(1, sizeof(DistMap));
    distmap_alloc(m, 14);
    sk_fill_dist(&lv, m, max_nodes);
    return m;
}

int lookup_dist(void* handle, const uint64_t* boxes, int32_t player) {
    DistMap* m = (DistMap*)handle;
    Key k;
    memset(&k, 0, sizeof(k));
    for (int wd = 0; wd < BOX_WORDS; wd++) k.boxes[wd] = boxes[wd];
    k.player = (uint16_t)player;
    return sk_dist_lookup(m, &k);
}

void free_dist(void* handle) {
    DistMap* m = (DistMap*)handle;
    distmap_free(m);
    free(m);
}
