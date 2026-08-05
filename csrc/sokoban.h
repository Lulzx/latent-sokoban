/* Sokoban environment and renderer in C.
 *
 * This is a port of latent_sokoban/env.py and latent_sokoban/render.py. The
 * renderer must be PIXEL-IDENTICAL to the Python one, because the evaluation
 * server renders observations with the Python code: any divergence means
 * training on a different distribution than you are scored on, and it would
 * surface as poor generalization rather than as an error. tests/test_cport.py
 * diffs the two over thousands of states and is the only thing that makes
 * this port trustworthy.
 *
 * Scope: the default (noise-free) theme, which is what the shared dataset and
 * the hidden set use. Split B's random_theme() adds Gaussian pixel noise drawn
 * from a numpy Generator, which cannot be reproduced bit-for-bit here; themed
 * rendering stays in Python.
 */

#ifndef SOKOBAN_H
#define SOKOBAN_H

#include <stdint.h>
#include <string.h>

#define IMG_SIZE 64
#define MAX_DIM 16
#define N_ACTIONS 4

typedef struct { int r, g, b; } Color;

typedef struct {
    Color floor, floor_alt, wall, wall_edge;
    Color goal, box, box_edge, box_on_goal, player;
    int checker;
} Theme;

static const Theme DEFAULT_THEME = {
    {222, 214, 186}, {212, 204, 176}, {94, 84, 74}, {64, 56, 48},
    {196, 60, 60}, {176, 122, 54}, {120, 80, 30}, {206, 160, 70},
    {48, 108, 188}, 1
};

typedef struct {
    int h, w;
    uint8_t walls[MAX_DIM][MAX_DIM];
    uint8_t goals[MAX_DIM][MAX_DIM];
    uint8_t boxes[MAX_DIM][MAX_DIM];   /* initial configuration */
    int player_r, player_c;
    int n_goals;
} Level;

typedef struct {
    uint8_t boxes[MAX_DIM][MAX_DIM];
    int pr, pc;
    int steps;
} State;

/* row delta, col delta -- must match latent_sokoban.env.ACTIONS ordering:
 * 0 up, 1 down, 2 left, 3 right. */
static const int ACT_DR[N_ACTIONS] = {-1, 1, 0, 0};
static const int ACT_DC[N_ACTIONS] = {0, 0, -1, 1};

/* ------------------------------------------------------------ dynamics -- */

static inline int sk_is_wall(const Level *lv, int r, int c) {
    if (r < 0 || r >= lv->h || c < 0 || c >= lv->w) return 1;
    return lv->walls[r][c];
}

static inline void sk_reset(const Level *lv, State *st) {
    memcpy(st->boxes, lv->boxes, sizeof(st->boxes));
    st->pr = lv->player_r;
    st->pc = lv->player_c;
    st->steps = 0;
}

static inline int sk_solved(const Level *lv, const State *st) {
    for (int r = 0; r < lv->h; r++)
        for (int c = 0; c < lv->w; c++)
            if (st->boxes[r][c] != lv->goals[r][c]) return 0;
    return 1;
}

/* Mirrors SokobanEnv.step: an invalid move still consumes a step. Returns 1
 * if the player moved, and sets *pushed when a crate was displaced. */
static inline int sk_step(const Level *lv, State *st, int action, int *pushed) {
    int dr = ACT_DR[action], dc = ACT_DC[action];
    int nr = st->pr + dr, nc = st->pc + dc;
    int moved = 0;
    *pushed = 0;

    if (!sk_is_wall(lv, nr, nc)) {
        if (st->boxes[nr][nc]) {
            int br = nr + dr, bc = nc + dc;
            if (!sk_is_wall(lv, br, bc) && !st->boxes[br][bc]) {
                st->boxes[nr][nc] = 0;
                st->boxes[br][bc] = 1;
                st->pr = nr; st->pc = nc;
                moved = 1; *pushed = 1;
            }
        } else {
            st->pr = nr; st->pc = nc;
            moved = 1;
        }
    }
    st->steps += 1;
    return moved;
}

/* ------------------------------------------------------------ renderer -- */

static inline void sk_fill(uint8_t *img, int y0, int x0, int nh, int nw,
                           Color col) {
    if (nh <= 0 || nw <= 0) return;
    for (int y = y0; y < y0 + nh; y++) {
        uint8_t *row = img + (y * IMG_SIZE + x0) * 3;
        for (int x = 0; x < nw; x++) {
            row[x * 3 + 0] = (uint8_t)col.r;
            row[x * 3 + 1] = (uint8_t)col.g;
            row[x * 3 + 2] = (uint8_t)col.b;
        }
    }
}

/* Writes IMG_SIZE*IMG_SIZE*3 bytes. Layout and draw order follow
 * render.py:83-125 exactly -- geometry, then per tile: checker, wall (and
 * skip), goal, box, player. */
static inline void sk_render(const Level *lv, const State *st,
                             const Theme *th, int show_player, uint8_t *out) {
    int cell = IMG_SIZE / (lv->h > lv->w ? lv->h : lv->w);
    int oy = (IMG_SIZE - lv->h * cell) / 2;
    int ox = (IMG_SIZE - lv->w * cell) / 2;

    sk_fill(out, 0, 0, IMG_SIZE, IMG_SIZE, th->floor);

    for (int r = 0; r < lv->h; r++) {
        for (int c = 0; c < lv->w; c++) {
            int y0 = oy + r * cell, x0 = ox + c * cell;
            int odd = th->checker && ((r + c) & 1);

            if (odd) sk_fill(out, y0, x0, cell, cell, th->floor_alt);

            if (lv->walls[r][c]) {
                sk_fill(out, y0, x0, cell, cell, th->wall);
                sk_fill(out, y0, x0, 1, cell, th->wall_edge);              /* top */
                sk_fill(out, y0 + cell - 1, x0, 1, cell, th->wall_edge);   /* bottom */
                sk_fill(out, y0, x0, cell, 1, th->wall_edge);              /* left */
                sk_fill(out, y0, x0 + cell - 1, cell, 1, th->wall_edge);   /* right */
                continue;
            }

            if (lv->goals[r][c]) {
                int m = cell / 4; if (m < 1) m = 1;
                sk_fill(out, y0 + m, x0 + m, cell - 2 * m, cell - 2 * m, th->goal);
                int m2 = m + (cell / 6 < 1 ? 1 : cell / 6);
                if (cell - 2 * m2 > 0) {
                    Color base = odd ? th->floor_alt : th->floor;
                    sk_fill(out, y0 + m2, x0 + m2, cell - 2 * m2, cell - 2 * m2,
                            base);
                }
            }

            if (st->boxes[r][c]) {
                Color col = lv->goals[r][c] ? th->box_on_goal : th->box;
                int e = cell / 8; if (e < 1) e = 1;
                sk_fill(out, y0 + e, x0 + e, cell - 2 * e, cell - 2 * e, col);
                int b = e + 1;
                if (cell - 2 * b > 0) {
                    int inner = cell - 2 * e;
                    sk_fill(out, y0 + e, x0 + e, b - e, inner, th->box_edge);
                    sk_fill(out, y0 + cell - b, x0 + e, b - e, inner, th->box_edge);
                    sk_fill(out, y0 + e, x0 + e, inner, b - e, th->box_edge);
                    sk_fill(out, y0 + e, x0 + cell - b, inner, b - e, th->box_edge);
                }
            }

            if (show_player && r == st->pr && c == st->pc) {
                double ctr = (cell - 1) / 2.0;
                double rad2 = (cell * 0.38) * (cell * 0.38);
                for (int yy = 0; yy < cell; yy++) {
                    for (int xx = 0; xx < cell; xx++) {
                        double dy = yy - ctr, dx = xx - ctr;
                        if (dy * dy + dx * dx <= rad2)
                            sk_fill(out, y0 + yy, x0 + xx, 1, 1, th->player);
                    }
                }
            }
        }
    }
}

/* Goal observation: every crate on a goal, player hidden (render_goal). */
static inline void sk_render_goal(const Level *lv, const Theme *th,
                                  uint8_t *out) {
    State g;
    memcpy(g.boxes, lv->goals, sizeof(g.boxes));
    g.pr = lv->player_r; g.pc = lv->player_c; g.steps = 0;
    sk_render(lv, &g, th, 0, out);
}

/* -------------------------------------------------------------- parsing -- */

/* Parse the same ASCII the Python Level.from_ascii accepts:
 * '#' wall, ' ' floor, '$' box, '.' goal, '@' player, '*' box on goal,
 * '+' player on goal. Returns 0 on success. */
static inline int sk_from_ascii(Level *lv, const char *s) {
    memset(lv, 0, sizeof(*lv));
    int r = 0, c = 0, maxw = 0;
    lv->player_r = -1;
    for (const char *p = s; *p; p++) {
        if (*p == '\n') { if (c > maxw) maxw = c; r++; c = 0; continue; }
        if (r >= MAX_DIM || c >= MAX_DIM) return -1;
        switch (*p) {
            case '#': lv->walls[r][c] = 1; break;
            case '$': lv->boxes[r][c] = 1; break;
            case '.': lv->goals[r][c] = 1; break;
            case '*': lv->boxes[r][c] = 1; lv->goals[r][c] = 1; break;
            case '@': lv->player_r = r; lv->player_c = c; break;
            case '+': lv->player_r = r; lv->player_c = c;
                      lv->goals[r][c] = 1; break;
            default: break;
        }
        c++;
    }
    if (c > maxw) maxw = c;
    if (c > 0) r++;
    lv->h = r; lv->w = maxw;
    for (int i = 0; i < lv->h; i++)
        for (int j = 0; j < lv->w; j++)
            lv->n_goals += lv->goals[i][j];
    return lv->player_r < 0 ? -1 : 0;
}

#endif /* SOKOBAN_H */
