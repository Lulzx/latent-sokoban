"""Pure-numpy renderer producing 64x64 RGB uint8 observations.

The default theme is what the shared training set uses. random_theme()
produces visually perturbed themes for Split B (visual generalization):
different floor/wall colours, sprite tints, checker patterns and pixel
noise, while keeping the underlying state fully recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latent_sokoban.env import Level, SokobanState

IMG_SIZE = 64


@dataclass(frozen=True)
class Theme:
    floor: tuple[int, int, int] = (222, 214, 186)
    floor_alt: tuple[int, int, int] = (212, 204, 176)  # checker second colour
    wall: tuple[int, int, int] = (94, 84, 74)
    wall_edge: tuple[int, int, int] = (64, 56, 48)
    goal: tuple[int, int, int] = (196, 60, 60)
    box: tuple[int, int, int] = (176, 122, 54)
    box_edge: tuple[int, int, int] = (120, 80, 30)
    box_on_goal: tuple[int, int, int] = (206, 160, 70)
    player: tuple[int, int, int] = (48, 108, 188)
    checker: bool = True
    noise_std: float = 0.0


def default_theme() -> Theme:
    return Theme()


def random_theme(rng: np.random.Generator, noise: bool = True) -> Theme:
    """Sample a Split-B style visual perturbation of the default theme."""

    def jitter(color, amount=70):
        c = np.array(color, dtype=np.int32)
        c += rng.integers(-amount, amount + 1, size=3)
        return tuple(int(v) for v in np.clip(c, 20, 235))

    base = Theme()
    floor = jitter(base.floor)
    return Theme(
        floor=floor,
        floor_alt=jitter(floor, 18),
        wall=jitter(base.wall),
        wall_edge=jitter(base.wall_edge, 30),
        goal=jitter(base.goal, 45),
        box=jitter(base.box, 45),
        box_edge=jitter(base.box_edge, 30),
        box_on_goal=jitter(base.box_on_goal, 45),
        player=jitter(base.player, 45),
        checker=bool(rng.integers(0, 2)),
        noise_std=float(rng.uniform(0.0, 6.0)) if noise else 0.0,
    )


def render(
    level: Level,
    state: SokobanState | None = None,
    theme: Theme | None = None,
    show_player: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render a state as a (64, 64, 3) uint8 image.

    With state=None the level's initial configuration is rendered.
    Goal observations use state=env.goal_state() and show_player=False.
    Noise (if the theme has any) is drawn from rng; pass a seeded rng for
    deterministic evaluation.
    """
    theme = theme or default_theme()
    boxes = state.boxes if state is not None else level.boxes
    player = state.player if state is not None else level.player

    h, w = level.height, level.width
    cell = IMG_SIZE // max(h, w)
    oy = (IMG_SIZE - h * cell) // 2
    ox = (IMG_SIZE - w * cell) // 2

    img = np.empty((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    img[:] = theme.floor

    for r in range(h):
        for c in range(w):
            y0, x0 = oy + r * cell, ox + c * cell
            tile = img[y0 : y0 + cell, x0 : x0 + cell]
            pos = (r, c)
            if theme.checker and (r + c) % 2:
                tile[:] = theme.floor_alt
            if level.walls[r][c]:
                tile[:] = theme.wall
                tile[0, :] = theme.wall_edge
                tile[-1, :] = theme.wall_edge
                tile[:, 0] = theme.wall_edge
                tile[:, -1] = theme.wall_edge
                continue
            if pos in level.goals:
                m = max(1, cell // 4)
                tile[m:-m, m:-m] = theme.goal
                m2 = m + max(1, cell // 6)
                if cell - 2 * m2 > 0:
                    base = theme.floor_alt if (theme.checker and (r + c) % 2) else theme.floor
                    tile[m2:-m2, m2:-m2] = base
            if pos in boxes:
                color = theme.box_on_goal if pos in level.goals else theme.box
                e = max(1, cell // 8)
                tile[e:-e, e:-e] = color
                b = e + 1
                if cell - 2 * b > 0:
                    tile[e:b, e:-e] = theme.box_edge
                    tile[-b:-e, e:-e] = theme.box_edge
                    tile[e:-e, e:b] = theme.box_edge
                    tile[e:-e, -b:-e] = theme.box_edge
            if show_player and pos == player:
                yy, xx = np.mgrid[0:cell, 0:cell]
                cy = cx = (cell - 1) / 2
                mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= (cell * 0.38) ** 2
                tile[mask] = theme.player

    if theme.noise_std > 0:
        rng = rng or np.random.default_rng(0)
        img += rng.normal(0.0, theme.noise_std, size=img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)


def render_goal(level: Level, theme: Theme | None = None,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Goal observation: all boxes on goals, player hidden."""
    goal_state = SokobanState(frozenset(level.goals), level.player)
    return render(level, goal_state, theme, show_player=False, rng=rng)


__all__ = ["Theme", "default_theme", "random_theme", "render", "render_goal", "IMG_SIZE"]
