"""The C port must be pixel-identical to the Python renderer.

The evaluation server renders observations with latent_sokoban/render.py. If
the C renderer differs by even one pixel, an agent trained on C renders is
being scored on a different distribution -- and that shows up as poor
generalization, not as a crash, which is the worst way for a bug to present.
So this diffs whole images across board sizes, crate counts and play depths
rather than spot-checking.
"""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from latent_sokoban.env import ACTIONS, Level, SokobanEnv
from latent_sokoban.levels import generate_level
from latent_sokoban.render import render, render_goal

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "csrc" / "shim.c"
LIB = ROOT / "csrc" / "libsokoban.so"


@pytest.fixture(scope="module")
def lib():
    if not SRC.exists():
        pytest.skip("csrc/shim.c not present")
    out = LIB if LIB.exists() else ROOT / "csrc" / "libsokoban.dylib"
    # Rebuild if the source is newer, so a stale artefact cannot make a
    # broken port look correct.
    if not out.exists() or out.stat().st_mtime < SRC.stat().st_mtime:
        r = subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(out),
                            str(SRC)], capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"cannot build C shim: {r.stderr[:200]}")
    dll = ctypes.CDLL(str(out))
    dll.c_render_ascii.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_uint8)]
    dll.c_render_goal_ascii.argtypes = [ctypes.c_char_p,
                                        ctypes.POINTER(ctypes.c_uint8)]
    dll.c_play_and_render.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8)] + [ctypes.POINTER(ctypes.c_int)] * 5
    return dll


def _buf():
    arr = np.zeros(64 * 64 * 3, dtype=np.uint8)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


# Generated once for the whole module: sampling is rejection-based and gets
# expensive at high crate counts, so regenerating per test made the suite look
# like the C port was slow when it was the Python generator.
@pytest.fixture(scope="module")
def levels():
    rng = np.random.default_rng(4)
    out = []
    for size, boxes, n in ((6, 1, 4), (8, 1, 4), (8, 3, 3), (10, 3, 2)):
        made = 0
        while made < n:
            try:
                lv, sol = generate_level(rng, size=size, n_boxes=boxes,
                                         wall_density=0.10, min_solution_len=3,
                                         max_solution_len=30, max_tries=20000)
            except RuntimeError:
                continue
            out.append((lv, sol))
            made += 1
    return out


def test_initial_render_matches(lib, levels):
    arr, ptr = _buf()
    for lv, _ in levels:
        assert lib.c_render_ascii(lv.to_ascii().encode(), 1, ptr) == 0
        want = render(lv)
        got = arr.reshape(64, 64, 3)
        bad = int((want != got).sum())
        assert bad == 0, (f"{bad} bytes differ on a {lv.height}x{lv.width} "
                          f"board\n{lv.to_ascii()}")


def test_goal_render_matches(lib, levels):
    arr, ptr = _buf()
    for lv, _ in levels:
        assert lib.c_render_goal_ascii(lv.to_ascii().encode(), ptr) == 0
        assert np.array_equal(arr.reshape(64, 64, 3), render_goal(lv))


def test_dynamics_and_render_match_along_play(lib, levels):
    """Walk each level with random actions, comparing pixels and state."""
    arr, ptr = _buf()
    rng = np.random.default_rng(7)
    pr = ctypes.c_int(); pc = ctypes.c_int(); steps = ctypes.c_int()
    solved = ctypes.c_int(); pushed = ctypes.c_int()

    for lv, _ in levels:
        acts = [int(rng.integers(0, 4)) for _ in range(25)]
        env = SokobanEnv(lv, max_steps=10_000)
        env.reset()
        pushed_any = False
        for k in range(1, len(acts) + 1):
            _, _, info = env.step(acts[k - 1])
            pushed_any |= info.pushed
            buf = (ctypes.c_int * k)(*acts[:k])
            assert lib.c_play_and_render(lv.to_ascii().encode(), buf, k, ptr,
                                         pr, pc, steps, solved, pushed) == 0
            assert (pr.value, pc.value) == env.state.player, f"player at step {k}"
            assert steps.value == env.state.steps
            assert bool(solved.value) == env.solved
            assert bool(pushed.value) == pushed_any
            assert np.array_equal(arr.reshape(64, 64, 3),
                                  render(lv, env.state)), f"pixels at step {k}"


def test_solved_state_matches(lib, levels):
    """A solved board is the case where box_on_goal colouring kicks in."""
    arr, ptr = _buf()
    for lv, sol in levels:
        env = SokobanEnv(lv, max_steps=10_000)
        env.reset()
        for a in sol:
            env.step(a)
        assert env.solved
        buf = (ctypes.c_int * len(sol))(*sol)
        pr = ctypes.c_int(); pc = ctypes.c_int(); steps = ctypes.c_int()
        solved = ctypes.c_int(); pushed = ctypes.c_int()
        assert lib.c_play_and_render(lv.to_ascii().encode(), buf, len(sol), ptr,
                                     pr, pc, steps, solved, pushed) == 0
        assert solved.value == 1
        assert np.array_equal(arr.reshape(64, 64, 3), render(lv, env.state))


def test_c_solver_agrees_on_optimal_length(lib, levels):
    """The band filter in level generation depends on this number."""
    lib.c_solve_ascii.argtypes = [ctypes.c_char_p]
    from latent_sokoban.solver import bfs_solve
    for lv, _ in levels:
        want = bfs_solve(lv)
        got = lib.c_solve_ascii(lv.to_ascii().encode())
        assert got == (len(want) if want is not None else -1), lv.to_ascii()


def test_generated_levels_are_solvable_and_in_band(lib):
    """C-generated boards, validated by the PYTHON solver.

    Guards the failure that a fixed-size search table caused: oversized
    searches were reported unsolvable, so hard levels were silently dropped
    and generated solution lengths skewed short. Every board here must be
    genuinely solvable at the length the generator claims.
    """
    lib.c_generate.argtypes = [
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    from latent_sokoban.solver import bfs_solve
    n, stride = 25, 512
    buf = ctypes.create_string_buffer(n * stride)
    lens = (ctypes.c_int * n)()
    got = lib.c_generate(99, n, 8, 3, 0.10, 10, 50, 20000, buf, stride, lens)
    assert got == n
    for i in range(n):
        ascii_ = buf[i * stride:(i + 1) * stride].split(b"\x00")[0].decode()
        sol = bfs_solve(Level.from_ascii(ascii_))
        assert sol is not None, f"unsolvable board generated\n{ascii_}"
        assert len(sol) == lens[i], "claimed length disagrees with Python solver"
        assert 10 <= len(sol) <= 50, "out of the requested band"
