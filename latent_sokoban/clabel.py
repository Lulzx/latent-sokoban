"""ctypes binding to the C distance-to-goal labeller (csrc/label_impl.c).

wm/generate.py's Python ``label`` runs a BFS per state plus one per successor,
which dominates data generation at 3-4 crates. The C side instead does ONE
reverse BFS from the goal states per level (csrc/solver.h, sk_fill_dist),
filling the distance-to-goal for every non-dead state, after which each state
is an O(1) lookup. The C solver agrees with the Python solver on optimal
lengths (tests/test_cport.py), and csrc/verify_label.py checks the labels.

Build the library first:

    cc -O2 -shared -o csrc/liblabelsokoban.dylib csrc/label_impl.c
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from latent_sokoban.env import Level, SokobanState

BOX_WORDS = 4

_LIB = ctypes.CDLL(
    str(Path(__file__).resolve().parents[1] / "csrc" / "liblabelsokoban.dylib")
)
_LIB.fill_dist.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # walls (h*w)
    ctypes.POINTER(ctypes.c_uint8),   # goals (h*w)
    ctypes.c_int, ctypes.c_int,       # h, w
    ctypes.c_int,                     # max_nodes
]
_LIB.fill_dist.restype = ctypes.c_void_p
_LIB.lookup_dist.argtypes = [
    ctypes.c_void_p,                  # handle
    ctypes.POINTER(ctypes.c_uint64),  # boxes bitboards (BOX_WORDS)
    ctypes.c_int32,                   # player cell index
]
_LIB.lookup_dist.restype = ctypes.c_int
_LIB.free_dist.argtypes = [ctypes.c_void_p]
_LIB.free_dist.restype = None


class DistTable:
    """Distance-to-goal for every non-dead state of one level.

    ``dist(state)`` returns the true optimal distance-to-goal as an int, or -1
    if the state is dead (unreachable from the goal). Building the table runs
    one reverse BFS; each lookup is an O(1) hash probe.
    """

    def __init__(self, level: Level, max_nodes: int = 2_000_000):
        h, w = level.height, level.width
        walls = np.zeros((h, w), dtype=np.uint8)
        goals = np.zeros((h, w), dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                walls[r, c] = level.walls[r][c]
                goals[r, c] = 1 if (r, c) in level.goals else 0
        self._w = w
        self._handle = _LIB.fill_dist(
            walls.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            goals.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(h), ctypes.c_int(w), ctypes.c_int(max_nodes),
        )

    def dist(self, state: SokobanState) -> int:
        boxes = np.zeros(BOX_WORDS, dtype=np.uint64)
        for br, bc in state.boxes:
            cell = br * self._w + bc
            boxes[cell >> 6] |= np.uint64(1) << np.uint64(cell & 63)
        player = state.player[0] * self._w + state.player[1]
        return int(_LIB.lookup_dist(
            self._handle,
            boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_int32(player),
        ))

    def close(self) -> None:
        if self._handle:
            _LIB.free_dist(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
