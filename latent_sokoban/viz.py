"""Minimal dependency-free visualization helpers (PNG writer, contact sheets)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: str | Path, arr: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 array as a PNG. Pure stdlib."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def contact_sheet(
    frames: list[np.ndarray] | np.ndarray,
    cols: int = 10,
    gap: int = 4,
    scale: int = 2,
    gap_value: int = 255,
) -> np.ndarray:
    """Tile equally-sized frames into a grid with gaps, upscaled by `scale`."""
    frames = [np.asarray(f) for f in frames]
    h, w, _ = frames[0].shape
    cols = min(cols, len(frames))
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * h + (rows + 1) * gap,
                     cols * w + (cols + 1) * gap, 3), gap_value, dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        y = gap + r * (h + gap)
        x = gap + c * (w + gap)
        sheet[y:y + h, x:x + w] = f
    if scale > 1:
        sheet = np.kron(sheet, np.ones((scale, scale, 1), dtype=np.uint8))
    return sheet
