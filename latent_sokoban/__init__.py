"""Shared infrastructure for the Latent Sokoban Challenge.

Environment, renderer, level generator, solver, dataset generator and
evaluation harness. This package is competition-neutral: both competitors
build their agents on top of exactly this code.
"""

from latent_sokoban.env import (
    ACTION_NAMES,
    ACTIONS,
    Level,
    SokobanEnv,
    SokobanState,
)
from latent_sokoban.render import Theme, default_theme, random_theme, render
from latent_sokoban.solver import bfs_solve, is_deadlocked
from latent_sokoban.levels import generate_level

__version__ = "0.1.0"

__all__ = [
    "ACTIONS",
    "ACTION_NAMES",
    "Level",
    "SokobanEnv",
    "SokobanState",
    "Theme",
    "default_theme",
    "random_theme",
    "render",
    "bfs_solve",
    "is_deadlocked",
    "generate_level",
]
