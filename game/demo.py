"""Демо-сцена: шум в слоях почвы и небольшой организм в центре поля."""

from __future__ import annotations

import numpy as np

from . import config
from .field import Field


def _coarse_noise(rng: np.random.Generator, height: int, width: int, block: int) -> np.ndarray:
    """Гладкий шум: низкое разрешение -> растяжение -> пара размытий."""
    gh = (height + block - 1) // block
    gw = (width + block - 1) // block
    coarse = rng.random((gh, gw), dtype=np.float32)
    noise = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)[:height, :width]
    for _ in range(3):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) * 0.2
    return noise


def seed_demo(field: Field, seed: int = 12345) -> None:
    """Заполнить почву шумом и поставить организм в центр."""
    rng = np.random.default_rng(seed)
    h, w = field.height, field.width

    organic = _coarse_noise(rng, h, w, block=64)
    energy = _coarse_noise(rng, h, w, block=48)

    field.organic[...] = organic * 120.0 + rng.random((h, w), dtype=np.float32) * 8.0
    field.soil_energy[...] = energy * 100.0 + rng.random((h, w), dtype=np.float32) * 6.0

    # Небольшой организм вокруг центра.
    cx, cy = w // 2, h // 2
    pattern = (
        (0, 0, config.SEED),
        (0, 1, config.STEM),
        (0, 2, config.STEM),
        (0, 3, config.STEM),
        (0, 4, config.STEM),
        (1, 4, config.STEM),
        (2, 4, config.STEM),
        (3, 4, config.STEM),
        (4, 4, config.GREEN),
        (-1, 4, config.STEM),
        (-2, 4, config.STEM),
        (-3, 4, config.STEM),
        (-4, 4, config.ORANGE),
        (0, -1, config.STEM),
        (0, -2, config.STEM),
        (0, -3, config.STEM),
        (1, -3, config.STEM),
        (2, -3, config.STEM),
        (3, -3, config.STEM),
        (4, -3, config.BLUE),
        (-1, -3, config.STEM),
        (-2, -3, config.STEM),
        (-3, -3, config.STEM),
        (-4, -3, config.GREEN),
    )
    for dx, dy, code in pattern:
        y = (cy + dy) % h
        x = (cx + dx) % w
        field.cell_type[y, x] = code
        field.organism_id[y, x] = 1
        if code == config.STEM:
            field.energy[y, x] = 0.0
            field.capacity[y, x] = config.STEM_CAPACITY
        elif code == config.SEED:
            field.energy[y, x] = 80.0
        else:
            field.energy[y, x] = 40.0

    field.touch()
