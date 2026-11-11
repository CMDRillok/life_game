"""Демо-сцена: шум в слоях почвы и несколько небольших организмов."""

from __future__ import annotations

import math

import numpy as np

from . import config
from .field import Field

# Крест вокруг семени: (dx, dy, код). Семя стоит в центре.
_CROSS = (
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

# Дополнительные семена вокруг креста (для нескольких семян в организме).
_EXTRA_SEEDS = ((0, 5), (0, -5), (5, 4), (-5, 4), (5, -3), (-5, -3))


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


def _place_organism(
    field: Field,
    cx: int,
    cy: int,
    org_id: int,
    seeds_per_org: int,
) -> None:
    """Поставить крест организма и, если надо, дополнительные семена."""
    h, w = field.height, field.width
    for dx, dy, code in _CROSS:
        y = (cy + dy) % h
        x = (cx + dx) % w
        field.cell_type[y, x] = code
        field.organism_id[y, x] = org_id
        if code == config.STEM:
            field.energy[y, x] = 0.0
            field.capacity[y, x] = config.STEM_CAPACITY
        elif code == config.SEED:
            field.energy[y, x] = 80.0
        else:
            field.energy[y, x] = 40.0

    for i in range(1, max(1, seeds_per_org)):
        dx, dy = _EXTRA_SEEDS[(i - 1) % len(_EXTRA_SEEDS)]
        y = (cy + dy) % h
        x = (cx + dx) % w
        if field.cell_type[y, x] != config.EMPTY:
            continue
        field.cell_type[y, x] = config.SEED
        field.organism_id[y, x] = org_id
        field.energy[y, x] = 80.0
        field.capacity[y, x] = 0.0


def seed_demo(
    field: Field,
    seed: int = 12345,
    organisms: int = 1,
    seeds_per_org: int = 1,
) -> None:
    """Заполнить почву шумом и расставить организмы по полю.

    По умолчанию один организм в центре — как в тестах и старом демо.
    """
    rng = np.random.default_rng(seed)
    h, w = field.height, field.width

    organic = _coarse_noise(rng, h, w, block=64)
    energy = _coarse_noise(rng, h, w, block=48)

    field.organic[...] = organic * 120.0 + rng.random((h, w), dtype=np.float32) * 8.0
    field.soil_energy[...] = energy * 100.0 + rng.random((h, w), dtype=np.float32) * 6.0

    count = max(1, int(organisms))
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    for index in range(count):
        row = index // cols
        col = index % cols
        cx = int((col + 1) * w / (cols + 1))
        cy = int((row + 1) * h / (rows + 1))
        _place_organism(field, cx, cy, index + 1, seeds_per_org)

    field.touch()
