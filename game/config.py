"""Константы проекта: размеры, коды клеток, цвета, режимы просмотра."""

from __future__ import annotations

import numpy as np

# --- поле ---
WIDTH = 2000
HEIGHT = 2000
TILE = 24  # пикселей на клетку при zoom == 1

# --- окно ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# --- камера ---
MIN_ZOOM = 0.001
MAX_ZOOM = 64.0
ZOOM_STEP = 1.15
KEY_PAN_SPEED = 900.0  # пикселей экрана в секунду
RESIZE_STEP = 64

# --- коды клеток ---
EMPTY = 0
GREEN = 1
ORANGE = 2
BLUE = 3
STEM = 4
SEED = 5

CELL_NAMES = {
    EMPTY: "пусто",
    GREEN: "зелёная",
    ORANGE: "оранжевая",
    BLUE: "синяя",
    STEM: "стебель",
    SEED: "семя",
}

CELL_COLORS = {
    EMPTY: (24, 26, 28),
    GREEN: (76, 175, 80),
    ORANGE: (255, 152, 0),
    BLUE: (33, 150, 243),
    STEM: (150, 154, 158),
    SEED: (245, 245, 245),
}

# Палитра для векторного рендера клеток: код -> RGB.
CELL_PALETTE = np.zeros((max(CELL_COLORS) + 1, 3), dtype=np.uint8)
for _code, _color in CELL_COLORS.items():
    CELL_PALETTE[_code] = _color

# --- режимы просмотра ---
VIEW_CELLS = 0
VIEW_ORGANIC = 1
VIEW_ENERGY = 2

VIEW_NAMES = {
    VIEW_CELLS: "клетки",
    VIEW_ORGANIC: "органика",
    VIEW_ENERGY: "энергия",
}

VIEW_ORDER = (VIEW_CELLS, VIEW_ORGANIC, VIEW_ENERGY)

# --- оформление ---
GRID_COLOR = (42, 46, 50)
HUD_BG = (0, 0, 0, 175)
HUD_FG = (232, 232, 232)
HUD_MUTED = (160, 164, 168)

# Цветовые шкалы для почвенных слоёв (позиция 0..1 -> RGB).
ORGANIC_STOPS = (
    (0.0, (18, 14, 10)),
    (0.25, (82, 52, 22)),
    (0.5, (128, 92, 38)),
    (0.75, (128, 166, 54)),
    (1.0, (206, 240, 110)),
)

ENERGY_STOPS = (
    (0.0, (10, 12, 22)),
    (0.3, (26, 62, 140)),
    (0.55, (20, 160, 200)),
    (0.8, (240, 220, 80)),
    (1.0, (255, 120, 40)),
)
