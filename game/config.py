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

# --- симуляция ---
SIM_HZ = 30.0
MAX_TICKS_PER_FRAME = 4
STATS_INTERVAL = 0.25  # как часто пересчитывать статистику для HUD

GREEN_RATE = 2.0
# Каждый непустой 8-сосед снижает эффективность зелёной на 1/8.
GREEN_CROWD_PENALTY = 1.0 / 8.0
ORANGE_MAX_INTAKE = 2.0
ORANGE_EFFICIENCY = 1.0
BLUE_MAX_INTAKE = 2.0
BLUE_EFFICIENCY = 1.0

WORKER_CAPACITY = 100.0
STEM_CAPACITY = 100.0
SEED_CAPACITY = 1000.0

DEATH_ORGANIC = 10.0
RESOURCE_RADIUS = 1  # 3x3

# --- действия семени ---
# Положительные коды — созданная клетка, отрицательные — особые действия.
SEED_ACTION_NONE = 0
SEED_ACTION_MOVE = -1
SEED_ACTION_SHOOT = -2
SEED_ACTION_DUPLICATE = -3
SEED_ACTION_TRAVEL = -4
SEED_ACTION_ESTABLISH = -5
SEED_ACTION_COLLIDE = -6
# Стебель создаётся только ходом: семя уходит, на старом месте остаётся стебель.
SEED_MOVE_COST = 5.0
WORKER_CREATE_COST = 20.0
# Семя платит за каждый тик жизни; если энергии не хватает — умирает.
SEED_UPKEEP = 1.0
# Семя-потомок после отстрела летит, не оставляя стеблей, и тратит энергию.
SEED_TRAVEL_COST = 2.0
# Остаток, с которым летящее семя укореняется и начинает строить организм.
SEED_ESTABLISH_ENERGY = 40.0
# Небольшая фиксированная мутация одного гена за тик.
TICK_MUTATION_CHANCE = 0.001
# Сколько тиков жизни даёт одна единица энергии, потраченная на создание клетки.
LIFETIME_TICKS_PER_ENERGY = 100.0
# Отстрел: потомок получает child_energy, родитель платит ещё и накладные.
SHOOT_EXTRA_COST = 10.0
# Внутреннее размножение: новое семя того же организма.
DUPLICATE_EXTRA_COST = 10.0
MAX_SEEDS_PER_ORGANISM = 4
# Пауза после размножения, чтобы популяция не взрывалась за десятки тиков.
SEED_SHOOT_COOLDOWN = 60
SEED_DUPLICATE_COOLDOWN = 90
# Предохранитель для длинных headless-прогонов: после лимита семена
# перестают размножаться, но продолжают расти и искать энергию.
MAX_TOTAL_SEEDS = 2000
# Сколько стеблей организм хочет иметь на одного работника (стартовый геном).
SEED_STEMS_PER_WORKER = 2

# Ограничения размещения: новая клетка появляется рядом со стеблем.
CREATE_ADJACENCY = 4

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
VIEW_CELL_ENERGY = 3

VIEW_NAMES = {
    VIEW_CELLS: "клетки",
    VIEW_ORGANIC: "органика почвы",
    VIEW_ENERGY: "энергия почвы",
    VIEW_CELL_ENERGY: "энергия клеток",
}

VIEW_ORDER = (VIEW_CELLS, VIEW_ORGANIC, VIEW_ENERGY, VIEW_CELL_ENERGY)

# --- оформление ---
GRID_COLOR = (42, 46, 50)
HUD_BG = (0, 0, 0, 175)
HUD_FG = (232, 232, 232)
HUD_MUTED = (160, 164, 168)

# Цветовые шкалы для почвенных слоёв (позиция 0..1 -> RGB).
# Ноль — чистый чёрный, дальше цвет светлеет с ростом значения.
ORGANIC_STOPS = (
    (0.0, (0, 0, 0)),
    (0.25, (0, 42, 12)),
    (0.5, (24, 112, 34)),
    (0.75, (92, 190, 70)),
    (1.0, (228, 255, 206)),
)

ENERGY_STOPS = (
    (0.0, (0, 0, 0)),
    (0.25, (4, 28, 78)),
    (0.5, (12, 92, 180)),
    (0.75, (44, 176, 230)),
    (1.0, (220, 248, 255)),
)

CELL_ENERGY_STOPS = (
    (0.0, (12, 14, 18)),
    (0.25, (22, 92, 150)),
    (0.5, (28, 176, 168)),
    (0.75, (238, 198, 72)),
    (1.0, (255, 92, 42)),
)
