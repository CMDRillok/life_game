"""Numba-ядро действий семени: поиск цели для создания клетки и движения.

Политика (что именно строить) живёт в Simulation, а здесь только дешёвый
поиск валидных клеток по BFS-очереди стеблей.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _neighbor(x: int, y: int, direction: int, h: int, w: int) -> tuple[int, int]:
    """Сосед по направлению: 0 вверх, 1 вниз, 2 влево, 3 вправо."""
    if direction == 0:
        y -= 1
        if y < 0:
            y = h - 1
    elif direction == 1:
        y += 1
        if y >= h:
            y = 0
    elif direction == 2:
        x -= 1
        if x < 0:
            x = w - 1
    else:
        x += 1
        if x >= w:
            x = 0
    return y, x


@njit(cache=True)
def find_worker_target(
    cell_type: np.ndarray,
    stem_queue: np.ndarray,
    start: int,
    end: int,
    stem_code: int,
    empty_code: int,
    direction_offset: int,
) -> tuple[int, int, int]:
    """Найти пустую клетку для работника рядом со стеблем организма.

    `stem_queue[start:end]` — стебли только этого организма в обратном порядке
    BFS: сначала самые дальние от семени, чтобы организм рос наружу.
    Работники теперь могут стоять вплотную друг к другу; плотность влияет
    только на эффективность зелёных клеток. Направления обходятся с поворотом
    на direction_offset.

    Возвращает (found, y, x).
    """
    h, w = cell_type.shape
    for i in range(start, end):
        idx = stem_queue[i]
        sy = idx // w
        sx = idx - sy * w
        if cell_type[sy, sx] != stem_code:
            continue

        for k in range(4):
            direction = (k + direction_offset) & 3
            ny, nx = _neighbor(sx, sy, direction, h, w)
            if cell_type[ny, nx] == empty_code:
                return 1, ny, nx

    return 0, 0, 0


@njit(cache=True)
def find_move_target(
    cell_type: np.ndarray,
    seed_y: int,
    seed_x: int,
    empty_code: int,
    direction_offset: int,
) -> tuple[int, int, int]:
    """Найти пустого 4-соседа для перемещения семени."""
    h, w = cell_type.shape
    for k in range(4):
        direction = (k + direction_offset) & 3
        ny, nx = _neighbor(seed_x, seed_y, direction, h, w)
        if cell_type[ny, nx] == empty_code:
            return 1, ny, nx
    return 0, 0, 0


@njit(cache=True)
def find_seed_target(
    cell_type: np.ndarray,
    stem_queue: np.ndarray,
    start: int,
    end: int,
    stem_code: int,
    empty_code: int,
    direction_offset: int,
) -> tuple[int, int, int]:
    """Найти пустую клетку рядом со стеблем организма.

    `stem_queue[start:end]` — стебли только этого организма в обратном
    порядке BFS. Используется внутренним размножением семени: новое семя
    должно попасть в сеть стеблей, чтобы сразу получать энергию. Работники
    при этом не учитываются — рядом с семенем им стоять можно.
    """
    h, w = cell_type.shape
    for i in range(start, end):
        idx = stem_queue[i]
        sy = idx // w
        sx = idx - sy * w
        if cell_type[sy, sx] != stem_code:
            continue

        for k in range(4):
            direction = (k + direction_offset) & 3
            ny, nx = _neighbor(sx, sy, direction, h, w)
            if cell_type[ny, nx] == empty_code:
                return 1, ny, nx

    return 0, 0, 0


@njit(cache=True)
def find_shoot_target(
    cell_type: np.ndarray,
    stem_queue: np.ndarray,
    start: int,
    end: int,
    stem_code: int,
    empty_code: int,
    direction_offset: int,
) -> tuple[int, int, int, int]:
    """Найти пустую клетку рядом со стеблем для отстрела нового организма.

    Возвращает (found, y, x, direction), где direction — направление от
    стебля к новой клетке (0 вверх, 1 вниз, 2 влево, 3 вправо). Семя-потомок
    полетит именно в эту сторону, не оставляя за собой стебли.
    """
    h, w = cell_type.shape
    for i in range(start, end):
        idx = stem_queue[i]
        sy = idx // w
        sx = idx - sy * w
        if cell_type[sy, sx] != stem_code:
            continue

        for k in range(4):
            direction = (k + direction_offset) & 3
            ny, nx = _neighbor(sx, sy, direction, h, w)
            if cell_type[ny, nx] == empty_code:
                return 1, ny, nx, direction

    return 0, 0, 0, 0
