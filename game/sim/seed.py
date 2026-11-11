"""Numba-ядро действий семени: поиск цели для создания клетки и движения.

Политика (что именно строить) живёт в Simulation, а здесь только дешёвый
поиск валидных клеток по BFS-очереди стеблей.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _is_worker(code: int, green: int, orange: int, blue: int) -> bool:
    return code == green or code == orange or code == blue


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
    queue: np.ndarray,
    queue_count: int,
    green_code: int,
    orange_code: int,
    blue_code: int,
    stem_code: int,
    empty_code: int,
    direction_offset: int,
) -> tuple[int, int, int]:
    """Найти пустую клетку для работника рядом со стеблем (4 направления).

    Дополнительно проверяется запрет соседства с другими работниками по
    8 направлениям.

    Стебли перебираются в обратном порядке BFS: сначала самые дальние от семени,
    чтобы организм рос наружу. Направления обходятся с поворотом на
    direction_offset, чтобы рост не сваливался в одну сторону.

    Возвращает (found, y, x).
    """
    h, w = cell_type.shape
    for i in range(queue_count - 1, 0, -1):
        idx = queue[i]
        sy = idx // w
        sx = idx - sy * w
        if cell_type[sy, sx] != stem_code:
            continue

        for k in range(4):
            direction = (k + direction_offset) & 3
            ny, nx = _neighbor(sx, sy, direction, h, w)
            if cell_type[ny, nx] != empty_code:
                continue

            blocked = False
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    wy = ny + dy
                    if wy < 0:
                        wy = h - 1
                    elif wy >= h:
                        wy = 0
                    wx = nx + dx
                    if wx < 0:
                        wx = w - 1
                    elif wx >= w:
                        wx = 0
                    if _is_worker(
                        cell_type[wy, wx], green_code, orange_code, blue_code
                    ):
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue

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
