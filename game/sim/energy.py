"""Numba-ядро энергетики: BFS расстояний, инъекция, переток по стеблям."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def build_bfs(
    cell_type: np.ndarray,
    organism_id: np.ndarray,
    dist_seed: np.ndarray,
    parent_dir: np.ndarray,
    queue: np.ndarray,
    seed_y: int,
    seed_x: int,
    org_id: int,
    stem_code: int,
) -> int:
    """BFS от семени по стеблям одного организма.

    dist_seed: 0 у семени, >0 у стеблей, -1 у всего остального.
    parent_dir: направление от клетки к родителю (0 вверх, 1 вниз, 2 влево, 3 вправо).
    Возвращает число посещённых клеток (включая семя).
    """
    dist_seed.fill(-1)
    parent_dir.fill(-1)

    h, w = cell_type.shape
    if seed_y < 0 or seed_x < 0:
        return 0

    head = 0
    tail = 0
    dist_seed[seed_y, seed_x] = 0
    queue[tail] = seed_y * w + seed_x
    tail += 1

    while head < tail:
        idx = queue[head]
        head += 1
        y = idx // w
        x = idx - y * w
        d = dist_seed[y, x]

        # вверх: сосед сверху, родитель для него — вниз (1)
        ny = y - 1
        if ny < 0:
            ny = h - 1
        if (
            cell_type[ny, x] == stem_code
            and dist_seed[ny, x] == -1
            and organism_id[ny, x] == org_id
        ):
            dist_seed[ny, x] = d + 1
            parent_dir[ny, x] = 1
            queue[tail] = ny * w + x
            tail += 1

        # вниз: родитель для соседа — вверх (0)
        ny = y + 1
        if ny >= h:
            ny = 0
        if (
            cell_type[ny, x] == stem_code
            and dist_seed[ny, x] == -1
            and organism_id[ny, x] == org_id
        ):
            dist_seed[ny, x] = d + 1
            parent_dir[ny, x] = 0
            queue[tail] = ny * w + x
            tail += 1

        # влево: родитель для соседа — вправо (3)
        nx = x - 1
        if nx < 0:
            nx = w - 1
        if (
            cell_type[y, nx] == stem_code
            and dist_seed[y, nx] == -1
            and organism_id[y, nx] == org_id
        ):
            dist_seed[y, nx] = d + 1
            parent_dir[y, nx] = 3
            queue[tail] = y * w + nx
            tail += 1

        # вправо: родитель для соседа — влево (2)
        nx = x + 1
        if nx >= w:
            nx = 0
        if (
            cell_type[y, nx] == stem_code
            and dist_seed[y, nx] == -1
            and organism_id[y, nx] == org_id
        ):
            dist_seed[y, nx] = d + 1
            parent_dir[y, nx] = 2
            queue[tail] = y * w + nx
            tail += 1

    return tail


@njit(cache=True)
def flow_stems(
    energy: np.ndarray,
    cell_type: np.ndarray,
    capacity: np.ndarray,
    dist_seed: np.ndarray,
    parent_dir: np.ndarray,
    queue: np.ndarray,
    count: int,
    seed_capacity: float,
    stem_code: int,
    seed_code: int,
) -> None:
    """Переток энергии по стеблям к семени.

    queue содержит клетки в порядке BFS (от семени наружу), поэтому
    за один тик энергия сдвигается максимум на одну клетку.
    """
    h, w = cell_type.shape
    for i in range(count):
        idx = queue[i]
        y = idx // w
        x = idx - y * w
        if cell_type[y, x] != stem_code:
            continue
        if dist_seed[y, x] <= 0:
            continue

        pd = parent_dir[y, x]
        if pd < 0:
            continue

        py = y
        px = x
        if pd == 0:
            py = y - 1
            if py < 0:
                py = h - 1
        elif pd == 1:
            py = y + 1
            if py >= h:
                py = 0
        elif pd == 2:
            px = x - 1
            if px < 0:
                px = w - 1
        else:
            px = x + 1
            if px >= w:
                px = 0

        current = energy[y, x]
        if current <= 0.0:
            continue

        target_type = cell_type[py, px]
        if target_type == seed_code:
            free = seed_capacity - energy[py, px]
        elif target_type == stem_code:
            free = capacity[py, px] - energy[py, px]
        else:
            continue
        if free <= 0.0:
            continue

        transfer = current if current < free else free
        energy[y, x] = current - transfer
        energy[py, px] += transfer


@njit(cache=True)
def inject_workers(
    energy: np.ndarray,
    cell_type: np.ndarray,
    capacity: np.ndarray,
    dist_seed: np.ndarray,
    worker_idx: np.ndarray,
    worker_count: int,
    green_code: int,
    orange_code: int,
    blue_code: int,
    stem_code: int,
) -> None:
    """Работники отдают энергию только соседним стеблям.

    Семя в инъекции не участвует: прямой передачи работник -> семя нет.
    """
    h, w = cell_type.shape
    for k in range(worker_count):
        idx = worker_idx[k]
        y = idx // w
        x = idx - y * w
        code = cell_type[y, x]
        if code != green_code and code != orange_code and code != blue_code:
            continue

        current = energy[y, x]
        if current <= 0.0:
            continue

        best_y = -1
        best_x = -1
        best_dist = 2147483647
        best_free = 0.0

        # вверх
        ny = y - 1
        if ny < 0:
            ny = h - 1
        if cell_type[ny, x] == stem_code and dist_seed[ny, x] >= 0:
            free = capacity[ny, x] - energy[ny, x]
            if free > 0.0 and dist_seed[ny, x] < best_dist:
                best_y, best_x = ny, x
                best_dist = dist_seed[ny, x]
                best_free = free

        # вниз
        ny = y + 1
        if ny >= h:
            ny = 0
        if cell_type[ny, x] == stem_code and dist_seed[ny, x] >= 0:
            free = capacity[ny, x] - energy[ny, x]
            if free > 0.0 and dist_seed[ny, x] < best_dist:
                best_y, best_x = ny, x
                best_dist = dist_seed[ny, x]
                best_free = free

        # влево
        nx = x - 1
        if nx < 0:
            nx = w - 1
        if cell_type[y, nx] == stem_code and dist_seed[y, nx] >= 0:
            free = capacity[y, nx] - energy[y, nx]
            if free > 0.0 and dist_seed[y, nx] < best_dist:
                best_y, best_x = y, nx
                best_dist = dist_seed[y, nx]
                best_free = free

        # вправо
        nx = x + 1
        if nx >= w:
            nx = 0
        if cell_type[y, nx] == stem_code and dist_seed[y, nx] >= 0:
            free = capacity[y, nx] - energy[y, nx]
            if free > 0.0 and dist_seed[y, nx] < best_dist:
                best_y, best_x = y, nx
                best_dist = dist_seed[y, nx]
                best_free = free

        if best_y < 0:
            continue

        transfer = current if current < best_free else best_free
        energy[y, x] = current - transfer
        energy[best_y, best_x] += transfer
