"""Numba-правила: производство энергии и голодная смерть.

Работников на поле мало, поэтому вместо полного прохода по 4 млн клеток
ядра обходят только их индексы (`worker_idx`), которые Simulation
пересобирает при изменении топологии.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _wrap(value: int, size: int) -> int:
    while value < 0:
        value += size
    while value >= size:
        value -= size
    return value


@njit(cache=True)
def produce(
    energy: np.ndarray,
    cell_type: np.ndarray,
    organic: np.ndarray,
    soil_energy: np.ndarray,
    produced: np.ndarray,
    worker_idx: np.ndarray,
    worker_count: int,
    green_code: int,
    orange_code: int,
    blue_code: int,
    green_rate: float,
    worker_capacity: float,
    orange_max_intake: float,
    orange_efficiency: float,
    blue_max_intake: float,
    blue_efficiency: float,
    radius: int,
    green_crowd_penalty: float,
    empty_code: int,
) -> None:
    """Производство энергии за тик.

    produced[y, x] = 1, если клетка произвела энергию в этом тике.
    Это же используется для проверки голодной смерти.

    Зелёная клетка теряет `green_crowd_penalty` эффективности за каждого
    непустого 8-соседа: в плотной застройке фотосинтез падает.
    """
    h, w = cell_type.shape
    for k in range(worker_count):
        idx = worker_idx[k]
        y = idx // w
        x = idx - y * w
        produced[y, x] = 0
        code = cell_type[y, x]

        if code == green_code:
            neighbors = 0
            for dy in range(-1, 2):
                ny = _wrap(y + dy, h)
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    nx = _wrap(x + dx, w)
                    if cell_type[ny, nx] != empty_code:
                        neighbors += 1

            efficiency = 1.0 - neighbors * green_crowd_penalty
            if efficiency < 0.0:
                efficiency = 0.0
            room = worker_capacity - energy[y, x]
            if room > 0.0 and efficiency > 0.0:
                gain = green_rate * efficiency
                if gain > room:
                    gain = room
                if gain > 0.0:
                    energy[y, x] += gain
                    produced[y, x] = 1

        elif code == orange_code:
            total = 0.0
            for dy in range(-radius, radius + 1):
                ny = _wrap(y + dy, h)
                for dx in range(-radius, radius + 1):
                    nx = _wrap(x + dx, w)
                    total += organic[ny, nx]
            if total <= 0.0:
                continue

            room = worker_capacity - energy[y, x]
            if room <= 0.0:
                continue

            intake = orange_max_intake
            if total < intake:
                intake = total
            if orange_efficiency > 0.0:
                by_room = room / orange_efficiency
                if by_room < intake:
                    intake = by_room
            if intake <= 0.0:
                continue

            for dy in range(-radius, radius + 1):
                ny = _wrap(y + dy, h)
                for dx in range(-radius, radius + 1):
                    nx = _wrap(x + dx, w)
                    part = intake * organic[ny, nx] / total
                    if part > organic[ny, nx]:
                        part = organic[ny, nx]
                    organic[ny, nx] -= part

            energy[y, x] += intake * orange_efficiency
            produced[y, x] = 1

        elif code == blue_code:
            total = 0.0
            for dy in range(-radius, radius + 1):
                ny = _wrap(y + dy, h)
                for dx in range(-radius, radius + 1):
                    nx = _wrap(x + dx, w)
                    total += soil_energy[ny, nx]
            if total <= 0.0:
                continue

            room = worker_capacity - energy[y, x]
            if room <= 0.0:
                continue

            intake = blue_max_intake
            if total < intake:
                intake = total
            if blue_efficiency > 0.0:
                by_room = room / blue_efficiency
                if by_room < intake:
                    intake = by_room
            if intake <= 0.0:
                continue

            for dy in range(-radius, radius + 1):
                ny = _wrap(y + dy, h)
                for dx in range(-radius, radius + 1):
                    nx = _wrap(x + dx, w)
                    part = intake * soil_energy[ny, nx] / total
                    if part > soil_energy[ny, nx]:
                        part = soil_energy[ny, nx]
                    soil_energy[ny, nx] -= part

            energy[y, x] += intake * blue_efficiency
            produced[y, x] = 1


@njit(cache=True)
def apply_deaths(
    cell_type: np.ndarray,
    energy: np.ndarray,
    organic: np.ndarray,
    soil_energy: np.ndarray,
    capacity: np.ndarray,
    organism_id: np.ndarray,
    dist_seed: np.ndarray,
    die_at: np.ndarray,
    produced: np.ndarray,
    alive_idx: np.ndarray,
    alive_pos: np.ndarray,
    alive_count: int,
    tick: int,
    green_code: int,
    orange_code: int,
    blue_code: int,
    stem_code: int,
    seed_code: int,
    empty_code: int,
    death_organic: float,
    radius: int,
    stem_idle_timeout: int,
) -> tuple[int, int]:
    """Убить голодающих/старых работников и стебли-тупики.

    Обходит только живые клетки из `alive_idx` и сжимает список на месте.
    Энергия умершей клетки равномерно рассеивается в `soil_energy` по области
    (2*radius+1)^2, в собственной клетке остаётся органика.

    Стебель не стареет. Он продлевает жизнь, пока является частью пути
    (2+ соседних стебля) или к нему прикреплены работник/семя. Тупик без
    потока энергии умирает после `stem_idle_timeout` тиков простоя.

    Семена сюда не попадают: у них нет срока жизни, они умирают отдельно,
    когда заканчивается энергия на поддержание.
    """
    h, w = cell_type.shape
    died = 0
    i = 0
    while i < alive_count:
        idx = alive_idx[i]
        y = idx // w
        x = idx - y * w
        code = cell_type[y, x]

        if code == empty_code:
            # Устаревшая запись: клетка уже исчезла.
            last = alive_idx[alive_count - 1]
            alive_idx[i] = last
            alive_pos[last] = i
            alive_pos[idx] = -1
            alive_count -= 1
            continue

        dead = False
        if code == stem_code:
            stem_neighbors = 0
            attached = False

            ny = _wrap(y - 1, h)
            ncode = cell_type[ny, x]
            if ncode == stem_code:
                stem_neighbors += 1
            elif ncode != empty_code:
                attached = True

            ny = _wrap(y + 1, h)
            ncode = cell_type[ny, x]
            if ncode == stem_code:
                stem_neighbors += 1
            elif ncode != empty_code:
                attached = True

            nx = _wrap(x - 1, w)
            ncode = cell_type[y, nx]
            if ncode == stem_code:
                stem_neighbors += 1
            elif ncode != empty_code:
                attached = True

            nx = _wrap(x + 1, w)
            ncode = cell_type[y, nx]
            if ncode == stem_code:
                stem_neighbors += 1
            elif ncode != empty_code:
                attached = True

            if attached or stem_neighbors > 1:
                die_at[y, x] = tick + stem_idle_timeout
            elif die_at[y, x] == 0:
                die_at[y, x] = tick + stem_idle_timeout
            elif tick >= die_at[y, x]:
                dead = True
        else:
            if (code == orange_code or code == blue_code) and produced[y, x] == 0:
                if energy[y, x] <= 0.0:
                    dead = True
            if (
                (code == green_code or code == orange_code or code == blue_code)
                and die_at[y, x] > 0
                and tick >= die_at[y, x]
            ):
                dead = True

        if not dead:
            i += 1
            continue

        spill = energy[y, x]
        if spill > 0.0:
            side = radius * 2 + 1
            part = spill / (side * side)
            for dy in range(-radius, radius + 1):
                ny = _wrap(y + dy, h)
                for dx in range(-radius, radius + 1):
                    nx = _wrap(x + dx, w)
                    soil_energy[ny, nx] += part

        organic[y, x] += death_organic
        cell_type[y, x] = empty_code
        energy[y, x] = 0.0
        capacity[y, x] = 0.0
        organism_id[y, x] = 0
        dist_seed[y, x] = -1
        die_at[y, x] = 0
        produced[y, x] = 0

        last = alive_idx[alive_count - 1]
        alive_idx[i] = last
        alive_pos[last] = i
        alive_pos[idx] = -1
        alive_count -= 1
        died += 1

    return died, alive_count
