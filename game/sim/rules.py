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
) -> None:
    """Производство энергии за тик.

    produced[y, x] = 1, если клетка произвела энергию в этом тике.
    Это же используется для проверки голодной смерти.
    """
    h, w = cell_type.shape
    for k in range(worker_count):
        idx = worker_idx[k]
        y = idx // w
        x = idx - y * w
        produced[y, x] = 0
        code = cell_type[y, x]

        if code == green_code:
            room = worker_capacity - energy[y, x]
            if room > 0.0:
                gain = green_rate if green_rate < room else room
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
def apply_starvation_death(
    cell_type: np.ndarray,
    energy: np.ndarray,
    organic: np.ndarray,
    capacity: np.ndarray,
    organism_id: np.ndarray,
    dist_seed: np.ndarray,
    produced: np.ndarray,
    worker_idx: np.ndarray,
    worker_count: int,
    orange_code: int,
    blue_code: int,
    empty_code: int,
    death_organic: float,
) -> int:
    """Убить голодающих оранжевых и синих.

    Условие: за тик не произвели энергию и их собственная энергия <= 0.
    В почву добавляется органика. Энергии при голодной смерти уже нет.
    """
    h, w = cell_type.shape
    died = 0
    for k in range(worker_count):
        idx = worker_idx[k]
        y = idx // w
        x = idx - y * w
        code = cell_type[y, x]
        if code != orange_code and code != blue_code:
            continue
        if produced[y, x] != 0:
            continue
        if energy[y, x] > 0.0:
            continue

        organic[y, x] += death_organic
        cell_type[y, x] = empty_code
        energy[y, x] = 0.0
        capacity[y, x] = 0.0
        organism_id[y, x] = 0
        dist_seed[y, x] = -1
        died += 1

    return died
