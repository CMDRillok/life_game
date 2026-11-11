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
    green_single_efficiency: float,
    green_pair_efficiency: float,
    stem_code: int,
    empty_code: int,
) -> None:
    """Производство энергии за тик.

    produced[y, x] = 1, если клетка произвела энергию в этом тике.
    Это же используется для проверки голодной смерти.

    Зелёная клетка считает всех занятых соседей по 8 направлениям, кроме
    стеблей: 0 соседей — 100%, 1 — `green_single_efficiency`, 2 —
    `green_pair_efficiency`, 3 и больше — 0%.
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
                    ncode = cell_type[ny, nx]
                    if ncode != empty_code and ncode != stem_code:
                        neighbors += 1

            if neighbors == 0:
                efficiency = 1.0
            elif neighbors == 1:
                efficiency = green_single_efficiency
            elif neighbors == 2:
                efficiency = green_pair_efficiency
            else:
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
def prune_dead_stems(
    cell_type: np.ndarray,
    energy: np.ndarray,
    organic: np.ndarray,
    soil_energy: np.ndarray,
    capacity: np.ndarray,
    organism_id: np.ndarray,
    dist_seed: np.ndarray,
    die_at: np.ndarray,
    alive_idx: np.ndarray,
    alive_pos: np.ndarray,
    alive_count: int,
    stem_degree: np.ndarray,
    prune_queue: np.ndarray,
    tick: int,
    grace: int,
    overflow_idx: np.ndarray,
    overflow_seen: np.ndarray,
    overflow_count: int,
    stem_code: int,
    empty_code: int,
    death_organic: float,
    radius: int,
    toxic_threshold: float,
) -> tuple[int, int, int]:
    """Удалить стебли, которые никуда не ведут или попали в токсичную почву.

    Стебель полезен, если он соединён с семенем и либо сам примыкает к
    семени/работнику, либо лежит на пути к другому полезному стеблю.
    Листья, у которых на конце ничего нет, помечаются каскадом. Стебли без
    пути к семени (`dist_seed < 0`) удаляются сразу: организм остался без
    семени. Живым тупикам даётся `grace` тиков, чтобы семя успело построить
    работника и превратить ветвь в полезную.

    Состояние степени кодируется в `stem_degree`:
    0..4 — обычный стебель, 8..12 — в очереди, -1 — тупик, -2 — полезный.
    """
    h, w = cell_type.shape
    queue = prune_queue
    head = 0
    tail = 0

    # Первый проход: степени и начальные кандидаты на удаление.
    for i in range(alive_count):
        idx = alive_idx[i]
        y = idx // w
        x = idx - y * w
        if cell_type[y, x] != stem_code:
            continue
        org = organism_id[y, x]
        degree = 0
        attached = False

        ny = _wrap(y - 1, h)
        ncode = cell_type[ny, x]
        if ncode == stem_code:
            if organism_id[ny, x] == org:
                degree += 1
        elif ncode != empty_code and organism_id[ny, x] == org:
            attached = True

        ny = _wrap(y + 1, h)
        ncode = cell_type[ny, x]
        if ncode == stem_code:
            if organism_id[ny, x] == org:
                degree += 1
        elif ncode != empty_code and organism_id[ny, x] == org:
            attached = True

        nx = _wrap(x - 1, w)
        ncode = cell_type[y, nx]
        if ncode == stem_code:
            if organism_id[y, nx] == org:
                degree += 1
        elif ncode != empty_code and organism_id[y, nx] == org:
            attached = True

        nx = _wrap(x + 1, w)
        ncode = cell_type[y, nx]
        if ncode == stem_code:
            if organism_id[y, nx] == org:
                degree += 1
        elif ncode != empty_code and organism_id[y, nx] == org:
            attached = True

        stem_degree[y, x] = degree
        if dist_seed[y, x] < 0 or (not attached and degree <= 1):
            stem_degree[y, x] = degree + 8
            queue[tail] = idx
            tail += 1

    # Каскад: удаляем бесполезные листья, пока дерево не стабилизируется.
    while head < tail:
        idx = queue[head]
        head += 1
        y = idx // w
        x = idx - y * w
        if cell_type[y, x] != stem_code:
            continue
        state = stem_degree[y, x]
        if state < 8:
            continue
        degree = state - 8
        org = organism_id[y, x]

        attached = False
        ny = _wrap(y - 1, h)
        ncode = cell_type[ny, x]
        if (
            ncode != empty_code
            and ncode != stem_code
            and organism_id[ny, x] == org
        ):
            attached = True
        else:
            ny = _wrap(y + 1, h)
            ncode = cell_type[ny, x]
            if (
                ncode != empty_code
                and ncode != stem_code
                and organism_id[ny, x] == org
            ):
                attached = True
            else:
                nx = _wrap(x - 1, w)
                ncode = cell_type[y, nx]
                if (
                    ncode != empty_code
                    and ncode != stem_code
                    and organism_id[y, nx] == org
                ):
                    attached = True
                else:
                    nx = _wrap(x + 1, w)
                    ncode = cell_type[y, nx]
                    if (
                        ncode != empty_code
                        and ncode != stem_code
                        and organism_id[y, nx] == org
                    ):
                        attached = True

        if not (dist_seed[y, x] < 0 or (not attached and degree <= 1)):
            stem_degree[y, x] = -2
            continue

        stem_degree[y, x] = -1

        ny = _wrap(y - 1, h)
        if cell_type[ny, x] == stem_code and organism_id[ny, x] == org:
            nidx = ny * w + x
            nstate = stem_degree[ny, x]
            if nstate >= 8:
                if nstate > 8:
                    stem_degree[ny, x] = nstate - 1
            elif nstate >= 0:
                ndegree = nstate - 1
                if ndegree <= 1:
                    stem_degree[ny, x] = ndegree + 8
                    queue[tail] = nidx
                    tail += 1
                else:
                    stem_degree[ny, x] = ndegree

        ny = _wrap(y + 1, h)
        if cell_type[ny, x] == stem_code and organism_id[ny, x] == org:
            nidx = ny * w + x
            nstate = stem_degree[ny, x]
            if nstate >= 8:
                if nstate > 8:
                    stem_degree[ny, x] = nstate - 1
            elif nstate >= 0:
                ndegree = nstate - 1
                if ndegree <= 1:
                    stem_degree[ny, x] = ndegree + 8
                    queue[tail] = nidx
                    tail += 1
                else:
                    stem_degree[ny, x] = ndegree

        nx = _wrap(x - 1, w)
        if cell_type[y, nx] == stem_code and organism_id[y, nx] == org:
            nidx = y * w + nx
            nstate = stem_degree[y, nx]
            if nstate >= 8:
                if nstate > 8:
                    stem_degree[y, nx] = nstate - 1
            elif nstate >= 0:
                ndegree = nstate - 1
                if ndegree <= 1:
                    stem_degree[y, nx] = ndegree + 8
                    queue[tail] = nidx
                    tail += 1
                else:
                    stem_degree[y, nx] = ndegree

        nx = _wrap(x + 1, w)
        if cell_type[y, nx] == stem_code and organism_id[y, nx] == org:
            nidx = y * w + nx
            nstate = stem_degree[y, nx]
            if nstate >= 8:
                if nstate > 8:
                    stem_degree[y, nx] = nstate - 1
            elif nstate >= 0:
                ndegree = nstate - 1
                if ndegree <= 1:
                    stem_degree[y, nx] = ndegree + 8
                    queue[tail] = nidx
                    tail += 1
                else:
                    stem_degree[y, nx] = ndegree

    # Второй проход: удалить помеченные стебли и рассеять их энергию.
    died = 0
    i = 0
    while i < alive_count:
        idx = alive_idx[i]
        y = idx // w
        x = idx - y * w
        code = cell_type[y, x]

        if code == empty_code:
            last = alive_idx[alive_count - 1]
            alive_idx[i] = last
            alive_pos[last] = i
            alive_pos[idx] = -1
            alive_count -= 1
            continue

        if code == stem_code:
            toxic = (
                soil_energy[y, x] >= toxic_threshold
                or organic[y, x] >= toxic_threshold
            )
            remove = False
            if toxic:
                # В токсичной почве стебель не выживает.
                remove = True
            elif stem_degree[y, x] == -1:
                if dist_seed[y, x] < 0:
                    remove = True
                elif die_at[y, x] == 0:
                    die_at[y, x] = tick + grace
                elif tick >= die_at[y, x]:
                    remove = True
            else:
                # Стебель снова стал полезным — снимаем отложенное удаление.
                die_at[y, x] = 0

            if remove:
                spill = energy[y, x]
                side = radius * 2 + 1
                area = side * side
                energy_part = spill / area if spill > 0.0 else 0.0
                organic_part = death_organic / area
                for dy in range(-radius, radius + 1):
                    ny = _wrap(y + dy, h)
                    for dx in range(-radius, radius + 1):
                        nx = _wrap(x + dx, w)
                        soil_energy[ny, nx] += energy_part
                        organic[ny, nx] += organic_part
                        oidx = ny * w + nx
                        if (
                            overflow_seen[ny, nx] == 0
                            and overflow_count < overflow_idx.size
                        ):
                            overflow_seen[ny, nx] = 1
                            overflow_idx[overflow_count] = oidx
                            overflow_count += 1

                cell_type[y, x] = empty_code
                energy[y, x] = 0.0
                capacity[y, x] = 0.0
                organism_id[y, x] = 0
                dist_seed[y, x] = -1
                die_at[y, x] = 0
                stem_degree[y, x] = 0

                last = alive_idx[alive_count - 1]
                alive_idx[i] = last
                alive_pos[last] = i
                alive_pos[idx] = -1
                alive_count -= 1
                died += 1
                continue

        i += 1

    return died, alive_count, overflow_count


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
    overflow_idx: np.ndarray,
    overflow_seen: np.ndarray,
    overflow_count: int,
    green_code: int,
    orange_code: int,
    blue_code: int,
    stem_code: int,
    empty_code: int,
    death_organic: float,
    radius: int,
    toxic_threshold: float,
) -> tuple[int, int, int]:
    """Убить работников: голод, старость, токсичная почва и отрыв от стебля.

    Обходит только живые клетки из `alive_idx` и сжимает список на месте.
    Энергия и органика умершей клетки распределяются по области
    (2*radius+1)^2.

    В токсичной почве гибнут все, кроме синей при избытке энергии и
    оранжевой при избытке органики. Работник без соседнего стебля своего
    организма тоже умирает сразу.

    Стебли и семена сюда не попадают: их обрабатывают `prune_dead_stems` и
    `Simulation.apply_toxic_seed_deaths`.
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

        worker = (
            code == green_code or code == orange_code or code == blue_code
        )
        dead = False
        if worker:
            if soil_energy[y, x] >= toxic_threshold and code != blue_code:
                dead = True
            if organic[y, x] >= toxic_threshold and code != orange_code:
                dead = True

            if not dead:
                org = organism_id[y, x]
                has_stem = False

                ny = _wrap(y - 1, h)
                if (
                    cell_type[ny, x] == stem_code
                    and organism_id[ny, x] == org
                ):
                    has_stem = True

                ny = _wrap(y + 1, h)
                if (
                    cell_type[ny, x] == stem_code
                    and organism_id[ny, x] == org
                ):
                    has_stem = True

                nx = _wrap(x - 1, w)
                if (
                    cell_type[y, nx] == stem_code
                    and organism_id[y, nx] == org
                ):
                    has_stem = True

                nx = _wrap(x + 1, w)
                if (
                    cell_type[y, nx] == stem_code
                    and organism_id[y, nx] == org
                ):
                    has_stem = True

                if not has_stem:
                    dead = True

            if (
                not dead
                and (code == orange_code or code == blue_code)
                and produced[y, x] == 0
                and energy[y, x] <= 0.0
            ):
                dead = True
            if not dead and die_at[y, x] > 0 and tick >= die_at[y, x]:
                dead = True

        if not dead:
            i += 1
            continue

        spill = energy[y, x]
        side = radius * 2 + 1
        area = side * side
        energy_part = spill / area if spill > 0.0 else 0.0
        organic_part = death_organic / area
        for dy in range(-radius, radius + 1):
            ny = _wrap(y + dy, h)
            for dx in range(-radius, radius + 1):
                nx = _wrap(x + dx, w)
                soil_energy[ny, nx] += energy_part
                organic[ny, nx] += organic_part
                oidx = ny * w + nx
                if (
                    overflow_seen[ny, nx] == 0
                    and overflow_count < overflow_idx.size
                ):
                    overflow_seen[ny, nx] = 1
                    overflow_idx[overflow_count] = oidx
                    overflow_count += 1

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

    return died, alive_count, overflow_count


@njit(cache=True)
def spread_overflow_queue(
    soil_energy: np.ndarray,
    organic: np.ndarray,
    queue: np.ndarray,
    count: int,
    seen: np.ndarray,
    threshold: float,
) -> tuple[int, bool]:
    """Разлить избыток из очереди ячеек и отложить соседей на следующий тик.

    Обрабатывается снимок очереди `[0:count)`: ячейка ограничивается порогом,
    избыток делится между 8 соседями. Соседи, которые после этого сами
    превысили порог, попадают в очередь на следующий тик.
    """
    h, w = soil_energy.shape
    head = 0
    new_count = count
    overflowed = False

    while head < count:
        idx = queue[head]
        head += 1
        y = idx // w
        x = idx - y * w
        seen[y, x] = 0

        value = soil_energy[y, x]
        if value > threshold:
            part = (value - threshold) / 8.0
            soil_energy[y, x] = threshold
            for dy in range(-1, 2):
                ny = _wrap(y + dy, h)
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    nx = _wrap(x + dx, w)
                    soil_energy[ny, nx] += part
                    if (
                        soil_energy[ny, nx] > threshold
                        and seen[ny, nx] == 0
                        and new_count < queue.size
                    ):
                        seen[ny, nx] = 1
                        queue[new_count] = ny * w + nx
                        new_count += 1
            overflowed = True

        value = organic[y, x]
        if value > threshold:
            part = (value - threshold) / 8.0
            organic[y, x] = threshold
            for dy in range(-1, 2):
                ny = _wrap(y + dy, h)
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    nx = _wrap(x + dx, w)
                    organic[ny, nx] += part
                    if (
                        organic[ny, nx] > threshold
                        and seen[ny, nx] == 0
                        and new_count < queue.size
                    ):
                        seen[ny, nx] = 1
                        queue[new_count] = ny * w + nx
                        new_count += 1
            overflowed = True

    next_count = new_count - count
    for i in range(next_count):
        queue[i] = queue[count + i]
    return next_count, overflowed
