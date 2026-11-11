"""Оркестрация тика симуляции.

В поле может жить несколько организмов (`organism_id`) и несколько семян
внутри одного организма. Состояние семян (координаты, геном, поколение)
хранится в Python-списке `SeedState`, а горячие слои поля остаются numpy SoA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from ..field import Field
from . import energy as energy_kernels
from . import rules
from . import seed as seed_kernels
from .genome import Genome


@dataclass
class SeedState:
    """Одно семя организма: позиция, геном и поколение."""

    uid: int
    y: int
    x: int
    org_id: int
    genome: Genome
    generation: int = 0
    last_shoot_tick: int = -10_000
    last_duplicate_tick: int = -10_000
    last_mutation_tick: int = -1
    # Семя после отстрела сначала летит, не оставляя стеблей.
    established: bool = True
    travel_dx: int = 0
    travel_dy: int = 0
    travel_steps: int = 0
    travel_limit: int = 0


class Simulation:
    """Тик: BFS -> производство -> инъекция -> переток -> смерть -> действия семян."""

    def __init__(
        self,
        field: Field,
        seed_actions: bool = True,
        rng_seed: int | None = None,
        random_genomes: bool = False,
    ) -> None:
        self.field = field
        self.seed_actions_enabled = seed_actions
        self.random_initial_genomes = random_genomes
        self.tick_count = 0
        self.dirty = True
        self.bfs_count = 0

        self.queue = np.empty(field.cells, dtype=np.int32)
        self.parent_dir = np.full(field.shape, -1, dtype=np.int8)
        self.produced = np.zeros(field.shape, dtype=np.uint8)
        self.worker_idx = np.empty(field.cells, dtype=np.int32)
        self.worker_count = 0
        self.workers_dirty = True
        # Стебли, сгруппированные по организмам в обратном порядке BFS.
        self.org_stem_idx = np.empty(field.cells, dtype=np.int32)
        self.org_stem_ranges: dict[int, tuple[int, int]] = {}

        # Все занятые клетки: смерть и старение обходят только их.
        self.alive_idx = np.empty(field.cells, dtype=np.int32)
        self.alive_pos = np.full(field.cells, -1, dtype=np.int32)
        self.alive_count = 0
        self.alive_dirty = True
        self.lifetimes_ready = False
        # Степени и очередь для каскадного удаления стеблей-тупиков.
        self.stem_degree = np.zeros(field.shape, dtype=np.int8)
        self.prune_queue = np.empty(field.cells, dtype=np.int32)
        # Очередь ячеек, где может понадобиться разлить избыток ресурса.
        self.overflow_idx = np.empty(field.cells, dtype=np.int32)
        self.overflow_seen = np.zeros(field.shape, dtype=np.uint8)
        self.overflow_count = 0

        self.rng = np.random.default_rng(rng_seed)
        self.seeds: list[SeedState] = []
        self.seed_idx = np.empty(16, dtype=np.int32)
        self.seed_idx_count = 0
        self.seed_slots: dict[int, int] = {}
        self.seeds_dirty = True
        self._next_seed_uid = 0
        self.next_org_id = 1

        self._seed_phase = False
        self._org_counts: dict[int, list[int]] = {}
        self.last_seed_action = config.SEED_ACTION_NONE
        self.refresh_seeds()
        self.queue_all_overflow()

    # ------------------------------------------------------------------ служебное

    def _ensure_scratch(self) -> None:
        if (
            self.queue.size != self.field.cells
            or self.parent_dir.shape != self.field.shape
            or self.produced.shape != self.field.shape
        ):
            self.queue = np.empty(self.field.cells, dtype=np.int32)
            self.parent_dir = np.full(self.field.shape, -1, dtype=np.int8)
            self.produced = np.zeros(self.field.shape, dtype=np.uint8)
            self.worker_idx = np.empty(self.field.cells, dtype=np.int32)
            self.worker_count = 0
            self.workers_dirty = True
            self.org_stem_idx = np.empty(self.field.cells, dtype=np.int32)
            self.org_stem_ranges = {}
            self.alive_idx = np.empty(self.field.cells, dtype=np.int32)
            self.alive_pos = np.full(self.field.cells, -1, dtype=np.int32)
            self.alive_count = 0
            self.alive_dirty = True
            self.lifetimes_ready = False
            self.stem_degree = np.zeros(self.field.shape, dtype=np.int8)
            self.prune_queue = np.empty(self.field.cells, dtype=np.int32)
            self.overflow_idx = np.empty(self.field.cells, dtype=np.int32)
            self.overflow_seen = np.zeros(
                self.field.shape, dtype=np.uint8
            )
            self.overflow_count = 0
            self.dirty = True
            self.seeds_dirty = True

    def rebuild_alive(self) -> int:
        """Собрать индексы всех занятых клеток и позиции для O(1) удаления."""
        indices = np.flatnonzero((self.field.cell_type != config.EMPTY).ravel())
        self.alive_count = int(indices.size)
        self.alive_idx[: self.alive_count] = indices
        self.alive_pos.fill(-1)
        if self.alive_count:
            self.alive_pos[indices] = np.arange(self.alive_count, dtype=np.int32)
        self.alive_dirty = False
        return self.alive_count

    def _ensure_alive(self) -> None:
        if self.alive_dirty or self.alive_idx.size != self.field.cells:
            self.rebuild_alive()

    def _alive_append(self, index: int) -> None:
        if self.alive_pos[index] >= 0:
            return
        if self.alive_count >= self.alive_idx.size:
            self.alive_dirty = True
            return
        self.alive_idx[self.alive_count] = index
        self.alive_pos[index] = self.alive_count
        self.alive_count += 1

    def _alive_remove(self, index: int) -> None:
        pos = int(self.alive_pos[index])
        if pos < 0:
            return
        last = int(self.alive_idx[self.alive_count - 1])
        self.alive_idx[pos] = last
        self.alive_pos[last] = pos
        self.alive_pos[index] = -1
        self.alive_count -= 1

    def initialize_lifetimes(self) -> None:
        """Назначить сроки клеткам без `die_at` (демо, старый чекпоинт)."""
        if self.lifetimes_ready:
            return
        f = self.field
        ct = f.cell_type
        missing = (f.die_at == 0) & (ct != config.EMPTY) & (ct != config.SEED)
        if missing.any():
            factor = config.LIFETIME_TICKS_PER_ENERGY
            workers = missing & (
                (ct == config.GREEN) | (ct == config.ORANGE) | (ct == config.BLUE)
            )
            if workers.any():
                f.die_at[workers] = self.tick_count + int(
                    config.WORKER_CREATE_COST * factor
                )
        # У стеблей нет срока жизни; старые чекпоинты могли хранить die_at.
        stems = ct == config.STEM
        if stems.any():
            f.die_at[stems] = 0
            # Миграция старых чекпоинтов с ёмкостью стебля 100.
            old_capacity = stems & (f.capacity == 100.0)
            if old_capacity.any():
                f.capacity[old_capacity] = config.STEM_CAPACITY
        self.lifetimes_ready = True

    def _schedule_lifetime(self, y: int, x: int, creation_energy: float) -> None:
        if creation_energy <= 0.0:
            self.field.die_at[y, x] = 0
            return
        self.field.die_at[y, x] = self.tick_count + int(
            creation_energy * config.LIFETIME_TICKS_PER_ENERGY
        )

    def _remove_seed_at(self, y: int, x: int) -> bool:
        w = self.field.width
        for i, state in enumerate(self.seeds):
            if state.y == y and state.x == x:
                last = self.seeds[-1]
                self.seeds[i] = last
                self.seeds.pop()
                self.seed_slots.pop(state.uid, None)
                if i < len(self.seeds):
                    self.seed_slots[last.uid] = i
                    self.seed_idx[i] = last.y * w + last.x
                self.seed_idx_count = len(self.seeds)
                return True
        return False

    def _queue_overflow(self, y: int, x: int) -> None:
        """Поставить ячейку в очередь на проверку избытка ресурса."""
        if self.overflow_seen[y, x]:
            return
        if self.overflow_count >= self.overflow_idx.size:
            return
        self.overflow_seen[y, x] = 1
        self.overflow_idx[self.overflow_count] = y * self.field.width + x
        self.overflow_count += 1

    def queue_all_overflow(self) -> None:
        """Один раз собрать все переполненные ячейки (старт/load/resize)."""
        f = self.field
        threshold = config.RESOURCE_OVERFLOW_THRESHOLD
        self.overflow_seen.fill(0)
        self.overflow_count = 0
        mask = (f.soil_energy > threshold) | (f.organic > threshold)
        for idx in np.flatnonzero(mask.ravel()).tolist():
            self._queue_overflow(idx // f.width, idx % f.width)

    def _destroy_cell(self, y: int, x: int) -> int:
        """Уничтожить клетку, рассеяв её энергию и оставив органику."""
        f = self.field
        code = int(f.cell_type[y, x])
        if code == config.EMPTY:
            return config.EMPTY

        self._ensure_alive()
        spill = float(f.energy[y, x])
        side = config.RESOURCE_RADIUS * 2 + 1
        area = side * side
        energy_part = spill / area if spill > 0.0 else 0.0
        organic_part = config.DEATH_ORGANIC / area
        for dy in range(-config.RESOURCE_RADIUS, config.RESOURCE_RADIUS + 1):
            ny = (y + dy) % f.height
            for dx in range(-config.RESOURCE_RADIUS, config.RESOURCE_RADIUS + 1):
                nx = (x + dx) % f.width
                f.soil_energy[ny, nx] += energy_part
                f.organic[ny, nx] += organic_part
                self._queue_overflow(ny, nx)

        f.cell_type[y, x] = config.EMPTY
        f.energy[y, x] = 0.0
        f.capacity[y, x] = 0.0
        f.organism_id[y, x] = 0
        f.dist_seed[y, x] = -1
        f.die_at[y, x] = 0
        self._alive_remove(y * f.width + x)

        if code == config.SEED:
            self._remove_seed_at(y, x)
        elif code in (config.GREEN, config.ORANGE, config.BLUE):
            self.prune_workers()

        self.dirty = True
        f.touch()
        return code

    def prune_stems(self) -> int:
        """Удалить стебли, которые не ведут к семени/работнику.

        Вызывать только при актуальном BFS: ядро использует `dist_seed`.
        """
        f = self.field
        pruned, self.alive_count, self.overflow_count = (
            rules.prune_dead_stems(
                f.cell_type,
                f.energy,
                f.organic,
                f.soil_energy,
                f.capacity,
                f.organism_id,
                f.dist_seed,
                f.die_at,
                self.alive_idx,
                self.alive_pos,
                self.alive_count,
                self.stem_degree,
                self.prune_queue,
                self.tick_count,
                config.STEM_PRUNE_GRACE,
                self.overflow_idx,
                self.overflow_seen,
                self.overflow_count,
                config.STEM,
                config.EMPTY,
                config.DEATH_ORGANIC,
                config.RESOURCE_RADIUS,
                config.RESOURCE_TOXIC_THRESHOLD,
            )
        )
        if pruned:
            self.dirty = True
            f.touch()
        return pruned

    def _seed_valid(self, state: SeedState) -> bool:
        f = self.field
        return (
            0 <= state.y < f.height
            and 0 <= state.x < f.width
            and f.cell_type[state.y, state.x] == config.SEED
        )

    def _new_seed_uid(self) -> int:
        uid = self._next_seed_uid
        self._next_seed_uid += 1
        return uid

    def _new_org_id(self) -> int:
        org_id = self.next_org_id
        self.next_org_id += 1
        return org_id

    def _rebuild_seed_index(self) -> None:
        """Пересобрать плоские индексы семян и слоты из списка состояний."""
        count = len(self.seeds)
        if self.seed_idx.size < count:
            self.seed_idx = np.empty(max(16, count), dtype=np.int32)
        w = self.field.width
        self.seed_slots = {}
        for i, state in enumerate(self.seeds):
            self.seed_idx[i] = state.y * w + state.x
            self.seed_slots[state.uid] = i
        self.seed_idx_count = count

    def _update_seed_position(self, state: SeedState) -> None:
        """Обновить один слот после хода семени без полной пересборки."""
        slot = self.seed_slots.get(state.uid)
        if slot is None:
            self._rebuild_seed_index()
            return
        self.seed_idx[slot] = state.y * self.field.width + state.x

    def _append_seed_state(self, state: SeedState) -> None:
        """Добавить семя и его слот без пересборки всего индекса."""
        slot = len(self.seeds)
        if slot >= self.seed_idx.size:
            new_size = max(16, (slot + 1) * 2)
            old = self.seed_idx
            self.seed_idx = np.empty(new_size, dtype=np.int32)
            self.seed_idx[: old.size] = old
        self.seeds.append(state)
        self.seed_slots[state.uid] = slot
        self.seed_idx[slot] = state.y * self.field.width + state.x
        self.seed_idx_count = slot + 1

    def refresh_seeds(self) -> None:
        """Синхронизировать список семян с полем.

        Существующие состояния ищутся по позиции и организму, поэтому genomes
        и поколения не теряются. Для новых клеток-семян создаётся стартовый
        или случайный геном. Полный скан поля выполняется только при внешних
        изменениях.
        """
        f = self.field
        ys, xs = np.nonzero(f.cell_type == config.SEED)
        existing = {(s.y, s.x, s.org_id): s for s in self.seeds}
        seeds: list[SeedState] = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            org_id = int(f.organism_id[y, x])
            state = existing.pop((y, x, org_id), None)
            if state is None:
                state = SeedState(
                    uid=self._new_seed_uid(),
                    y=y,
                    x=x,
                    org_id=org_id,
                    genome=(
                        Genome.random(
                            self.rng, spread=config.START_GENOME_SPREAD
                        )
                        if self.random_initial_genomes
                        else Genome()
                    ),
                    generation=0,
                )
            seeds.append(state)
        self.seeds = seeds
        self._rebuild_seed_index()
        self.next_org_id = max(self.next_org_id, int(f.organism_id.max()) + 1)
        self.seeds_dirty = False
        self.dirty = True

    def _sync_seeds(self) -> None:
        if self.seeds_dirty:
            self.refresh_seeds()

    def on_field_changed(self) -> None:
        """Вызывать после resize/load/ручного изменения топологии."""
        self._ensure_scratch()
        self.seeds_dirty = True
        self.refresh_seeds()
        self.workers_dirty = True
        self.alive_dirty = True
        self.lifetimes_ready = False
        self.queue_all_overflow()

    # ------------------------------------------------ совместимость со старым API

    @property
    def seed_y(self) -> int:
        return self.seeds[0].y if self.seeds else -1

    @property
    def seed_x(self) -> int:
        return self.seeds[0].x if self.seeds else -1

    @property
    def org_id(self) -> int:
        return self.seeds[0].org_id if self.seeds else 0

    # ---------------------------------------------------------------------- тик

    def rebuild_bfs(self) -> int:
        """Мульти-источниковый BFS от всех семян по стеблям их организмов."""
        self._sync_seeds()
        self.bfs_count = energy_kernels.build_bfs(
            self.field.cell_type,
            self.field.organism_id,
            self.field.dist_seed,
            self.parent_dir,
            self.queue,
            self.seed_idx,
            self.seed_idx_count,
            config.STEM,
            config.SEED,
        )
        self.dirty = False
        return self.bfs_count

    def rebuild_workers(self) -> int:
        """Собрать плоские индексы всех работников (зелёные/оранжевые/синие)."""
        ct = self.field.cell_type
        mask = (
            (ct == config.GREEN) | (ct == config.ORANGE) | (ct == config.BLUE)
        ).ravel()
        indices = np.flatnonzero(mask)
        self.worker_count = int(indices.size)
        self.worker_idx[: self.worker_count] = indices
        self.workers_dirty = False
        return self.worker_count

    def prune_workers(self) -> int:
        """Убрать из списка умерших работников без полного скана поля."""
        if self.worker_count <= 0:
            return 0
        current = self.worker_idx[: self.worker_count]
        codes = self.field.cell_type.ravel()[current]
        keep = (
            (codes == config.GREEN)
            | (codes == config.ORANGE)
            | (codes == config.BLUE)
        )
        kept = current[keep]
        self.worker_count = int(kept.size)
        self.worker_idx[: self.worker_count] = kept
        return self.worker_count

    # --------------------------------------------------------- статистика организмов

    def _refresh_org_counts(self) -> None:
        """Посчитать семена/стебли/работников по организмам за текущую топологию."""
        counts: dict[int, list[int]] = {}
        self.org_stem_ranges = {}
        for state in self.seeds:
            counts.setdefault(state.org_id, [0, 0, 0])[0] += 1

        if self.bfs_count > 0:
            idx = self.queue[: self.bfs_count]
            types = self.field.cell_type.ravel()[idx]
            stems = idx[types == config.STEM]
            if stems.size:
                # Обратный BFS-порядок внутри каждого организма: дальние
                # стебли первыми. Это позволяет каждому семени перебирать
                # только стебли своего организма, а не всю очередь.
                stems = stems[::-1]
                orgs = self.field.organism_id.ravel()[stems]
                order = np.argsort(orgs, kind="stable")
                stems = stems[order]
                orgs = orgs[order]
                self.org_stem_idx[: stems.size] = stems
                unique, amounts = np.unique(orgs, return_counts=True)
                cursor = 0
                for org, amount in zip(unique.tolist(), amounts.tolist()):
                    end = cursor + int(amount)
                    self.org_stem_ranges[int(org)] = (cursor, end)
                    counts.setdefault(int(org), [0, 0, 0])[1] += int(amount)
                    cursor = end

        if self.worker_count > 0:
            idx = self.worker_idx[: self.worker_count]
            codes = self.field.cell_type.ravel()[idx]
            keep = (
                (codes == config.GREEN)
                | (codes == config.ORANGE)
                | (codes == config.BLUE)
            )
            idx = idx[keep]
            if idx.size:
                orgs = self.field.organism_id.ravel()[idx]
                unique, amounts = np.unique(orgs, return_counts=True)
                for org, amount in zip(unique.tolist(), amounts.tolist()):
                    counts.setdefault(int(org), [0, 0, 0])[2] += int(amount)

        self._org_counts = counts

    def _org_stat(self, org_id: int) -> list[int]:
        return self._org_counts.setdefault(org_id, [0, 0, 0])

    # ------------------------------------------------------------ действия семени

    def _worker_code_for_seed(self, state: SeedState, workers: int) -> int:
        """Выбрать тип работника по весам генома.

        При равных весах типы детерминированно чередуются, поэтому стартовая
        популяция строит все три вида. Гены с меньшим весом проигрывают
        максимальному и постепенно вытесняются.
        """
        genome = state.genome
        weighted = (
            (config.GREEN, genome.green_weight),
            (config.ORANGE, genome.orange_weight),
            (config.BLUE, genome.blue_weight),
        )
        best = max(weight for _, weight in weighted)
        if best <= 0.0:
            candidates = [config.GREEN, config.ORANGE, config.BLUE]
        else:
            candidates = [
                code for code, weight in weighted if weight >= best - 1e-9
            ]
        shift = self.tick_count + workers + state.uid
        return candidates[shift % len(candidates)]

    def _ensure_bfs_for_action(self) -> None:
        if self.dirty and not self._seed_phase:
            self.rebuild_bfs()

    def seed_create_auto(self, code: int) -> bool:
        """Создать работника для первого семени.

        Стебли семя напрямую не создаёт: они появляются только при движении.
        """
        if code not in (config.GREEN, config.ORANGE, config.BLUE):
            raise ValueError(f"семя не может создать клетку кода {code}")
        self._sync_seeds()
        if not self.seeds:
            return False
        state = self.seeds[0]
        if not self._seed_valid(state):
            return False
        self._ensure_bfs_for_action()
        self._refresh_org_counts()
        return self._seed_create_worker(state, code)

    def _seed_create_worker(self, state: SeedState, code: int) -> bool:
        if not self._seed_valid(state):
            return False

        f = self.field
        if float(f.energy[state.y, state.x]) < config.WORKER_CREATE_COST:
            return False

        start, end = self.org_stem_ranges.get(state.org_id, (0, 0))
        found, ty, tx = seed_kernels.find_worker_target(
            f.cell_type,
            self.org_stem_idx,
            start,
            end,
            config.STEM,
            config.EMPTY,
            self.tick_count & 3,
        )
        if not found:
            return False

        f.energy[state.y, state.x] -= config.WORKER_CREATE_COST
        f.cell_type[ty, tx] = code
        f.energy[ty, tx] = 0.0
        f.capacity[ty, tx] = config.WORKER_CAPACITY
        f.organism_id[ty, tx] = state.org_id
        f.dist_seed[ty, tx] = -1
        self._schedule_lifetime(ty, tx, config.WORKER_CREATE_COST)
        self._alive_append(ty * f.width + tx)

        if self.worker_count < self.worker_idx.size:
            self.worker_idx[self.worker_count] = ty * f.width + tx
            self.worker_count += 1
        else:
            self.workers_dirty = True

        self._org_stat(state.org_id)[2] += 1
        self.dirty = True
        f.touch()
        return True

    def _apply_seed_move(self, state: SeedState, ny: int, nx: int) -> bool:
        """Переместить семя в пустого соседа, оставив на старом месте стебель.

        Ход оплачивает появившийся стебель (SEED_MOVE_COST).
        """
        if not self._seed_valid(state):
            return False

        f = self.field
        if f.cell_type[ny, nx] != config.EMPTY:
            return False

        current = float(f.energy[state.y, state.x])
        if current < config.SEED_MOVE_COST:
            return False

        sy, sx = state.y, state.x
        f.cell_type[sy, sx] = config.STEM
        f.energy[sy, sx] = 0.0
        f.capacity[sy, sx] = config.STEM_CAPACITY
        f.dist_seed[sy, sx] = -1
        f.die_at[sy, sx] = 0

        f.cell_type[ny, nx] = config.SEED
        f.energy[ny, nx] = current - config.SEED_MOVE_COST
        f.capacity[ny, nx] = 0.0
        f.organism_id[ny, nx] = state.org_id
        f.dist_seed[ny, nx] = -1
        f.die_at[ny, nx] = 0
        self._alive_append(ny * f.width + nx)

        state.y, state.x = ny, nx
        self._update_seed_position(state)
        self._org_stat(state.org_id)[1] += 1
        self.dirty = True
        f.touch()
        return True

    def seed_move(self, dx: int, dy: int) -> bool:
        """Переместить первое семя на одну клетку в заданном направлении."""
        if abs(dx) + abs(dy) != 1:
            raise ValueError("семя ходит ровно на одну клетку по 4 направлениям")
        self._sync_seeds()
        if not self.seeds:
            return False
        state = self.seeds[0]
        f = self.field
        ny = (state.y + dy) % f.height
        nx = (state.x + dx) % f.width
        return self._apply_seed_move(state, ny, nx)

    def _seed_move_auto(self, state: SeedState) -> bool:
        """Сдвинуть семя в первого пустого 4-соседа, оставив стебель."""
        f = self.field
        if float(f.energy[state.y, state.x]) < config.SEED_MOVE_COST:
            return False
        found, ny, nx = seed_kernels.find_move_target(
            f.cell_type,
            state.y,
            state.x,
            config.EMPTY,
            self.tick_count & 3,
        )
        if not found:
            return False
        return self._apply_seed_move(state, ny, nx)

    def _seed_duplicate(self, state: SeedState) -> bool:
        """Создать ещё одно семя в том же организме (с мутацией генома)."""
        if not self._seed_valid(state):
            return False

        counts = self._org_stat(state.org_id)
        if counts[0] >= config.MAX_SEEDS_PER_ORGANISM:
            return False

        genome = state.genome
        child_energy = genome.duplicate_energy
        cost = child_energy + config.DUPLICATE_EXTRA_COST
        f = self.field
        if float(f.energy[state.y, state.x]) < cost:
            return False

        start, end = self.org_stem_ranges.get(state.org_id, (0, 0))
        found, ty, tx = seed_kernels.find_seed_target(
            f.cell_type,
            self.org_stem_idx,
            start,
            end,
            config.STEM,
            config.EMPTY,
            self.tick_count & 3,
        )
        if not found:
            # Запасной вариант: рядом с родительским семенем.
            found, ty, tx = seed_kernels.find_move_target(
                f.cell_type,
                state.y,
                state.x,
                config.EMPTY,
                self.tick_count & 3,
            )
        if not found:
            return False

        f.energy[state.y, state.x] -= cost
        f.cell_type[ty, tx] = config.SEED
        f.energy[ty, tx] = child_energy
        f.capacity[ty, tx] = 0.0
        f.organism_id[ty, tx] = state.org_id
        f.dist_seed[ty, tx] = -1
        f.die_at[ty, tx] = 0
        self._alive_append(ty * f.width + tx)

        child = SeedState(
            uid=self._new_seed_uid(),
            y=ty,
            x=tx,
            org_id=state.org_id,
            genome=genome.mutated(self.rng),
            generation=state.generation + 1,
        )
        self._append_seed_state(child)
        counts[0] += 1
        state.last_duplicate_tick = self.tick_count
        self.dirty = True
        f.touch()
        return True

    def _seed_shoot(self, state: SeedState) -> bool:
        """Отстрелить новое семя — нового организма с мутировавшим геномом.

        Потомок появляется рядом со стеблем родителя и сразу начинает полёт
        в направлении от стебля. В полёте он не оставляет стеблей.
        """
        if not self._seed_valid(state):
            return False

        genome = state.genome
        child_energy = genome.child_energy
        cost = child_energy + config.SHOOT_EXTRA_COST
        f = self.field
        if float(f.energy[state.y, state.x]) < cost:
            return False

        start, end = self.org_stem_ranges.get(state.org_id, (0, 0))
        found, ty, tx, direction = seed_kernels.find_shoot_target(
            f.cell_type,
            self.org_stem_idx,
            start,
            end,
            config.STEM,
            config.EMPTY,
            self.tick_count & 3,
        )
        if not found:
            return False

        if direction == 0:
            travel_dy, travel_dx = -1, 0
        elif direction == 1:
            travel_dy, travel_dx = 1, 0
        elif direction == 2:
            travel_dy, travel_dx = 0, -1
        else:
            travel_dy, travel_dx = 0, 1

        f.energy[state.y, state.x] -= cost
        f.cell_type[ty, tx] = config.SEED
        f.energy[ty, tx] = child_energy
        f.capacity[ty, tx] = 0.0
        f.die_at[ty, tx] = 0
        org_id = self._new_org_id()
        f.organism_id[ty, tx] = org_id
        f.dist_seed[ty, tx] = -1
        self._alive_append(ty * f.width + tx)

        child = SeedState(
            uid=self._new_seed_uid(),
            y=ty,
            x=tx,
            org_id=org_id,
            genome=genome.mutated(self.rng),
            generation=state.generation + 1,
            established=False,
            travel_dy=travel_dy,
            travel_dx=travel_dx,
            travel_steps=0,
            travel_limit=max(1, int(genome.shoot_range)),
        )
        self._append_seed_state(child)
        self._org_counts[org_id] = [1, 0, 0]
        state.last_shoot_tick = self.tick_count
        self.dirty = True
        f.touch()
        return True

    def _seed_travel_action(self, state: SeedState) -> int:
        """Полёт семени после отстрела: без стеблей, до энергии или столкновения."""
        if not self._seed_valid(state):
            return config.SEED_ACTION_NONE

        f = self.field
        energy = float(f.energy[state.y, state.x])
        if (
            state.travel_steps >= state.travel_limit
            or energy <= config.SEED_ESTABLISH_ENERGY
            or energy < config.SEED_TRAVEL_COST
        ):
            state.established = True
            return config.SEED_ACTION_ESTABLISH

        ny = (state.y + state.travel_dy) % f.height
        nx = (state.x + state.travel_dx) % f.width
        if f.cell_type[ny, nx] != config.EMPTY:
            # Столкновение: разрушаем клетку на пути и само семя.
            self._destroy_cell(ny, nx)
            self._destroy_cell(state.y, state.x)
            return config.SEED_ACTION_COLLIDE

        oy, ox = state.y, state.x
        self._alive_remove(oy * f.width + ox)
        f.cell_type[oy, ox] = config.EMPTY
        f.energy[oy, ox] = 0.0
        f.capacity[oy, ox] = 0.0
        f.organism_id[oy, ox] = 0
        f.dist_seed[oy, ox] = -1
        f.die_at[oy, ox] = 0

        f.cell_type[ny, nx] = config.SEED
        f.energy[ny, nx] = energy - config.SEED_TRAVEL_COST
        f.capacity[ny, nx] = 0.0
        f.organism_id[ny, nx] = state.org_id
        f.dist_seed[ny, nx] = -1
        f.die_at[ny, nx] = 0
        self._alive_append(ny * f.width + nx)

        state.y, state.x = ny, nx
        state.travel_steps += 1
        self._update_seed_position(state)
        self.dirty = True
        f.touch()
        return config.SEED_ACTION_TRAVEL

    def _seed_action(self, state: SeedState) -> int:
        """Одно решение семени за тик по его геному."""
        if not state.established:
            return self._seed_travel_action(state)

        f = self.field
        energy = float(f.energy[state.y, state.x])
        genome = state.genome
        counts = self._org_stat(state.org_id)

        population_full = len(self.seeds) >= config.MAX_TOTAL_SEEDS
        can_shoot = (
            not population_full
            and self.tick_count - state.last_shoot_tick
            >= config.SEED_SHOOT_COOLDOWN
        )
        if (
            can_shoot
            and energy >= genome.shoot_threshold
            and self._seed_shoot(state)
        ):
            return config.SEED_ACTION_SHOOT

        can_duplicate = (
            not population_full
            and self.tick_count - state.last_duplicate_tick
            >= config.SEED_DUPLICATE_COOLDOWN
        )
        if (
            can_duplicate
            and energy >= genome.duplicate_threshold
            and counts[0] < config.MAX_SEEDS_PER_ORGANISM
            and self._seed_duplicate(state)
        ):
            return config.SEED_ACTION_DUPLICATE

        workers = counts[2]
        target_stems = genome.stems_per_worker * workers + genome.stem_slack
        prefer_move = counts[1] < target_stems
        if (
            not prefer_move
            and genome.move_bias > 0.0
            and self.rng.random() < genome.move_bias
        ):
            prefer_move = True

        worker_code = self._worker_code_for_seed(state, workers)
        if prefer_move:
            if self._seed_move_auto(state):
                return config.SEED_ACTION_MOVE
            if self._seed_create_worker(state, worker_code):
                return worker_code
        else:
            if self._seed_create_worker(state, worker_code):
                return worker_code
            if self._seed_move_auto(state):
                return config.SEED_ACTION_MOVE

        return config.SEED_ACTION_NONE

    def apply_toxic_seed_deaths(self) -> int:
        """Убить семена, оказавшиеся в токсичной почве."""
        self._sync_seeds()
        if not self.seeds:
            return 0

        died = 0
        f = self.field
        threshold = config.RESOURCE_TOXIC_THRESHOLD
        for state in list(self.seeds):
            if not self._seed_valid(state):
                continue
            if (
                f.soil_energy[state.y, state.x] >= threshold
                or f.organic[state.y, state.x] >= threshold
            ):
                self._destroy_cell(state.y, state.x)
                died += 1

        if died:
            self.dirty = True
        return died

    def apply_seed_upkeep(self) -> int:
        """Списать энергию за жизнь семян; нехватка энергии убивает семя."""
        self._sync_seeds()
        if not self.seeds:
            return 0

        alive_states = [s for s in self.seeds if self._seed_valid(s)]
        if len(alive_states) != len(self.seeds):
            # Редкий случай внешней рассинхронизации: пересобираем индексы.
            self.seeds = alive_states
            self._rebuild_seed_index()

        died = 0
        f = self.field
        for state in list(self.seeds):
            energy = float(f.energy[state.y, state.x])
            if energy <= config.SEED_UPKEEP:
                self._destroy_cell(state.y, state.x)
                died += 1
            else:
                f.energy[state.y, state.x] = energy - config.SEED_UPKEEP

        if died:
            self.dirty = True
        return died

    def _mutate_seeds(self) -> None:
        """Небольшая мутация одного гена у каждого семени раз в тик."""
        for state in list(self.seeds):
            if state.last_mutation_tick == self.tick_count:
                continue
            state.last_mutation_tick = self.tick_count
            if not self._seed_valid(state):
                continue
            if self.rng.random() < config.TICK_MUTATION_CHANCE:
                state.genome.mutate_one(self.rng)

    def seed_step(self) -> int:
        """Одно действие первого семени за тик (совместимость и отладка)."""
        self._sync_seeds()
        if not self.seeds:
            return config.SEED_ACTION_NONE
        state = self.seeds[0]
        if not self._seed_valid(state):
            return config.SEED_ACTION_NONE
        self._ensure_bfs_for_action()
        self._refresh_org_counts()
        action = self._seed_action(state)
        self.last_seed_action = action
        return action

    def seed_step_all(self) -> int:
        """Дать по одному действию каждому семени текущего тика.

        Список семян фиксируется до начала фазы: потомки, появившиеся от
        отстрела или размножения, начнут действовать только со следующего тика.
        """
        self._sync_seeds()
        if not self.seeds:
            return 0
        self._mutate_seeds()
        self._ensure_bfs_for_action()
        self._refresh_org_counts()

        acted = 0
        self._seed_phase = True
        try:
            for state in list(self.seeds):
                if not self._seed_valid(state):
                    continue
                action = self._seed_action(state)
                if action != config.SEED_ACTION_NONE:
                    acted += 1
                    self.last_seed_action = action
        finally:
            self._seed_phase = False
        return acted

    def tick(self) -> int:
        f = self.field
        self._ensure_scratch()
        self._sync_seeds()
        self.initialize_lifetimes()
        self._ensure_alive()
        if self.dirty:
            self.rebuild_bfs()
        if self.workers_dirty:
            self.rebuild_workers()

        rules.produce(
            f.energy,
            f.cell_type,
            f.organic,
            f.soil_energy,
            self.produced,
            self.worker_idx,
            self.worker_count,
            config.GREEN,
            config.ORANGE,
            config.BLUE,
            config.GREEN_RATE,
            config.WORKER_CAPACITY,
            config.ORANGE_MAX_INTAKE,
            config.ORANGE_EFFICIENCY,
            config.BLUE_MAX_INTAKE,
            config.BLUE_EFFICIENCY,
            config.RESOURCE_RADIUS,
            config.GREEN_NEIGHBOR_EFFICIENCY[1],
            config.GREEN_NEIGHBOR_EFFICIENCY[2],
            config.STEM,
            config.EMPTY,
        )

        energy_kernels.inject_workers(
            f.energy,
            f.cell_type,
            f.capacity,
            f.dist_seed,
            f.organism_id,
            self.worker_idx,
            self.worker_count,
            config.GREEN,
            config.ORANGE,
            config.BLUE,
            config.STEM,
        )

        energy_kernels.flow_stems(
            f.energy,
            f.cell_type,
            f.capacity,
            f.dist_seed,
            self.parent_dir,
            self.queue,
            self.bfs_count,
            config.SEED_CAPACITY,
            config.STEM,
            config.SEED,
        )

        # Сначала стебли: тогда работник, оставшийся без стебля, умирает
        # в этом же тике.
        self.prune_stems()

        died, self.alive_count, self.overflow_count = rules.apply_deaths(
            f.cell_type,
            f.energy,
            f.organic,
            f.soil_energy,
            f.capacity,
            f.organism_id,
            f.dist_seed,
            f.die_at,
            self.produced,
            self.alive_idx,
            self.alive_pos,
            self.alive_count,
            self.tick_count,
            self.overflow_idx,
            self.overflow_seen,
            self.overflow_count,
            config.GREEN,
            config.ORANGE,
            config.BLUE,
            config.STEM,
            config.EMPTY,
            config.DEATH_ORGANIC,
            config.RESOURCE_RADIUS,
            config.RESOURCE_TOXIC_THRESHOLD,
        )
        if died:
            self.dirty = True
            # Чистим список сразу: иначе семя может занять место умершего
            # работника и добавить уже существующий индекс повторно.
            self.prune_workers()

        # Семена в токсичной почве гибнут до своих действий.
        if self.apply_toxic_seed_deaths():
            self.rebuild_bfs()
            self.prune_stems()

        if self.seed_actions_enabled:
            # Поддержание жизни семян может убрать источники BFS.
            upkeep_died = self.apply_seed_upkeep()
            if upkeep_died:
                # Организм остался без семени — сразу убираем его стебли.
                self.rebuild_bfs()
                self.prune_stems()
            self.seed_step_all()

        # Избыток органики/энергии разливаем из очереди по соседям.
        self.overflow_count, _ = rules.spread_overflow_queue(
            f.soil_energy,
            f.organic,
            self.overflow_idx,
            self.overflow_count,
            self.overflow_seen,
            config.RESOURCE_OVERFLOW_THRESHOLD,
        )

        self.tick_count += 1
        f.touch()
        return died

    # ------------------------------------------------------------------ статистика

    def stats(self) -> dict[str, object]:
        f = self.field
        counts = np.bincount(f.cell_type.ravel(), minlength=len(config.CELL_NAMES))
        seed_energy = 0.0
        for state in self.seeds:
            if self._seed_valid(state):
                seed_energy += float(f.energy[state.y, state.x])

        occupied = f.cell_type != config.EMPTY
        if occupied.any():
            organism_count = int(np.unique(f.organism_id[occupied]).size)
        else:
            organism_count = 0

        generations = [state.generation for state in self.seeds]
        return {
            "tick": self.tick_count,
            "seed_energy": seed_energy,
            "seed_count": len(self.seeds),
            "organism_count": organism_count,
            "generation_max": max(generations, default=0),
            "generation_avg": (
                sum(generations) / len(generations) if generations else 0.0
            ),
            "total_energy": float(f.energy.sum()),
            "counts": {code: int(counts[code]) for code in config.CELL_NAMES},
            "dirty": self.dirty,
        }
