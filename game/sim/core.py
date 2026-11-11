"""Оркестрация тика симуляции."""

from __future__ import annotations

import numpy as np

from .. import config
from ..field import Field
from . import energy as energy_kernels
from . import rules
from . import seed as seed_kernels


class Simulation:
    """Тик организма: BFS -> производство -> инъекция -> переток -> смерть -> семя."""

    def __init__(self, field: Field, seed_actions: bool = True) -> None:
        self.field = field
        self.seed_actions_enabled = seed_actions
        self.tick_count = 0
        self.dirty = True
        self.bfs_count = 0

        self.queue = np.empty(field.cells, dtype=np.int32)
        self.parent_dir = np.full(field.shape, -1, dtype=np.int8)
        self.produced = np.zeros(field.shape, dtype=np.uint8)
        self.worker_idx = np.empty(field.cells, dtype=np.int32)
        self.worker_count = 0
        self.workers_dirty = True

        self.seed_y = -1
        self.seed_x = -1
        self.org_id = 0
        self.last_seed_action = 0
        self.refresh_seed()

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
            self.dirty = True

    def _seed_valid(self) -> bool:
        f = self.field
        return (
            0 <= self.seed_y < f.height
            and 0 <= self.seed_x < f.width
            and f.cell_type[self.seed_y, self.seed_x] == config.SEED
        )

    def refresh_seed(self) -> None:
        """Найти первое семя на поле и запомнить его организм."""
        ys, xs = np.nonzero(self.field.cell_type == config.SEED)
        if len(ys) > 0:
            self.seed_y = int(ys[0])
            self.seed_x = int(xs[0])
            self.org_id = int(self.field.organism_id[self.seed_y, self.seed_x])
        else:
            self.seed_y = -1
            self.seed_x = -1
            self.org_id = 0
        self.dirty = True

    def on_field_changed(self) -> None:
        """Вызывать после resize/load/ручного изменения топологии."""
        self._ensure_scratch()
        self.refresh_seed()
        self.workers_dirty = True

    # ---------------------------------------------------------------------- тик

    def rebuild_bfs(self) -> int:
        if not self._seed_valid():
            self.refresh_seed()
        if self._seed_valid():
            seed_y, seed_x = self.seed_y, self.seed_x
        else:
            seed_y, seed_x = -1, -1
        self.bfs_count = energy_kernels.build_bfs(
            self.field.cell_type,
            self.field.organism_id,
            self.field.dist_seed,
            self.parent_dir,
            self.queue,
            seed_y,
            seed_x,
            self.org_id,
            config.STEM,
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

    # ------------------------------------------------------------ действия семени

    def _worker_code_for_tick(self) -> int:
        """Детерминированно чередовать тип нового работника."""
        return (config.GREEN, config.ORANGE, config.BLUE)[
            (self.tick_count + self.worker_count) % 3
        ]

    def seed_create_auto(self, code: int) -> bool:
        """Создать клетку в первой подходящей пустой клетке рядом со стеблем.

        Стоимость списывается с энергии семени. Для работников проверяется
        запрет соседства с другими работниками по 8 направлениям.
        """
        if code not in (
            config.STEM,
            config.GREEN,
            config.ORANGE,
            config.BLUE,
        ):
            raise ValueError(f"семя не может создать клетку кода {code}")

        if not self._seed_valid():
            return False

        f = self.field
        cost = (
            config.STEM_CREATE_COST
            if code == config.STEM
            else config.WORKER_CREATE_COST
        )
        if float(f.energy[self.seed_y, self.seed_x]) < cost:
            return False

        if self.dirty:
            self.rebuild_bfs()

        found, ty, tx = seed_kernels.find_create_target(
            f.cell_type,
            self.queue,
            self.bfs_count,
            code != config.STEM,
            config.GREEN,
            config.ORANGE,
            config.BLUE,
            config.STEM,
            config.EMPTY,
            self.tick_count & 3,
        )
        if not found:
            return False

        f.energy[self.seed_y, self.seed_x] -= cost
        f.cell_type[ty, tx] = code
        f.energy[ty, tx] = 0.0
        f.capacity[ty, tx] = (
            config.STEM_CAPACITY if code == config.STEM else config.WORKER_CAPACITY
        )
        f.organism_id[ty, tx] = self.org_id
        f.dist_seed[ty, tx] = -1

        if code != config.STEM:
            if self.worker_count < self.worker_idx.size:
                self.worker_idx[self.worker_count] = ty * f.width + tx
                self.worker_count += 1
            else:
                self.workers_dirty = True

        self.dirty = True
        f.touch()
        return True

    def _apply_seed_move(self, ny: int, nx: int) -> bool:
        """Переместить семя в пустого соседа, оставив на старом месте стебель."""
        if not self._seed_valid():
            return False

        f = self.field
        if f.cell_type[ny, nx] != config.EMPTY:
            return False

        cost = config.SEED_MOVE_COST
        current = float(f.energy[self.seed_y, self.seed_x])
        if current < cost:
            return False

        sy, sx = self.seed_y, self.seed_x
        f.cell_type[sy, sx] = config.STEM
        f.energy[sy, sx] = 0.0
        f.capacity[sy, sx] = config.STEM_CAPACITY
        f.dist_seed[sy, sx] = -1

        f.cell_type[ny, nx] = config.SEED
        f.energy[ny, nx] = current - cost
        f.capacity[ny, nx] = 0.0
        f.organism_id[ny, nx] = self.org_id
        f.dist_seed[ny, nx] = -1

        self.seed_y, self.seed_x = ny, nx
        self.dirty = True
        f.touch()
        return True

    def seed_move(self, dx: int, dy: int) -> bool:
        """Переместить семя на одну клетку в заданном направлении."""
        if abs(dx) + abs(dy) != 1:
            raise ValueError("семя ходит ровно на одну клетку по 4 направлениям")
        if not self._seed_valid():
            return False
        f = self.field
        ny = (self.seed_y + dy) % f.height
        nx = (self.seed_x + dx) % f.width
        return self._apply_seed_move(ny, nx)

    def seed_step(self) -> int:
        """Одно действие семени за тик (default-политика для этапа 4b).

        Возвращает код созданной клетки, SEED_ACTION_MOVE при ходе
        или 0, если действий не было.
        """
        if not self._seed_valid():
            return 0

        f = self.field
        energy = float(f.energy[self.seed_y, self.seed_x])
        stems = max(0, self.bfs_count - 1)
        workers = self.worker_count
        worker_code = self._worker_code_for_tick()

        want_stem = stems < config.SEED_STEMS_PER_WORKER * workers + 3

        if want_stem:
            if self.seed_create_auto(config.STEM):
                self.last_seed_action = config.STEM
                return config.STEM
            if self.seed_create_auto(worker_code):
                self.last_seed_action = worker_code
                return worker_code
        else:
            if self.seed_create_auto(worker_code):
                self.last_seed_action = worker_code
                return worker_code
            if self.seed_create_auto(config.STEM):
                self.last_seed_action = config.STEM
                return config.STEM

        # Если строить некуда, но энергии хватает хотя бы на стебель,
        # семя может сдвинуться и оставить за собой стебель.
        if energy >= config.STEM_CREATE_COST:
            found, ny, nx = seed_kernels.find_move_target(
                f.cell_type,
                self.seed_y,
                self.seed_x,
                config.EMPTY,
                self.tick_count & 3,
            )
            if found and self._apply_seed_move(ny, nx):
                self.last_seed_action = config.SEED_ACTION_MOVE
                return config.SEED_ACTION_MOVE

        self.last_seed_action = 0
        return 0

    def tick(self) -> int:
        f = self.field
        self._ensure_scratch()
        if not self._seed_valid():
            self.refresh_seed()
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
        )

        energy_kernels.inject_workers(
            f.energy,
            f.cell_type,
            f.capacity,
            f.dist_seed,
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

        died = rules.apply_starvation_death(
            f.cell_type,
            f.energy,
            f.organic,
            f.capacity,
            f.organism_id,
            f.dist_seed,
            self.produced,
            self.worker_idx,
            self.worker_count,
            config.ORANGE,
            config.BLUE,
            config.EMPTY,
            config.DEATH_ORGANIC,
        )
        if died:
            self.dirty = True
            # Чистим список сразу: иначе семя может занять место умершего
            # работника и добавить уже существующий индекс повторно.
            self.prune_workers()

        if self.seed_actions_enabled:
            self.seed_step()

        self.tick_count += 1
        f.touch()
        return died

    # ------------------------------------------------------------------ статистика

    def stats(self) -> dict[str, object]:
        f = self.field
        counts = np.bincount(f.cell_type.ravel(), minlength=len(config.CELL_NAMES))
        seed_energy = 0.0
        if self._seed_valid():
            seed_energy = float(f.energy[self.seed_y, self.seed_x])
        return {
            "tick": self.tick_count,
            "seed_energy": seed_energy,
            "total_energy": float(f.energy.sum()),
            "counts": {code: int(counts[code]) for code in config.CELL_NAMES},
            "dirty": self.dirty,
        }
