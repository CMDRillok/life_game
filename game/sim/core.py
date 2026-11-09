"""Оркестрация тика симуляции."""

from __future__ import annotations

import numpy as np

from .. import config
from ..field import Field
from . import energy as energy_kernels
from . import rules


class Simulation:
    """Статичный организм: производство -> инъекция -> переток -> смерть."""

    def __init__(self, field: Field) -> None:
        self.field = field
        self.tick_count = 0
        self.dirty = True
        self.bfs_count = 0

        self.queue = np.empty(field.cells, dtype=np.int32)
        self.parent_dir = np.full(field.shape, -1, dtype=np.int8)
        self.produced = np.zeros(field.shape, dtype=np.uint8)
        self.worker_idx = np.empty(field.cells, dtype=np.int32)
        self.worker_count = 0

        self.seed_y = -1
        self.seed_x = -1
        self.org_id = 0
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
        return self.worker_count

    def tick(self) -> int:
        f = self.field
        self._ensure_scratch()
        if not self._seed_valid():
            self.refresh_seed()
        if self.dirty:
            self.rebuild_bfs()
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
