"""Поле: тороидальная сетка и её слои.

Все слои — плоские numpy-массивы (SoA), без объектов на клетку.
Индексация: [y, x] (строки, столбцы), как принято в numpy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


class Field:
    """Тороидальное поле с отдельными слоями для клеток, энергии и почвы."""

    def __init__(self, width: int = config.WIDTH, height: int = config.HEIGHT) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.version = 0

        shape = (self.height, self.width)
        self.cell_type = np.zeros(shape, dtype=np.uint8)
        self.energy = np.zeros(shape, dtype=np.float32)
        self.organic = np.zeros(shape, dtype=np.float32)
        self.soil_energy = np.zeros(shape, dtype=np.float32)
        self.capacity = np.zeros(shape, dtype=np.float32)
        self.organism_id = np.zeros(shape, dtype=np.int32)
        self.dist_seed = np.full(shape, -1, dtype=np.int32)
        # Абсолютный тик, на котором клетка умирает от старости.
        # 0 — срок ещё не назначен; у семян срока нет.
        self.die_at = np.zeros(shape, dtype=np.int32)

    # ------------------------------------------------------------------ базовое

    def touch(self) -> None:
        """Пометить поле изменённым (сбрасывает кэш рендера)."""
        self.version += 1

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def cells(self) -> int:
        return self.width * self.height

    def memory_bytes(self) -> int:
        total = 0
        for name in self.layer_names():
            total += getattr(self, name).nbytes
        return total

    @staticmethod
    def layer_names() -> tuple[str, ...]:
        return (
            "cell_type",
            "energy",
            "organic",
            "soil_energy",
            "capacity",
            "organism_id",
            "dist_seed",
            "die_at",
        )

    def layer(self, name: str) -> np.ndarray:
        if name not in self.layer_names():
            raise KeyError(f"неизвестный слой: {name!r}")
        return getattr(self, name)

    # ------------------------------------------------------------ тороидальность

    def wrap_x(self, x: int) -> int:
        return int(x) % self.width

    def wrap_y(self, y: int) -> int:
        return int(y) % self.height

    def wrap(self, x: int, y: int) -> tuple[int, int]:
        return self.wrap_x(x), self.wrap_y(y)

    def get_cell(self, x: int, y: int) -> int:
        return int(self.cell_type[self.wrap_y(y), self.wrap_x(x)])

    def set_cell(self, x: int, y: int, code: int, energy: float = 0.0) -> None:
        xx, yy = self.wrap(x, y)
        self.cell_type[yy, xx] = code
        self.energy[yy, xx] = energy
        self.touch()

    def neighbors4(self, x: int, y: int) -> tuple[tuple[int, int], ...]:
        """Четыре соседа с учётом зацикливания."""
        return (
            (self.wrap_x(x + 1), y % self.height),
            (self.wrap_x(x - 1), y % self.height),
            (x % self.width, self.wrap_y(y + 1)),
            (x % self.width, self.wrap_y(y - 1)),
        )

    # -------------------------------------------------------------- изменение

    def resize(self, width: int, height: int, keep: bool = True) -> None:
        """Изменить размер поля.

        При keep=True содержимое сохраняется в левом верхнем углу,
        недостающее заполняется нулями, лишнее обрезается.
        """
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height:
            return

        def _resized(array: np.ndarray) -> np.ndarray:
            out = np.zeros((height, width), dtype=array.dtype)
            if keep:
                h = min(self.height, height)
                w = min(self.width, width)
                out[:h, :w] = array[:h, :w]
            return out

        self.cell_type = _resized(self.cell_type)
        self.energy = _resized(self.energy)
        self.organic = _resized(self.organic)
        self.soil_energy = _resized(self.soil_energy)
        self.capacity = _resized(self.capacity)
        self.organism_id = _resized(self.organism_id)
        self.dist_seed = _resized(self.dist_seed)
        self.die_at = _resized(self.die_at)
        self.width = width
        self.height = height
        self.touch()

    def clear(self) -> None:
        self.cell_type.fill(0)
        self.energy.fill(0.0)
        self.organic.fill(0.0)
        self.soil_energy.fill(0.0)
        self.capacity.fill(0.0)
        self.organism_id.fill(0)
        self.dist_seed.fill(-1)
        self.die_at.fill(0)
        self.touch()

    # ------------------------------------------------------------- чекпоинты

    def save(self, path: str | Path) -> Path:
        """Сохранить все слои в сжатый .npz."""
        target = Path(path)
        if target.suffix != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            width=np.int32(self.width),
            height=np.int32(self.height),
            **{name: getattr(self, name) for name in self.layer_names()},
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Field":
        """Загрузить поле из .npz."""
        source = Path(path)
        if source.suffix != ".npz":
            source = source.with_suffix(".npz")
        with np.load(source) as data:
            field = cls(int(data["width"]), int(data["height"]))
            for name in cls.layer_names():
                if name in data:
                    getattr(field, name)[...] = data[name]
                elif name == "die_at":
                    # Старые чекпоинты не знали о сроках жизни.
                    getattr(field, name).fill(0)
                else:
                    raise KeyError(f"в чекпоинте нет слоя {name!r}")
        field.touch()
        return field

    def __repr__(self) -> str:  # pragma: no cover - для отладки
        mb = self.memory_bytes() / (1024 * 1024)
        return f"Field({self.width}x{self.height}, {mb:.1f} MiB, v{self.version})"
