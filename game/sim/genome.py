"""Геном семени и его мутации.

Геном — небольшой вектор числовых параметров. Он не хранится в слоях поля:
на каждое семя приходится один объект `Genome`, а клетки ссылаются на
организм через `organism_id`.

Параметры специально подобраны так, чтобы политика семени была простой и
детерминированной: геном задаёт пороги и веса, а не сложную сеть.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import numpy as np


@dataclass
class Genome:
    """Параметры поведения одного семени."""

    stems_per_worker: float = 2.0
    stem_slack: float = 3.0
    shoot_threshold: float = 280.0
    # Максимальное число клеток, которое пролетает семя после отстрела.
    shoot_range: int = 20
    child_energy: float = 100.0
    duplicate_threshold: float = 180.0
    duplicate_energy: float = 60.0
    green_weight: float = 1.0
    orange_weight: float = 1.0
    blue_weight: float = 1.0
    move_bias: float = 0.15
    # Мутации при размножении слабые: реже и с меньшим шагом.
    mutation_rate: float = 0.04
    mutation_scale: float = 0.05

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, (low, high) in GENOME_BOUNDS.items():
            value = float(values[name])
            if name == "shoot_range":
                value = float(round(value))
            value = min(high, max(low, value))
            setattr(self, name, int(value) if name == "shoot_range" else value)

    def clone(self) -> "Genome":
        return replace(self)

    @classmethod
    def random(
        cls, rng: np.random.Generator, spread: float = 0.35
    ) -> "Genome":
        """Случайный, но жизнеспособный геном вокруг стартовых значений.

        Каждый ген получает равномерное смещение в пределах `spread` от
        полного диапазона и зажимается в границы. Так новый запуск получает
        другой набор параметров, но не экстремальные комбинации, при которых
        одиночное семя не успевает построить организм.
        """
        values = asdict(cls())
        for name, (low, high) in GENOME_BOUNDS.items():
            span = (high - low) * spread
            value = float(values[name]) + float(rng.uniform(-span, span))
            values[name] = min(high, max(low, value))
        # В пустой стартовой почве оранжевые/синие работники не находят еды.
        # Держим веса равными, чтобы семя не строило только нежизнеспособный
        # тип и могло пережить первый этап.
        values["green_weight"] = 1.0
        values["orange_weight"] = 1.0
        values["blue_weight"] = 1.0
        # Мутационные гены тоже не разгоняем на старте.
        values["mutation_rate"] = 0.04
        values["mutation_scale"] = 0.05
        return cls(**values)

    def mutated(self, rng: np.random.Generator) -> "Genome":
        """Вернуть мутировавшую копию.

        Каждый ген меняется с вероятностью `mutation_rate` на нормальную
        величину, пропорциональную диапазону допустимых значений.
        """
        values = asdict(self)
        rate = min(0.5, max(0.0, float(self.mutation_rate)))
        scale = min(0.5, max(0.0, float(self.mutation_scale)))
        for name, (low, high) in GENOME_BOUNDS.items():
            if rng.random() >= rate:
                continue
            value = float(values[name])
            value += float(rng.normal(0.0, scale * (high - low)))
            value = min(high, max(low, value))
            if name == "shoot_range":
                value = float(round(value))
            values[name] = value
        return Genome(**values)

    def mutate_one(self, rng: np.random.Generator) -> str:
        """Мутировать ровно один случайный ген и вернуть его имя."""
        names = tuple(GENOME_BOUNDS)
        name = names[int(rng.integers(len(names)))]
        low, high = GENOME_BOUNDS[name]
        scale = min(0.5, max(0.0, float(self.mutation_scale)))
        value = float(getattr(self, name))
        value += float(rng.normal(0.0, scale * (high - low)))
        value = min(high, max(low, value))
        if name == "shoot_range":
            setattr(self, name, int(round(value)))
        else:
            setattr(self, name, value)
        return name

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


GENOME_BOUNDS: dict[str, tuple[float, float]] = {
    "stems_per_worker": (0.25, 5.0),
    "stem_slack": (0.0, 12.0),
    "shoot_threshold": (60.0, 900.0),
    "shoot_range": (2.0, 50.0),
    "child_energy": (20.0, 250.0),
    "duplicate_threshold": (60.0, 700.0),
    "duplicate_energy": (20.0, 200.0),
    "green_weight": (0.0, 2.0),
    "orange_weight": (0.0, 2.0),
    "blue_weight": (0.0, 2.0),
    "move_bias": (0.0, 1.0),
    "mutation_rate": (0.0, 0.5),
    "mutation_scale": (0.0, 0.5),
}


def genome_fields() -> tuple[str, ...]:
    return tuple(field.name for field in fields(Genome))
