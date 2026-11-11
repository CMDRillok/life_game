"""Симуляция: numba-ядро тика, геном и состояние семян."""

from .core import SeedState, Simulation
from .genome import Genome

__all__ = ["Genome", "SeedState", "Simulation"]
