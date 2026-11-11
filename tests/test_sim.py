"""Тесты numba-ядра симуляции (stdlib unittest)."""

from __future__ import annotations

import unittest

from game import config
from game.field import Field
from game.sim import Simulation


def make_field(width: int = 9, height: int = 9) -> Field:
    return Field(width, height)


def put_stem(field: Field, x: int, y: int, energy: float = 0.0, org: int = 1) -> None:
    field.cell_type[y, x] = config.STEM
    field.organism_id[y, x] = org
    field.capacity[y, x] = config.STEM_CAPACITY
    field.energy[y, x] = energy


def put_seed(field: Field, x: int, y: int, energy: float = 0.0, org: int = 1) -> None:
    field.cell_type[y, x] = config.SEED
    field.organism_id[y, x] = org
    field.energy[y, x] = energy


def put_worker(
    field: Field, x: int, y: int, code: int, energy: float = 0.0, org: int = 1
) -> None:
    field.cell_type[y, x] = code
    field.organism_id[y, x] = org
    field.energy[y, x] = energy


class BfsTests(unittest.TestCase):
    def test_bfs_wraps_and_marks_distances(self) -> None:
        field = make_field(10, 10)
        put_seed(field, 0, 0)
        for x in (9, 8, 7):
            put_stem(field, x, 0)

        sim = Simulation(field)
        count = sim.rebuild_bfs()

        self.assertEqual(count, 4)
        self.assertEqual(int(field.dist_seed[0, 0]), 0)
        self.assertEqual(int(field.dist_seed[0, 9]), 1)
        self.assertEqual(int(field.dist_seed[0, 8]), 2)
        self.assertEqual(int(field.dist_seed[0, 7]), 3)

    def test_bfs_ignores_other_organisms(self) -> None:
        field = make_field(10, 10)
        put_seed(field, 0, 0)
        put_stem(field, 1, 0, org=1)
        put_stem(field, 2, 0, org=2)
        put_stem(field, 2, 1, org=2)

        sim = Simulation(field)
        sim.rebuild_bfs()

        self.assertEqual(int(field.dist_seed[0, 1]), 1)
        self.assertEqual(int(field.dist_seed[0, 2]), -1)
        self.assertEqual(int(field.dist_seed[1, 2]), -1)


class FlowTests(unittest.TestCase):
    def test_energy_moves_one_cell_per_tick(self) -> None:
        field = make_field(10, 3)
        put_seed(field, 0, 0)
        put_stem(field, 1, 0)
        put_stem(field, 2, 0, energy=10.0)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        # Дальний стебель отдал энергию ближнему, но до семени она не дошла.
        self.assertAlmostEqual(float(field.energy[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(field.energy[0, 1]), 10.0, places=5)
        self.assertAlmostEqual(float(field.energy[0, 2]), 0.0, places=5)

        sim.tick()
        self.assertAlmostEqual(float(field.energy[0, 0]), 10.0, places=5)
        self.assertAlmostEqual(float(field.energy[0, 1]), 0.0, places=5)

    def test_flow_respects_seed_capacity(self) -> None:
        field = make_field(6, 3)
        put_seed(field, 0, 0, energy=995.0)
        put_stem(field, 1, 0, energy=50.0)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertAlmostEqual(float(field.energy[0, 0]), config.SEED_CAPACITY, places=5)
        self.assertAlmostEqual(float(field.energy[0, 1]), 45.0, places=5)

    def test_detached_stem_is_pruned(self) -> None:
        field = make_field(12, 12)
        put_seed(field, 0, 0)
        put_stem(field, 1, 0)
        put_stem(field, 8, 8, energy=50.0)  # оторван от семени

        sim = Simulation(field)
        sim.tick()

        # Оторванный от семени стебель никуда не ведёт и удаляется.
        self.assertEqual(int(field.cell_type[8, 8]), config.EMPTY)
        self.assertAlmostEqual(
            float(field.organic[8, 8]), config.DEATH_ORGANIC, places=5
        )
        self.assertAlmostEqual(float(field.soil_energy.sum()), 50.0, places=4)


class InjectionTests(unittest.TestCase):
    def test_worker_injects_only_into_stem(self) -> None:
        field = make_field(6, 6)
        put_seed(field, 2, 2)
        put_stem(field, 2, 3)
        put_stem(field, 2, 4)
        put_worker(field, 1, 4, config.GREEN, energy=10.0)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        # Соседние стебли не считаются соседями и не снижают производство.
        gain = config.GREEN_RATE
        self.assertAlmostEqual(float(field.energy[4, 1]), 0.0, places=5)
        self.assertAlmostEqual(float(field.energy[4, 2]), 0.0, places=5)
        self.assertAlmostEqual(float(field.energy[3, 2]), 10.0 + gain, places=5)

    def test_worker_cannot_inject_directly_into_seed(self) -> None:
        field = make_field(6, 6)
        put_seed(field, 2, 2)
        put_worker(field, 2, 3, config.GREEN)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        # Семя — не стебель, поэтому считается соседом и даёт 50%.
        gain = config.GREEN_RATE * config.GREEN_NEIGHBOR_EFFICIENCY[1]
        self.assertAlmostEqual(float(field.energy[2, 2]), 0.0, places=5)
        self.assertAlmostEqual(float(field.energy[3, 2]), gain, places=5)


class ProductionTests(unittest.TestCase):
    def test_green_produces_fixed_energy(self) -> None:
        field = make_field(5, 5)
        put_worker(field, 2, 2, config.GREEN)

        sim = Simulation(field)
        sim.tick()

        self.assertEqual(int(field.cell_type[2, 2]), config.GREEN)
        self.assertAlmostEqual(
            float(field.energy[2, 2]), config.GREEN_RATE, places=5
        )

    def test_orange_consumes_organic(self) -> None:
        field = make_field(7, 7)
        field.organic[2, 2] = 100.0
        put_worker(field, 3, 3, config.ORANGE)

        sim = Simulation(field)
        sim.tick()

        self.assertEqual(int(field.cell_type[3, 3]), config.ORANGE)
        self.assertAlmostEqual(
            float(field.energy[3, 3]), config.ORANGE_MAX_INTAKE, places=5
        )
        self.assertAlmostEqual(float(field.organic[2, 2]), 98.0, places=4)

    def test_blue_consumes_soil_energy(self) -> None:
        field = make_field(7, 7)
        field.soil_energy[4, 4] = 100.0
        put_worker(field, 3, 3, config.BLUE)

        sim = Simulation(field)
        sim.tick()

        self.assertEqual(int(field.cell_type[3, 3]), config.BLUE)
        self.assertAlmostEqual(
            float(field.energy[3, 3]), config.BLUE_MAX_INTAKE, places=5
        )
        self.assertAlmostEqual(float(field.soil_energy[4, 4]), 98.0, places=4)


class DeathTests(unittest.TestCase):
    def test_orange_dies_without_organic_and_energy(self) -> None:
        field = make_field(7, 7)
        put_worker(field, 3, 3, config.ORANGE)

        sim = Simulation(field)
        died = sim.tick()

        self.assertEqual(died, 1)
        self.assertEqual(int(field.cell_type[3, 3]), config.EMPTY)
        self.assertAlmostEqual(
            float(field.organic[3, 3]), config.DEATH_ORGANIC, places=5
        )

    def test_blue_dies_without_soil_energy_and_energy(self) -> None:
        field = make_field(7, 7)
        put_worker(field, 3, 3, config.BLUE)

        sim = Simulation(field)
        died = sim.tick()

        self.assertEqual(died, 1)
        self.assertEqual(int(field.cell_type[3, 3]), config.EMPTY)
        self.assertAlmostEqual(
            float(field.organic[3, 3]), config.DEATH_ORGANIC, places=5
        )

    def test_orange_with_stored_energy_survives(self) -> None:
        field = make_field(7, 7)
        put_worker(field, 3, 3, config.ORANGE, energy=5.0)

        sim = Simulation(field)
        died = sim.tick()

        self.assertEqual(died, 0)
        self.assertEqual(int(field.cell_type[3, 3]), config.ORANGE)

    def test_death_dirties_topology(self) -> None:
        field = make_field(7, 7)
        put_worker(field, 3, 3, config.ORANGE)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertTrue(sim.dirty)

    def test_worker_list_pruned_after_death(self) -> None:
        field = make_field(7, 7)
        put_worker(field, 3, 3, config.ORANGE)

        sim = Simulation(field)
        self.assertEqual(sim.tick(), 1)
        # Умерший работник удаляется из списка сразу после смерти.
        self.assertEqual(sim.worker_count, 0)

        self.assertEqual(sim.tick(), 0)
        self.assertEqual(sim.worker_count, 0)


if __name__ == "__main__":
    unittest.main()
