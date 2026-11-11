"""Тесты токсичного избытка органики/энергии, переполнения и рендера."""

from __future__ import annotations

import unittest

from game import config
from game.field import Field
from game.renderer import Renderer
from game.sim import Simulation


def make_field(width: int = 14, height: int = 14) -> Field:
    return Field(width, height)


def put_seed(field: Field, x: int, y: int, energy: float = 0.0, org: int = 1) -> None:
    field.cell_type[y, x] = config.SEED
    field.organism_id[y, x] = org
    field.energy[y, x] = energy
    field.touch()


def put_stem(field: Field, x: int, y: int, energy: float = 0.0, org: int = 1) -> None:
    field.cell_type[y, x] = config.STEM
    field.organism_id[y, x] = org
    field.capacity[y, x] = config.STEM_CAPACITY
    field.energy[y, x] = energy
    field.touch()


def put_support_stem(field: Field, x: int, y: int, org: int = 1) -> None:
    put_stem(field, x, y, org=org)
    field.capacity[y, x] = 0.0


def put_worker(
    field: Field, x: int, y: int, code: int, energy: float = 0.0, org: int = 1
) -> None:
    field.cell_type[y, x] = code
    field.organism_id[y, x] = org
    field.energy[y, x] = energy
    field.touch()


class ToxicEnergyTests(unittest.TestCase):
    def test_energy_toxicity_kills_all_except_blue(self) -> None:
        field = make_field()

        # Синяя в токсичной энергии выживает.
        put_worker(field, 2, 2, config.BLUE, org=1)
        put_support_stem(field, 3, 2, org=1)
        put_seed(field, 4, 2, org=1)
        field.soil_energy[2, 2] = 900.0

        # Зелёная погибает.
        put_worker(field, 8, 2, config.GREEN, org=2)
        put_support_stem(field, 9, 2, org=2)
        put_seed(field, 10, 2, org=2)
        field.soil_energy[2, 8] = 900.0

        # Оранжевая тоже погибает от энергии.
        put_worker(field, 2, 8, config.ORANGE, org=3)
        put_support_stem(field, 3, 8, org=3)
        put_seed(field, 4, 8, org=3)
        field.soil_energy[8, 2] = 900.0

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertEqual(int(field.cell_type[2, 2]), config.BLUE)
        self.assertEqual(int(field.cell_type[2, 8]), config.EMPTY)
        self.assertEqual(int(field.cell_type[8, 2]), config.EMPTY)

    def test_stem_and_seed_die_in_toxic_energy(self) -> None:
        field = make_field(10, 10)
        put_seed(field, 5, 5)
        put_stem(field, 4, 5)
        field.soil_energy[5, 5] = 900.0
        field.soil_energy[5, 4] = 900.0

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertEqual(int(field.cell_type[5, 5]), config.EMPTY)
        self.assertEqual(int(field.cell_type[5, 4]), config.EMPTY)


class ToxicOrganicTests(unittest.TestCase):
    def test_organic_toxicity_kills_all_except_orange(self) -> None:
        field = make_field()

        # Оранжевая (жёлтая) в токсичной органике выживает.
        put_worker(field, 2, 2, config.ORANGE, org=1)
        put_support_stem(field, 3, 2, org=1)
        put_seed(field, 4, 2, org=1)
        field.organic[2, 2] = 900.0

        # Зелёная погибает.
        put_worker(field, 8, 2, config.GREEN, org=2)
        put_support_stem(field, 9, 2, org=2)
        put_seed(field, 10, 2, org=2)
        field.organic[2, 8] = 900.0

        # Синяя тоже погибает от органики.
        put_worker(field, 2, 8, config.BLUE, org=3)
        put_support_stem(field, 3, 8, org=3)
        put_seed(field, 4, 8, org=3)
        field.organic[8, 2] = 900.0

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertEqual(int(field.cell_type[2, 2]), config.ORANGE)
        self.assertEqual(int(field.cell_type[2, 8]), config.EMPTY)
        self.assertEqual(int(field.cell_type[8, 2]), config.EMPTY)

    def test_stem_and_seed_die_in_toxic_organic(self) -> None:
        field = make_field(10, 10)
        put_seed(field, 5, 5)
        put_stem(field, 4, 5)
        field.organic[5, 5] = 900.0
        field.organic[5, 4] = 900.0

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertEqual(int(field.cell_type[5, 5]), config.EMPTY)
        self.assertEqual(int(field.cell_type[5, 4]), config.EMPTY)


class NoStemTests(unittest.TestCase):
    def test_worker_without_stem_dies_immediately(self) -> None:
        field = make_field(9, 9)
        put_worker(field, 4, 4, config.GREEN)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertEqual(int(field.cell_type[4, 4]), config.EMPTY)
        self.assertAlmostEqual(
            float(field.organic.sum()), config.DEATH_ORGANIC, places=4
        )

    def test_foreign_stem_does_not_count_as_support(self) -> None:
        field = make_field(9, 9)
        put_worker(field, 4, 4, config.GREEN, org=1)
        put_stem(field, 5, 4, org=2)
        put_seed(field, 6, 4, org=2)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertEqual(int(field.cell_type[4, 4]), config.EMPTY)

    def test_worker_with_own_stem_survives(self) -> None:
        field = make_field(9, 9)
        put_worker(field, 4, 4, config.GREEN)
        put_support_stem(field, 5, 4)
        put_seed(field, 6, 4)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertEqual(int(field.cell_type[4, 4]), config.GREEN)


class OverflowTests(unittest.TestCase):
    def test_energy_overflow_spreads_to_neighbors(self) -> None:
        field = make_field(7, 7)
        field.soil_energy[3, 3] = 2500.0

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertAlmostEqual(
            float(field.soil_energy[3, 3]),
            config.RESOURCE_OVERFLOW_THRESHOLD,
            places=4,
        )
        self.assertAlmostEqual(float(field.soil_energy.sum()), 2500.0, places=3)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                self.assertAlmostEqual(
                    float(field.soil_energy[3 + dy, 3 + dx]),
                    (2500.0 - config.RESOURCE_OVERFLOW_THRESHOLD) / 8.0,
                    places=3,
                )

    def test_overflow_pending_cells_are_processed_on_later_ticks(self) -> None:
        field = make_field(7, 7)
        field.soil_energy[3, 3] = 2500.0
        field.soil_energy[3, 4] = 2500.0

        sim = Simulation(field, seed_actions=False)
        for _ in range(4):
            sim.tick()

        self.assertLessEqual(float(field.soil_energy.max()), 2010.0)
        self.assertAlmostEqual(float(field.soil_energy.sum()), 5000.0, places=2)

    def test_organic_overflow_spreads_to_neighbors(self) -> None:
        field = make_field(7, 7)
        field.organic[3, 3] = 2500.0

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertAlmostEqual(
            float(field.organic[3, 3]),
            config.RESOURCE_OVERFLOW_THRESHOLD,
            places=4,
        )
        self.assertAlmostEqual(float(field.organic.sum()), 2500.0, places=3)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                self.assertAlmostEqual(
                    float(field.organic[3 + dy, 3 + dx]),
                    (2500.0 - config.RESOURCE_OVERFLOW_THRESHOLD) / 8.0,
                    places=3,
                )


class ToxicRenderTests(unittest.TestCase):
    def test_energy_and_organic_maps_use_bright_red_above_threshold(self) -> None:
        field = make_field(6, 6)
        field.soil_energy[1, 1] = 900.0
        field.soil_energy[2, 2] = 100.0
        field.organic[4, 4] = 900.0
        field.organic[3, 3] = 100.0

        renderer = Renderer()
        energy_indices, energy_surface = renderer.build_source(
            field, config.VIEW_ENERGY
        )
        organic_indices, organic_surface = renderer.build_source(
            field, config.VIEW_ORGANIC
        )

        self.assertEqual(int(energy_indices[1, 1]), 255)
        self.assertEqual(int(organic_indices[4, 4]), 255)
        self.assertNotEqual(int(energy_indices[2, 2]), 255)
        self.assertNotEqual(int(organic_indices[3, 3]), 255)

        energy_palette = energy_surface.get_palette()
        organic_palette = organic_surface.get_palette()
        self.assertEqual(
            tuple(int(v) for v in energy_palette[255])[:3],
            config.TOXIC_COLOR,
        )
        self.assertEqual(
            tuple(int(v) for v in organic_palette[255])[:3],
            config.TOXIC_COLOR,
        )


if __name__ == "__main__":
    unittest.main()
