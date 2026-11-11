"""Тесты этапа 4d: плотность зелёных, смерть, полёт семян и мутации."""

from __future__ import annotations

import unittest
from dataclasses import asdict
from unittest import mock

import numpy as np

from game import config
from game.field import Field
from game.renderer import make_lut
from game.sim import Genome, Simulation
from game.sim.genome import GENOME_BOUNDS


def make_field(width: int = 12, height: int = 12) -> Field:
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


def put_worker(
    field: Field, x: int, y: int, code: int, energy: float = 0.0, org: int = 1
) -> None:
    field.cell_type[y, x] = code
    field.organism_id[y, x] = org
    field.energy[y, x] = energy
    field.touch()


class GreenCrowdTests(unittest.TestCase):
    def test_green_alone_produces_full_rate(self) -> None:
        field = make_field()
        put_worker(field, 5, 5, config.GREEN)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertAlmostEqual(
            float(field.energy[5, 5]), config.GREEN_RATE, places=5
        )

    def test_green_loses_eighth_per_neighbor(self) -> None:
        field = make_field()
        put_worker(field, 5, 5, config.GREEN)
        for x, y in ((5, 4), (5, 6), (4, 5), (6, 5)):
            put_stem(field, x, y)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        expected = config.GREEN_RATE * (
            1.0 - 4 * config.GREEN_CROWD_PENALTY
        )
        self.assertAlmostEqual(float(field.energy[5, 5]), expected, places=5)

    def test_green_surrounded_by_eight_neighbors_produces_nothing(self) -> None:
        field = make_field()
        put_worker(field, 5, 5, config.GREEN)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                put_stem(field, 5 + dx, 5 + dy)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertEqual(int(field.cell_type[5, 5]), config.GREEN)
        self.assertAlmostEqual(float(field.energy[5, 5]), 0.0, places=5)


class SeedUpkeepTests(unittest.TestCase):
    def test_seed_pays_upkeep_each_tick(self) -> None:
        field = make_field()
        put_seed(field, 3, 3, energy=5.0)

        sim = Simulation(field)
        sim.apply_seed_upkeep()

        self.assertAlmostEqual(float(field.energy[3, 3]), 4.0, places=5)
        self.assertEqual(len(sim.seeds), 1)

    def test_seed_dies_and_spills_energy_when_upkeep_unpaid(self) -> None:
        field = make_field()
        put_seed(field, 3, 3, energy=2.0)

        sim = Simulation(field)
        self.assertEqual(sim.apply_seed_upkeep(), 0)
        self.assertAlmostEqual(float(field.energy[3, 3]), 1.0, places=5)

        self.assertEqual(sim.apply_seed_upkeep(), 1)
        self.assertEqual(int(field.cell_type[3, 3]), config.EMPTY)
        self.assertEqual(len(sim.seeds), 0)
        self.assertAlmostEqual(
            float(field.organic[3, 3]), config.DEATH_ORGANIC, places=5
        )
        # 1.0 энергии рассеялась по 3x3.
        self.assertAlmostEqual(float(field.soil_energy.sum()), 1.0, places=4)

    def test_upkeep_is_not_charged_when_seed_actions_disabled(self) -> None:
        field = make_field()
        put_seed(field, 3, 3, energy=2.0)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertAlmostEqual(float(field.energy[3, 3]), 2.0, places=5)
        self.assertEqual(len(sim.seeds), 1)


class LifetimeTests(unittest.TestCase):
    def test_stem_dies_on_scheduled_tick_and_spills_energy(self) -> None:
        field = make_field()
        put_stem(field, 4, 4, energy=9.0)
        field.die_at[4, 4] = 1
        field.touch()

        sim = Simulation(field, seed_actions=False)
        sim.tick()
        self.assertEqual(int(field.cell_type[4, 4]), config.STEM)

        sim.tick()
        self.assertEqual(int(field.cell_type[4, 4]), config.EMPTY)
        self.assertAlmostEqual(
            float(field.organic[4, 4]), config.DEATH_ORGANIC, places=5
        )
        self.assertAlmostEqual(float(field.soil_energy.sum()), 9.0, places=4)

    def test_worker_lifetime_is_fixed_at_creation(self) -> None:
        field = make_field()
        put_seed(field, 5, 5, energy=100.0)
        put_stem(field, 5, 6)

        sim = Simulation(field)
        self.assertTrue(sim.seed_create_auto(config.GREEN))

        workers = np.argwhere(field.cell_type == config.GREEN)
        self.assertEqual(len(workers), 1)
        y, x = (int(v) for v in workers[0])
        self.assertEqual(
            int(field.die_at[y, x]),
            sim.tick_count
            + int(config.WORKER_CREATE_COST * config.LIFETIME_TICKS_PER_ENERGY),
        )


class TravelTests(unittest.TestCase):
    def _travelling_seed(
        self, field: Field, x: int, y: int, energy: float, limit: int = 10
    ) -> Simulation:
        put_seed(field, x, y, energy=energy)
        sim = Simulation(field)
        state = sim.seeds[0]
        state.established = False
        state.travel_dy = 1
        state.travel_dx = 0
        state.travel_steps = 0
        state.travel_limit = limit
        sim._rebuild_seed_index()
        return sim

    def test_travel_moves_without_leaving_a_stem(self) -> None:
        field = make_field()
        sim = self._travelling_seed(field, 4, 4, energy=50.0)

        action = sim.seed_step()

        self.assertEqual(action, config.SEED_ACTION_TRAVEL)
        self.assertEqual(int(field.cell_type[4, 4]), config.EMPTY)
        self.assertEqual(int(field.cell_type[5, 4]), config.SEED)
        self.assertAlmostEqual(
            float(field.energy[5, 4]),
            50.0 - config.SEED_TRAVEL_COST,
            places=5,
        )
        self.assertEqual(len(sim.seeds), 1)
        self.assertFalse(sim.seeds[0].established)
        self.assertEqual(sim.seeds[0].travel_steps, 1)

    def test_travel_establishes_when_energy_is_low(self) -> None:
        field = make_field()
        sim = self._travelling_seed(
            field, 4, 4, energy=config.SEED_ESTABLISH_ENERGY
        )

        action = sim.seed_step()

        self.assertEqual(action, config.SEED_ACTION_ESTABLISH)
        self.assertTrue(sim.seeds[0].established)
        self.assertEqual(int(field.cell_type[4, 4]), config.SEED)

    def test_collision_destroys_target_and_seed(self) -> None:
        field = make_field()
        put_stem(field, 4, 5, energy=7.0)
        sim = self._travelling_seed(field, 4, 4, energy=50.0)

        action = sim.seed_step()

        self.assertEqual(action, config.SEED_ACTION_COLLIDE)
        self.assertEqual(len(sim.seeds), 0)
        self.assertEqual(int(field.cell_type[4, 4]), config.EMPTY)
        self.assertEqual(int(field.cell_type[5, 4]), config.EMPTY)
        self.assertAlmostEqual(
            float(field.organic[4, 4]), config.DEATH_ORGANIC, places=5
        )
        self.assertAlmostEqual(
            float(field.organic[5, 4]), config.DEATH_ORGANIC, places=5
        )
        self.assertAlmostEqual(float(field.soil_energy.sum()), 57.0, places=4)


class MutationTests(unittest.TestCase):
    class _FakeRng:
        def __init__(self, index: int, delta: float) -> None:
            self.index = index
            self.delta = delta

        def integers(self, high: int) -> int:
            return self.index

        def normal(self, loc: float, scale: float) -> float:
            return self.delta

    def test_mutate_one_changes_exactly_one_gene(self) -> None:
        names = tuple(GENOME_BOUNDS)
        for index, name in enumerate(names):
            with self.subTest(gene=name):
                genome = Genome()
                before = asdict(genome)
                changed = genome.mutate_one(self._FakeRng(index, 1.0))
                after = asdict(genome)

                self.assertEqual(changed, name)
                diffs = [key for key in before if before[key] != after[key]]
                self.assertEqual(diffs, [name])
                low, high = GENOME_BOUNDS[name]
                self.assertGreaterEqual(float(after[name]), low)
                self.assertLessEqual(float(after[name]), high)

    def test_tick_mutation_mutates_seed_at_most_once(self) -> None:
        field = make_field()
        put_seed(field, 4, 4, energy=100.0)
        put_stem(field, 4, 5)

        sim = Simulation(field, rng_seed=4)
        state = sim.seeds[0]
        before = asdict(state.genome)

        with mock.patch.object(config, "TICK_MUTATION_CHANCE", 1.0):
            sim._mutate_seeds()
            after_first = asdict(state.genome)
            sim._mutate_seeds()
            after_second = asdict(state.genome)

        diffs = [key for key in before if before[key] != after_first[key]]
        self.assertEqual(len(diffs), 1)
        self.assertEqual(after_first, after_second)
        self.assertEqual(state.last_mutation_tick, sim.tick_count)


class PaletteTests(unittest.TestCase):
    def test_organic_and_energy_maps_start_black(self) -> None:
        for stops in (config.ORGANIC_STOPS, config.ENERGY_STOPS):
            lut = make_lut(stops)
            self.assertEqual(tuple(int(v) for v in lut[0]), (0, 0, 0))
            self.assertGreater(int(lut[255].sum()), 0)


if __name__ == "__main__":
    unittest.main()
