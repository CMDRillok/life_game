"""Тесты этапа 4d: плотность зелёных, смерть, полёт семян и мутации."""

from __future__ import annotations

import unittest
from dataclasses import asdict
from unittest import mock

import numpy as np

from game import config
from game.demo import seed_start
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

    def test_one_neighbor_cell_halves_production(self) -> None:
        field = make_field()
        put_worker(field, 5, 5, config.GREEN, org=1)
        put_worker(field, 6, 5, config.GREEN, org=2)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        expected = config.GREEN_RATE * config.GREEN_NEIGHBOR_EFFICIENCY[1]
        self.assertAlmostEqual(float(field.energy[5, 5]), expected, places=5)
        self.assertAlmostEqual(float(field.energy[5, 6]), expected, places=5)

    def test_two_neighbor_cells_leave_ten_percent(self) -> None:
        field = make_field()
        put_worker(field, 5, 5, config.GREEN)
        put_worker(field, 5, 4, config.GREEN)
        put_worker(field, 5, 6, config.GREEN)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        expected = config.GREEN_RATE * config.GREEN_NEIGHBOR_EFFICIENCY[2]
        self.assertAlmostEqual(float(field.energy[5, 5]), expected, places=5)
        self.assertAlmostEqual(
            float(field.energy[5, 5]), config.GREEN_RATE * 0.1, places=5
        )

    def test_three_neighbor_cells_produce_nothing(self) -> None:
        field = make_field()
        put_worker(field, 5, 5, config.GREEN)
        for x, y in ((5, 4), (5, 6), (4, 5)):
            put_worker(field, x, y, config.GREEN)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertAlmostEqual(float(field.energy[5, 5]), 0.0, places=5)

    def test_stems_do_not_count_as_neighbors(self) -> None:
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
        self.assertAlmostEqual(
            float(field.energy[5, 5]), config.GREEN_RATE, places=5
        )

    def test_seed_and_worker_neighbors_count(self) -> None:
        field = make_field()
        put_worker(field, 5, 5, config.GREEN)
        put_seed(field, 6, 5)
        put_worker(field, 4, 5, config.ORANGE, energy=5.0)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        # Семя и оранжевая клетка — не стебли, поэтому это 2 соседа.
        expected = config.GREEN_RATE * config.GREEN_NEIGHBOR_EFFICIENCY[2]
        self.assertAlmostEqual(float(field.energy[5, 5]), expected, places=5)

    def test_mixed_neighborhood_ignores_only_stems(self) -> None:
        field = make_field()
        put_worker(field, 5, 5, config.GREEN)
        put_stem(field, 5, 4)
        put_seed(field, 6, 5)
        put_seed(field, 4, 5)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        expected = config.GREEN_RATE * config.GREEN_NEIGHBOR_EFFICIENCY[2]
        self.assertAlmostEqual(float(field.energy[5, 5]), expected, places=5)


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
    def test_isolated_stem_is_pruned_and_spills_energy(self) -> None:
        field = make_field()
        put_stem(field, 4, 4, energy=9.0)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        self.assertEqual(int(field.cell_type[4, 4]), config.EMPTY)
        self.assertAlmostEqual(
            float(field.organic[4, 4]), config.DEATH_ORGANIC, places=5
        )
        self.assertAlmostEqual(float(field.soil_energy.sum()), 9.0, places=4)

    def test_tail_without_workers_is_pruned_to_seed_buffer(self) -> None:
        field = make_field()
        put_seed(field, 0, 0)
        for x in (1, 2, 3):
            put_stem(field, x, 0)

        with mock.patch.object(config, "STEM_PRUNE_GRACE", 2):
            sim = Simulation(field, seed_actions=False)
            sim.tick()
            sim.tick()
            sim.tick()

        # Хвост без работников отрезается, у семени остаётся буферный стебель.
        self.assertEqual(int(field.cell_type[0, 1]), config.STEM)
        self.assertEqual(int(field.cell_type[0, 2]), config.EMPTY)
        self.assertEqual(int(field.cell_type[0, 3]), config.EMPTY)

    def test_side_branch_with_empty_end_is_pruned(self) -> None:
        field = make_field()
        put_seed(field, 5, 5)
        put_stem(field, 5, 6)  # A, примыкает к семени
        put_stem(field, 6, 6)  # M, главный путь
        put_worker(field, 7, 6, config.GREEN)  # W, конец главного пути
        put_stem(field, 4, 6)  # B, боковая ветвь
        put_stem(field, 3, 6)  # C, пустой конец боковой ветви

        with mock.patch.object(config, "STEM_PRUNE_GRACE", 2):
            sim = Simulation(field, seed_actions=False)
            sim.tick()
            sim.tick()
            sim.tick()

        # Боковая ветвь B-C никуда не ведёт и удаляется.
        self.assertEqual(int(field.cell_type[6, 4]), config.EMPTY)
        self.assertEqual(int(field.cell_type[6, 3]), config.EMPTY)
        # Путь к работнику сохраняется.
        self.assertEqual(int(field.cell_type[6, 5]), config.STEM)
        self.assertEqual(int(field.cell_type[6, 6]), config.STEM)

    def test_branch_becomes_useful_before_grace_expires(self) -> None:
        field = make_field()
        put_seed(field, 0, 0)
        put_stem(field, 1, 0)
        put_stem(field, 2, 0)

        with mock.patch.object(config, "STEM_PRUNE_GRACE", 4):
            sim = Simulation(field, seed_actions=False)
            sim.tick()
            # Пока ветвь была тупиком, ей назначили срок.
            self.assertGreater(int(field.die_at[0, 2]), 0)

            # Появился работник — ветвь стала полезной.
            put_worker(field, 3, 0, config.GREEN)
            sim.tick()
            sim.tick()
            sim.tick()

        self.assertEqual(int(field.cell_type[0, 2]), config.STEM)
        self.assertEqual(int(field.die_at[0, 2]), 0)

    def test_stems_of_seedless_organism_are_removed(self) -> None:
        field = make_field()
        put_stem(field, 4, 4)
        put_stem(field, 5, 4)
        put_worker(field, 4, 5, config.GREEN)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        # Семени нет — весь организм мёртв, стебли убираются сразу.
        self.assertEqual(int(field.cell_type[4, 4]), config.EMPTY)
        self.assertEqual(int(field.cell_type[4, 5]), config.EMPTY)

    def test_energy_in_dead_tail_does_not_save_it(self) -> None:
        field = make_field()
        put_seed(field, 0, 0)
        put_stem(field, 1, 0)
        put_stem(field, 2, 0, energy=10.0)

        with mock.patch.object(config, "STEM_PRUNE_GRACE", 2):
            sim = Simulation(field, seed_actions=False)
            sim.tick()
            sim.tick()
            sim.tick()

        # Энергия в тупике не делает ветвь полезной: её убирают.
        self.assertEqual(int(field.cell_type[0, 2]), config.EMPTY)
        self.assertEqual(int(field.cell_type[0, 1]), config.STEM)

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


class StartWorldTests(unittest.TestCase):
    def test_seed_start_scatters_seeds_with_energy(self) -> None:
        field = make_field(60, 40)
        placed = seed_start(field, seed=3, count=25, energy=77.0)

        self.assertEqual(placed, 25)
        seeds = np.argwhere(field.cell_type == config.SEED)
        self.assertEqual(len(seeds), 25)
        self.assertEqual(
            len({int(field.organism_id[y, x]) for y, x in seeds}), 25
        )
        for y, x in seeds:
            self.assertAlmostEqual(float(field.energy[y, x]), 77.0, places=5)

    def test_simulation_starts_paused_by_default(self) -> None:
        self.assertTrue(config.START_PAUSED)


class PaletteTests(unittest.TestCase):
    def test_organic_and_energy_maps_start_black(self) -> None:
        for stops in (config.ORGANIC_STOPS, config.ENERGY_STOPS):
            lut = make_lut(stops)
            self.assertEqual(tuple(int(v) for v in lut[0]), (0, 0, 0))
            self.assertGreater(int(lut[255].sum()), 0)


if __name__ == "__main__":
    unittest.main()
