"""Тесты нескольких организмов, нескольких семян, отстрела и генома."""

from __future__ import annotations

import unittest

import numpy as np

from game import config
from game.field import Field
from game.sim import Genome, Simulation
from game.sim.genome import GENOME_BOUNDS


def make_field(width: int = 24, height: int = 24) -> Field:
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


class MultiSeedBfsTests(unittest.TestCase):
    def test_bfs_starts_from_every_seed(self) -> None:
        field = make_field()
        put_seed(field, 2, 2, org=1)
        put_seed(field, 15, 15, org=1)
        put_stem(field, 3, 2, org=1)
        put_stem(field, 16, 15, org=1)

        sim = Simulation(field)
        count = sim.rebuild_bfs()

        self.assertEqual(count, 4)
        self.assertEqual(int(field.dist_seed[2, 2]), 0)
        self.assertEqual(int(field.dist_seed[15, 15]), 0)
        self.assertEqual(int(field.dist_seed[2, 3]), 1)
        self.assertEqual(int(field.dist_seed[15, 16]), 1)

    def test_bfs_keeps_organisms_apart(self) -> None:
        field = make_field()
        put_seed(field, 2, 2, org=1)
        put_seed(field, 12, 2, org=2)
        put_stem(field, 3, 2, org=1)
        put_stem(field, 13, 2, org=2)
        # Стебель organism 2 вплотную к стеблю organism 1.
        put_stem(field, 4, 2, org=2)

        sim = Simulation(field)
        sim.rebuild_bfs()

        self.assertEqual(int(field.dist_seed[2, 3]), 1)
        self.assertEqual(int(field.dist_seed[2, 4]), -1)
        self.assertEqual(int(field.dist_seed[2, 13]), 1)


class OrganismIsolationTests(unittest.TestCase):
    def test_worker_cannot_inject_into_foreign_stem(self) -> None:
        field = make_field(10, 10)
        put_worker(field, 2, 2, config.GREEN, org=1)
        put_stem(field, 1, 2, org=1)
        field.capacity[2, 1] = 0.0
        put_seed(field, 0, 2, org=1)
        put_seed(field, 4, 2, org=2)
        put_stem(field, 3, 2, org=2)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        # Чужой стебель достижим для своего семени (dist >= 0), но работник
        # другого организма не имеет права в него вливать энергию.
        self.assertGreaterEqual(int(field.dist_seed[2, 3]), 0)
        gain = config.GREEN_RATE
        self.assertAlmostEqual(float(field.energy[2, 2]), gain, places=5)
        self.assertAlmostEqual(float(field.energy[2, 3]), 0.0, places=5)

    def test_worker_injects_into_own_stem(self) -> None:
        field = make_field(10, 10)
        put_seed(field, 4, 2, org=1)
        put_worker(field, 2, 2, config.GREEN, org=1)
        put_stem(field, 3, 2, org=1)

        sim = Simulation(field, seed_actions=False)
        sim.tick()

        # Энергия прошла работник -> стебель -> семя за один тик.
        gain = config.GREEN_RATE
        self.assertAlmostEqual(float(field.energy[2, 2]), 0.0, places=5)
        self.assertAlmostEqual(float(field.energy[2, 3]), 0.0, places=5)
        self.assertAlmostEqual(float(field.energy[2, 4]), gain, places=5)

    def test_seed_builds_only_on_own_stems(self) -> None:
        field = make_field(12, 12)
        put_seed(field, 1, 1, energy=200.0, org=1)
        put_stem(field, 2, 1, org=1)
        put_seed(field, 9, 9, energy=200.0, org=2)
        put_stem(field, 8, 9, org=2)

        sim = Simulation(field)
        self.assertTrue(sim.seed_create_auto(config.GREEN))

        workers = np.argwhere(field.cell_type == config.GREEN)
        self.assertEqual(len(workers), 1)
        y, x = (int(v) for v in workers[0])
        self.assertEqual(int(field.organism_id[y, x]), 1)
        # Новая клетка появилась у стебля organism 1, а не organism 2.
        self.assertLessEqual(abs(x - 2) + abs(y - 1), 1)


class SeedActionTests(unittest.TestCase):
    def test_two_seeds_of_one_organism_act_in_same_tick(self) -> None:
        field = make_field()
        put_seed(field, 3, 3, energy=50.0, org=1)
        put_seed(field, 18, 18, energy=50.0, org=1)

        sim = Simulation(field)
        acted = sim.seed_step_all()

        self.assertEqual(acted, 2)
        self.assertEqual(len(sim.seeds), 2)
        self.assertEqual(len({s.org_id for s in sim.seeds}), 1)
        self.assertEqual(int((field.cell_type == config.STEM).sum()), 2)
        self.assertEqual(int((field.cell_type == config.SEED).sum()), 2)

    def test_internal_duplication_keeps_organism(self) -> None:
        field = make_field()
        put_seed(field, 5, 5, energy=300.0, org=1)
        put_stem(field, 5, 6, org=1)

        sim = Simulation(field, rng_seed=7)
        genome = sim.seeds[0].genome
        genome.shoot_threshold = 10_000.0
        genome.duplicate_threshold = 10.0
        genome.duplicate_energy = 40.0

        action = sim.seed_step()

        self.assertEqual(action, config.SEED_ACTION_DUPLICATE)
        self.assertEqual(len(sim.seeds), 2)
        self.assertEqual(int((field.cell_type == config.SEED).sum()), 2)
        self.assertEqual(
            len({s.org_id for s in sim.seeds}),
            1,
        )
        child = sim.seeds[1]
        self.assertEqual(child.generation, 1)
        self.assertEqual(int(field.organism_id[child.y, child.x]), 1)
        self.assertAlmostEqual(float(field.energy[child.y, child.x]), 40.0, places=5)
        self.assertAlmostEqual(
            float(field.energy[5, 5]),
            300.0 - 40.0 - config.DUPLICATE_EXTRA_COST,
            places=5,
        )

    def test_shoot_creates_new_organism(self) -> None:
        field = make_field(30, 30)
        put_seed(field, 10, 10, energy=500.0, org=1)
        # Потомок появляется рядом со стеблем родителя, а не в радиусе от семени.
        put_stem(field, 10, 11, org=1)

        sim = Simulation(field, rng_seed=11)
        genome = sim.seeds[0].genome
        genome.shoot_threshold = 10.0
        genome.child_energy = 50.0
        genome.shoot_range = 6

        action = sim.seed_step()

        self.assertEqual(action, config.SEED_ACTION_SHOOT)
        self.assertEqual(len(sim.seeds), 2)
        child = sim.seeds[1]
        self.assertNotEqual(child.org_id, 1)
        self.assertEqual(child.org_id, 2)
        self.assertEqual(child.generation, 1)
        self.assertEqual(int(field.organism_id[child.y, child.x]), 2)
        self.assertAlmostEqual(float(field.energy[child.y, child.x]), 50.0, places=5)
        self.assertAlmostEqual(
            float(field.energy[10, 10]),
            500.0 - 50.0 - config.SHOOT_EXTRA_COST,
            places=5,
        )
        # Новая клетка 4-соседняя со стеблем и сразу летит от него прочь.
        self.assertEqual(abs(child.x - 10) + abs(child.y - 11), 1)
        self.assertFalse(child.established)
        self.assertEqual((child.travel_dy, child.travel_dx), (1, 0))
        self.assertEqual(child.travel_steps, 0)
        self.assertEqual(child.travel_limit, 6)

        # У нового организма пока только семя; родитель остался на месте.
        self.assertEqual(int(field.cell_type[10, 10]), config.SEED)
        self.assertEqual(int(field.organism_id[10, 10]), 1)

    def test_seed_cap_limits_duplication(self) -> None:
        field = make_field()
        put_seed(field, 5, 5, energy=10_000.0, org=1)
        for x, y in ((4, 5), (6, 5), (5, 4), (5, 6)):
            put_stem(field, x, y, org=1)

        sim = Simulation(field, rng_seed=3)
        genome = sim.seeds[0].genome
        genome.shoot_threshold = 100_000.0
        genome.duplicate_threshold = 1.0
        genome.duplicate_energy = 10.0
        genome.stems_per_worker = 0.0
        genome.stem_slack = 0.0

        sim.rebuild_bfs()
        sim._refresh_org_counts()

        # Добиваем количество семян до лимита вручную, чтобы не зависеть
        # от свободных клеток вокруг стеблей.
        for _ in range(config.MAX_SEEDS_PER_ORGANISM - 1):
            self.assertTrue(sim._seed_duplicate(sim.seeds[0]))
        self.assertEqual(len(sim.seeds), config.MAX_SEEDS_PER_ORGANISM)
        self.assertFalse(sim._seed_duplicate(sim.seeds[0]))


class GenomeTests(unittest.TestCase):
    def test_mutation_stays_within_bounds(self) -> None:
        genome = Genome()
        genome.mutation_rate = 1.0
        genome.mutation_scale = 0.5
        rng = np.random.default_rng(5)

        for _ in range(50):
            genome = genome.mutated(rng)
            for name, (low, high) in GENOME_BOUNDS.items():
                value = float(getattr(genome, name))
                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

    def test_clone_is_independent(self) -> None:
        genome = Genome()
        clone = genome.clone()
        clone.shoot_threshold = 123.0
        self.assertNotEqual(genome.shoot_threshold, clone.shoot_threshold)


class StatsTests(unittest.TestCase):
    def test_stats_report_seeds_organisms_and_generation(self) -> None:
        field = make_field(30, 30)
        put_seed(field, 5, 5, energy=400.0, org=1)
        put_stem(field, 5, 6, org=1)
        put_seed(field, 20, 20, energy=50.0, org=2)

        sim = Simulation(field, rng_seed=2)
        genome = sim.seeds[0].genome
        genome.shoot_threshold = 10.0
        genome.child_energy = 30.0
        genome.shoot_range = 4
        sim.seed_step_all()

        stats = sim.stats()
        self.assertEqual(stats["seed_count"], 3)
        self.assertEqual(stats["organism_count"], 3)
        self.assertEqual(stats["generation_max"], 1)
        self.assertGreater(stats["seed_energy"], 0.0)
        self.assertEqual(stats["counts"][config.SEED], 3)


if __name__ == "__main__":
    unittest.main()
