"""Тесты действий семени: создание клеток, движение, запрет соседства."""

from __future__ import annotations

import unittest

import numpy as np

from game import config
from game.field import Field
from game.sim import Simulation


def make_field(width: int = 9, height: int = 9) -> Field:
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


class SeedCreateTests(unittest.TestCase):
    def test_seed_cannot_create_stem_directly(self) -> None:
        field = make_field(9, 9)
        put_seed(field, 4, 4, energy=80.0)
        put_stem(field, 4, 5)

        sim = Simulation(field)
        with self.assertRaises(ValueError):
            sim.seed_create_auto(config.STEM)

        # Стебель появляется только при ходе семени.
        self.assertEqual(int(field.cell_type[4, 4]), config.SEED)
        self.assertAlmostEqual(float(field.energy[4, 4]), 80.0, places=5)
        self.assertEqual(
            int((field.cell_type != config.EMPTY).sum()),
            2,
        )

    def test_worker_creation_can_be_far_from_seed(self) -> None:
        field = make_field(9, 9)
        put_seed(field, 0, 0, energy=200.0)
        for x in (1, 2, 3):
            put_stem(field, x, 0)

        sim = Simulation(field)
        self.assertTrue(sim.seed_create_auto(config.GREEN))

        # Дальний стебель (3, 0), направление "вверх" с зацикливанием.
        self.assertEqual(int(field.cell_type[8, 3]), config.GREEN)
        # Новая клетка не обязана соседствовать с семенем.
        self.assertNotIn((8, 3), ((0, 1), (1, 0)))

    def test_create_worker_next_to_stem(self) -> None:
        field = make_field(9, 9)
        put_seed(field, 4, 4, energy=80.0)
        put_stem(field, 4, 5)

        sim = Simulation(field)
        self.assertTrue(sim.seed_create_auto(config.GREEN))

        self.assertEqual(int(field.cell_type[6, 4]), config.GREEN)
        self.assertEqual(int(field.organism_id[6, 4]), 1)
        self.assertAlmostEqual(
            float(field.capacity[6, 4]), config.WORKER_CAPACITY, places=5
        )
        self.assertAlmostEqual(
            float(field.energy[4, 4]), 80.0 - config.WORKER_CREATE_COST, places=5
        )
        self.assertEqual(sim.worker_count, 1)
        self.assertEqual(
            int(sim.worker_idx[0]), 6 * field.width + 4
        )

    def test_worker_creation_blocked_by_diagonal_neighbor(self) -> None:
        field = make_field(11, 11)
        put_seed(field, 5, 5, energy=80.0)
        put_stem(field, 5, 6)
        put_worker(field, 5, 7, config.GREEN)

        sim = Simulation(field)
        # Оба свободных соседа стебля (4,6) и (6,6) диагональны работнику.
        self.assertFalse(sim.seed_create_auto(config.GREEN))
        self.assertAlmostEqual(float(field.energy[5, 5]), 80.0, places=5)
        self.assertEqual(sim.worker_count, 0)

    def test_create_fails_without_energy(self) -> None:
        field = make_field(9, 9)
        put_seed(field, 4, 4, energy=config.WORKER_CREATE_COST - 0.1)
        put_stem(field, 4, 5)

        sim = Simulation(field)
        self.assertFalse(sim.seed_create_auto(config.GREEN))
        self.assertAlmostEqual(
            float(field.energy[4, 4]), config.WORKER_CREATE_COST - 0.1, places=5
        )
        self.assertEqual(int(field.cell_type[5, 4]), config.STEM)

    def test_move_leaves_new_stem_in_bfs(self) -> None:
        field = make_field(9, 9)
        put_seed(field, 4, 4, energy=80.0)
        put_stem(field, 4, 5)

        sim = Simulation(field)
        self.assertTrue(sim.seed_move(-1, 0))
        self.assertEqual(sim.rebuild_bfs(), 3)
        self.assertEqual(int(field.dist_seed[4, 3]), 0)
        self.assertEqual(int(field.dist_seed[4, 4]), 1)
        self.assertEqual(int(field.dist_seed[5, 4]), 2)


class SeedMoveTests(unittest.TestCase):
    def test_move_leaves_stem_and_carries_energy(self) -> None:
        field = make_field(7, 7)
        put_seed(field, 3, 3, energy=50.0)

        sim = Simulation(field)
        self.assertTrue(sim.seed_move(1, 0))

        self.assertEqual((sim.seed_x, sim.seed_y), (4, 3))
        self.assertEqual(int(field.cell_type[3, 3]), config.STEM)
        self.assertAlmostEqual(
            float(field.capacity[3, 3]), config.STEM_CAPACITY, places=5
        )
        self.assertAlmostEqual(float(field.energy[3, 3]), 0.0, places=5)
        self.assertEqual(int(field.cell_type[3, 4]), config.SEED)
        self.assertAlmostEqual(
            float(field.energy[3, 4]), 50.0 - config.SEED_MOVE_COST, places=5
        )
        self.assertEqual(int(field.organism_id[3, 3]), 1)
        self.assertEqual(int(field.organism_id[3, 4]), 1)
        self.assertTrue(sim.dirty)

        self.assertEqual(sim.rebuild_bfs(), 2)
        self.assertEqual(int(field.dist_seed[3, 3]), 1)

    def test_policy_builds_worker_when_stems_enough(self) -> None:
        field = make_field(11, 11)
        put_seed(field, 5, 5, energy=200.0)
        for x, y in ((5, 6), (6, 5), (5, 4)):
            put_stem(field, x, y)

        sim = Simulation(field)
        action = sim.seed_step()

        # Трёх стеблей при нуле работников достаточно: семя стоит на месте,
        # а не тянет новый стебель, и создаёт работника.
        self.assertEqual(action, config.GREEN)
        self.assertEqual((sim.seed_x, sim.seed_y), (5, 5))
        self.assertEqual(int(field.cell_type[5, 5]), config.SEED)
        self.assertEqual(int(field.cell_type[4, 6]), config.GREEN)

    def test_move_requires_energy_for_left_stem(self) -> None:
        field = make_field(7, 7)
        put_seed(field, 3, 3, energy=config.SEED_MOVE_COST - 0.1)

        sim = Simulation(field)
        self.assertFalse(sim.seed_move(1, 0))
        self.assertEqual((sim.seed_x, sim.seed_y), (3, 3))
        self.assertEqual(int(field.cell_type[3, 3]), config.SEED)
        self.assertAlmostEqual(
            float(field.energy[3, 3]), config.SEED_MOVE_COST - 0.1, places=5
        )

    def test_move_into_occupied_cell_fails(self) -> None:
        field = make_field(7, 7)
        put_seed(field, 3, 3, energy=50.0)
        put_stem(field, 4, 3)

        sim = Simulation(field)
        self.assertFalse(sim.seed_move(1, 0))
        self.assertEqual((sim.seed_x, sim.seed_y), (3, 3))
        self.assertAlmostEqual(float(field.energy[3, 3]), 50.0, places=5)

    def test_lone_seed_bootstraps_by_moving(self) -> None:
        field = make_field(7, 7)
        put_seed(field, 3, 3, energy=50.0)

        sim = Simulation(field)
        action = sim.seed_step()

        self.assertEqual(action, config.SEED_ACTION_MOVE)
        self.assertEqual(sim.last_seed_action, config.SEED_ACTION_MOVE)
        self.assertEqual(int(field.cell_type[3, 3]), config.STEM)
        self.assertEqual(int(field.cell_type[2, 3]), config.SEED)
        self.assertEqual((sim.seed_x, sim.seed_y), (3, 2))
        self.assertAlmostEqual(
            float(field.energy[2, 3]), 50.0 - config.SEED_MOVE_COST, places=5
        )

    def test_one_cell_action_per_tick(self) -> None:
        field = make_field(11, 11)
        put_seed(field, 5, 5, energy=200.0)
        put_stem(field, 5, 6)

        sim = Simulation(field)
        filled_before = int((field.cell_type != config.EMPTY).sum())
        sim.tick()
        filled_after_one = int((field.cell_type != config.EMPTY).sum())
        self.assertEqual(filled_after_one, filled_before + 1)
        self.assertEqual(sim.last_seed_action, config.SEED_ACTION_MOVE)
        self.assertEqual(int(field.cell_type[5, 5]), config.STEM)

        sim.tick()
        filled_after_two = int((field.cell_type != config.EMPTY).sum())
        self.assertEqual(filled_after_two, filled_after_one + 1)
        self.assertEqual(sim.last_seed_action, config.SEED_ACTION_MOVE)


class GrowthTests(unittest.TestCase):
    def test_growth_keeps_workers_apart_and_connected(self) -> None:
        field = Field(48, 48)
        from game.demo import seed_demo

        seed_demo(field)
        sim = Simulation(field)
        stems_before = int((field.cell_type == config.STEM).sum())

        for _ in range(150):
            sim.tick()

        self.assertGreater(int((field.cell_type == config.STEM).sum()), stems_before)
        self.assertGreater(sim.worker_count, 0)
        self.assertGreater(sim.tick_count, 0)

        # Все стебли по-прежнему связаны с семенем.
        sim.rebuild_bfs()
        stems = np.argwhere(field.cell_type == config.STEM)
        self.assertGreater(len(stems), 0)
        for y, x in stems:
            self.assertGreaterEqual(int(field.dist_seed[y, x]), 1)

        # Работники не соседствуют друг с другом по 8 направлениям.
        workers = np.argwhere(
            (field.cell_type == config.GREEN)
            | (field.cell_type == config.ORANGE)
            | (field.cell_type == config.BLUE)
        )
        # Инкрементальный список не должен содержать мёртвых или дубликаты:
        # иначе производство/инъекция обработают клетку дважды.
        self.assertEqual(sim.worker_count, len(workers))
        self.assertEqual(
            set(int(i) for i in sim.worker_idx[: sim.worker_count]),
            {int(y) * field.width + int(x) for y, x in workers},
        )
        height, width = field.shape
        for y, x in workers:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbor = field.cell_type[(y + dy) % height, (x + dx) % width]
                    self.assertNotIn(
                        int(neighbor),
                        (config.GREEN, config.ORANGE, config.BLUE),
                    )

        self.assertFalse((field.energy < 0).any())


if __name__ == "__main__":
    unittest.main()
