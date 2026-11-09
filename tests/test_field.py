"""Базовые тесты поля и камеры (stdlib unittest)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from game import config
from game.camera import Camera
from game.field import Field


class FieldTests(unittest.TestCase):
    def test_wrap(self) -> None:
        field = Field(10, 5)
        self.assertEqual(field.wrap(-1, -1), (9, 4))
        self.assertEqual(field.wrap(10, 5), (0, 0))
        self.assertEqual(field.wrap(13, -2), (3, 3))

    def test_neighbors4_wrap(self) -> None:
        field = Field(10, 10)
        neighbors = field.neighbors4(0, 0)
        self.assertEqual(len(neighbors), 4)
        self.assertIn((9, 0), neighbors)
        self.assertIn((1, 0), neighbors)
        self.assertIn((0, 9), neighbors)
        self.assertIn((0, 1), neighbors)

    def test_set_get_cell_wraps(self) -> None:
        field = Field(10, 10)
        field.set_cell(-1, -1, config.GREEN, energy=7.5)
        self.assertEqual(field.cell_type[9, 9], config.GREEN)
        self.assertAlmostEqual(float(field.energy[9, 9]), 7.5)
        self.assertEqual(field.get_cell(19, 19), config.GREEN)

    def test_resize_keeps_top_left(self) -> None:
        field = Field(10, 10)
        field.cell_type[0, 0] = config.SEED
        field.organic[5, 5] = 3.0
        field.resize(6, 20)
        self.assertEqual((field.width, field.height), (6, 20))
        self.assertEqual(field.cell_type[0, 0], config.SEED)
        self.assertAlmostEqual(float(field.organic[5, 5]), 3.0)
        # Обрезанная часть не должна протечь в новые данные.
        self.assertEqual(field.cell_type.shape, (20, 6))

    def test_save_load_roundtrip(self) -> None:
        field = Field(20, 12)
        field.cell_type[1, 2] = config.ORANGE
        field.energy[1, 2] = 42.0
        field.organic[3, 4] = 5.5
        field.touch()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npz"
            saved = field.save(path)
            loaded = Field.load(saved)
            self.assertEqual((loaded.width, loaded.height), (20, 12))
            self.assertEqual(loaded.cell_type[1, 2], config.ORANGE)
            self.assertAlmostEqual(float(loaded.energy[1, 2]), 42.0)
            self.assertAlmostEqual(float(loaded.organic[3, 4]), 5.5)
            self.assertGreater(len(np.unique(loaded.dist_seed)), 0)


class CameraTests(unittest.TestCase):
    def test_screen_to_cell(self) -> None:
        field = Field(100, 100)
        camera = Camera(tile=24)
        camera.offset_x = 0.0
        camera.offset_y = 0.0
        camera.zoom = 1.0
        self.assertEqual(camera.screen_to_cell(0, 0, field), (0, 0))
        self.assertEqual(camera.screen_to_cell(23, 23, field), (0, 0))
        self.assertEqual(camera.screen_to_cell(24, 24, field), (1, 1))
        # Отрицательные экранные координаты зацикливаются.
        self.assertEqual(camera.screen_to_cell(-1, -1, field), (99, 99))

    def test_zoom_keeps_point_under_cursor(self) -> None:
        camera = Camera(tile=24)
        camera.zoom = 1.0
        camera.offset_x = 120.0
        camera.offset_y = 240.0
        cursor = (400.0, 300.0)
        before = camera.screen_to_world(*cursor)
        camera.zoom_at(*cursor, 2.5)
        after = camera.screen_to_world(*cursor)
        self.assertAlmostEqual(before[0], after[0], places=6)
        self.assertAlmostEqual(before[1], after[1], places=6)

    def test_fit_contains_field(self) -> None:
        field = Field(2000, 2000)
        camera = Camera(tile=24)
        camera.fit(field, 1280, 720)
        world_w = field.width * camera.tile
        world_h = field.height * camera.tile
        visible_w = 1280 / camera.zoom
        visible_h = 720 / camera.zoom
        self.assertGreaterEqual(visible_w, world_w * 0.9)
        self.assertGreaterEqual(visible_h, world_h * 0.9)


if __name__ == "__main__":
    unittest.main()
