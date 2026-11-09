"""Камера: панорама, зум и преобразования экран <-> мир <-> клетка."""

from __future__ import annotations

import math

from . import config
from .field import Field


class Camera:
    """offset хранится в мировых пикселях, zoom — масштаб."""

    def __init__(self, tile: int = config.TILE) -> None:
        self.tile = tile
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom = 1.0

    # ------------------------------------------------------------ преобразования

    @property
    def tile_size(self) -> float:
        return self.tile * self.zoom

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx / self.zoom + self.offset_x, sy / self.zoom + self.offset_y)

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        return ((wx - self.offset_x) * self.zoom, (wy - self.offset_y) * self.zoom)

    def screen_to_cell(self, sx: float, sy: float, field: Field) -> tuple[int, int]:
        wx, wy = self.screen_to_world(sx, sy)
        cx = math.floor(wx / self.tile) % field.width
        cy = math.floor(wy / self.tile) % field.height
        return cx, cy

    def cell_to_screen(self, cx: int, cy: int) -> tuple[float, float]:
        return self.world_to_screen(cx * self.tile, cy * self.tile)

    # ----------------------------------------------------------------- движение

    def pan_pixels(self, dx: float, dy: float) -> None:
        """Сдвинуть содержимое за курсором на (dx, dy) экранных пикселей."""
        self.offset_x -= dx / self.zoom
        self.offset_y -= dy / self.zoom

    def move_pixels(self, dx: float, dy: float) -> None:
        """Сдвинуть сам вид (камера идёт вправо/вниз при dx/dy > 0)."""
        self.offset_x += dx / self.zoom
        self.offset_y += dy / self.zoom

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        """Изменить масштаб, сохранив точку под курсором на месте."""
        wx, wy = self.screen_to_world(sx, sy)
        new_zoom = min(config.MAX_ZOOM, max(config.MIN_ZOOM, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom
        self.offset_x = wx - sx / self.zoom
        self.offset_y = wy - sy / self.zoom

    def center_on(self, wx: float, wy: float, view_w: float, view_h: float) -> None:
        self.offset_x = wx - (view_w / 2.0) / self.zoom
        self.offset_y = wy - (view_h / 2.0) / self.zoom

    # ------------------------------------------------------------------- режимы

    def reset(self, field: Field, view_w: float, view_h: float) -> None:
        """Сброс: zoom=1, центр поля в центре экрана."""
        self.zoom = 1.0
        self.center_on(
            field.width * self.tile / 2.0,
            field.height * self.tile / 2.0,
            view_w,
            view_h,
        )

    def fit(self, field: Field, view_w: float, view_h: float, margin: float = 0.95) -> None:
        """Вписать всё поле в экран."""
        world_w = field.width * self.tile
        world_h = field.height * self.tile
        if world_w <= 0 or world_h <= 0:
            return
        self.zoom = min(view_w / world_w, view_h / world_h) * margin
        self.zoom = min(config.MAX_ZOOM, max(config.MIN_ZOOM, self.zoom))
        self.center_on(world_w / 2.0, world_h / 2.0, view_w, view_h)

    # ------------------------------------------------------------------ состояние

    def state(self) -> tuple[float, float, float, int]:
        return (self.offset_x, self.offset_y, self.zoom, self.tile)
