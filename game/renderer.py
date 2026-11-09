"""LOD-рендер поля и HUD.

Поле рисуется через 8-битные палитровые поверхности:
- источник = один кадр на всё поле (клетки или шкала слоя), кэшируется по версии;
- на экран попадает только видимый регион, который масштабируется средствами pygame.

Координаты клеток берутся по модулю размеров поля, поэтому тор бесшовный.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from . import config
from .camera import Camera
from .field import Field

CELL_PALETTE_LIST = [
    tuple(int(channel) for channel in config.CELL_COLORS.get(code, (0, 0, 0)))
    for code in range(256)
]


def make_lut(stops: tuple[tuple[float, tuple[int, int, int]], ...]) -> np.ndarray:
    """Построить палитру из 256 цветов по опорным точкам."""
    positions = np.array([p for p, _ in stops], dtype=np.float64)
    colors = np.array([c for _, c in stops], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.interp(t, positions, colors[:, channel]).astype(np.uint8)
    return lut


def lut_to_palette(lut: np.ndarray) -> list[tuple[int, int, int]]:
    return [tuple(int(v) for v in row) for row in lut]


class Renderer:
    def __init__(self) -> None:
        self.organic_lut = make_lut(config.ORGANIC_STOPS)
        self.energy_lut = make_lut(config.ENERGY_STOPS)
        self.cell_energy_lut = make_lut(config.CELL_ENERGY_STOPS)
        self.organic_palette = lut_to_palette(self.organic_lut)
        self.energy_palette = lut_to_palette(self.energy_lut)
        self.cell_energy_palette = lut_to_palette(self.cell_energy_lut)

        self._source_key: tuple | None = None
        self._source_indices: np.ndarray | None = None
        self._source_surface: pygame.Surface | None = None
        self._source_palette: list[tuple[int, int, int]] = CELL_PALETTE_LIST

        self._scaled_key: tuple | None = None
        self._scaled_surface: pygame.Surface | None = None

        self._vmax_key: tuple | None = None
        self._vmax: float = 1.0

        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._gradients: dict[int, pygame.Surface] = {}

    # ------------------------------------------------------------------ источники

    def _value_scale(self, field: Field, view_mode: int) -> float:
        key = (view_mode, field.version)
        if key != self._vmax_key:
            if view_mode == config.VIEW_ORGANIC:
                source = field.organic
            elif view_mode == config.VIEW_ENERGY:
                source = field.soil_energy
            else:
                source = field.energy
            vmax = float(source.max()) if source.size else 0.0
            self._vmax = vmax if vmax > 1e-9 else 1.0
            self._vmax_key = key
        return self._vmax

    def build_source(self, field: Field, view_mode: int) -> tuple[np.ndarray, pygame.Surface]:
        """Собрать 8-битный индексный кадр всего поля и его палитру."""
        if view_mode == config.VIEW_CELLS:
            indices = field.cell_type
            palette = CELL_PALETTE_LIST
        elif view_mode in (
            config.VIEW_ORGANIC,
            config.VIEW_ENERGY,
            config.VIEW_CELL_ENERGY,
        ):
            if view_mode == config.VIEW_ORGANIC:
                values = field.organic
                palette = self.organic_palette
            elif view_mode == config.VIEW_ENERGY:
                values = field.soil_energy
                palette = self.energy_palette
            else:
                values = field.energy
                palette = self.cell_energy_palette
            vmax = self._value_scale(field, view_mode)
            indices = np.clip(values * (255.0 / vmax), 0, 255).astype(np.uint8)
        else:  # pragma: no cover
            raise ValueError(f"неизвестный режим просмотра: {view_mode}")

        surface = pygame.surfarray.make_surface(indices.T)
        surface.set_palette(palette)
        return indices, surface

    def source(self, field: Field, view_mode: int) -> pygame.Surface:
        key = (view_mode, field.version, field.width, field.height)
        if key != self._source_key:
            indices, surface = self.build_source(field, view_mode)
            self._source_indices = indices
            self._source_surface = surface
            self._source_palette = surface.get_palette()
            self._source_key = key
            self._scaled_key = None
        assert self._source_surface is not None
        return self._source_surface

    # ------------------------------------------------------------------- отрисовка

    def render_field(
        self,
        target: pygame.Surface,
        field: Field,
        camera: Camera,
        view_mode: int,
    ) -> None:
        source = self.source(field, view_mode)
        view_w, view_h = target.get_size()
        zoom = camera.zoom
        tile = camera.tile
        tile_size = tile * zoom

        x0 = math.floor(camera.offset_x / tile)
        y0 = math.floor(camera.offset_y / tile)
        x1 = math.ceil((camera.offset_x + view_w / zoom) / tile)
        y1 = math.ceil((camera.offset_y + view_h / zoom) / tile)
        region_w = max(1, x1 - x0)
        region_h = max(1, y1 - y0)

        if region_w > field.width or region_h > field.height:
            self._render_tiled(target, field, camera, source, view_w, view_h)
            return

        if 0 <= x0 and 0 <= y0 and x1 <= field.width and y1 <= field.height:
            region = source.subsurface((x0, y0, region_w, region_h))
        else:
            assert self._source_indices is not None
            xs = np.arange(x0, x1) % field.width
            ys = np.arange(y0, y1) % field.height
            wrapped = self._source_indices[np.ix_(ys, xs)]
            region = pygame.surfarray.make_surface(wrapped.T)
            region.set_palette(self._source_palette)

        dest_w = max(1, int(round(region_w * tile_size)))
        dest_h = max(1, int(round(region_h * tile_size)))
        if region.get_size() != (dest_w, dest_h):
            scaled = pygame.transform.scale(region, (dest_w, dest_h))
            scaled.set_palette(self._source_palette)
        else:
            scaled = region

        dest_x = (x0 * tile - camera.offset_x) * zoom
        dest_y = (y0 * tile - camera.offset_y) * zoom
        target.blit(scaled, (int(round(dest_x)), int(round(dest_y))))

    def _render_tiled(
        self,
        target: pygame.Surface,
        field: Field,
        camera: Camera,
        source: pygame.Surface,
        view_w: int,
        view_h: int,
    ) -> None:
        """Уменьшенный масштаб: рисуем поле повторяющимися тайлами."""
        tile_size = camera.tile * camera.zoom
        world_w = field.width * tile_size
        world_h = field.height * tile_size
        if world_w <= 0 or world_h <= 0:
            return

        scaled_w = max(1, int(round(world_w)))
        scaled_h = max(1, int(round(world_h)))

        # Совсем мелкое поле: тайлинг дал бы десятки тысяч blit-ов.
        # В этом случае дешевле один раз сэмплировать экран.
        if scaled_w < 32 or scaled_h < 32:
            self._render_direct(target, field, camera, view_w, view_h)
            return

        key = (self._source_key, scaled_w, scaled_h)
        if key != self._scaled_key:
            self._scaled_surface = pygame.transform.scale(source, (scaled_w, scaled_h))
            self._scaled_surface.set_palette(self._source_palette)
            self._scaled_key = key
        assert self._scaled_surface is not None

        origin_x = (-camera.offset_x) * camera.zoom
        origin_y = (-camera.offset_y) * camera.zoom
        start_x = origin_x % world_w - world_w
        start_y = origin_y % world_h - world_h

        y = start_y
        while y < view_h:
            x = start_x
            while x < view_w:
                target.blit(self._scaled_surface, (int(round(x)), int(round(y))))
                x += world_w
            y += world_h

    def _render_direct(
        self,
        target: pygame.Surface,
        field: Field,
        camera: Camera,
        view_w: int,
        view_h: int,
    ) -> None:
        """Один экранный кадр с сэмплированием индексов (для экстремального отдаления)."""
        assert self._source_indices is not None
        zoom = camera.zoom
        tile = camera.tile
        xs = np.floor(
            (np.arange(view_w, dtype=np.float64) / zoom + camera.offset_x) / tile
        ).astype(np.int64)
        ys = np.floor(
            (np.arange(view_h, dtype=np.float64) / zoom + camera.offset_y) / tile
        ).astype(np.int64)
        xs %= field.width
        ys %= field.height
        sampled = self._source_indices[np.ix_(ys, xs)]
        frame = pygame.surfarray.make_surface(sampled.T)
        frame.set_palette(self._source_palette)
        target.blit(frame, (0, 0))

    # ----------------------------------------------------------------------- HUD

    def _ensure_fonts(self) -> None:
        if self._font is not None:
            return
        if not pygame.font.get_init():
            pygame.font.init()
        path = pygame.font.match_font("dejavusans,liberationsans,notosans,freesans")
        self._font = pygame.font.Font(path, 18) if path else pygame.font.Font(None, 18)
        self._small_font = pygame.font.Font(path, 15) if path else pygame.font.Font(None, 15)

    def draw(
        self,
        screen: pygame.Surface,
        field: Field,
        camera: Camera,
        view_mode: int,
        show_grid: bool,
        mouse_pos: tuple[int, int],
        fps: float,
        sim_stats: dict | None = None,
    ) -> None:
        self._ensure_fonts()
        self.render_field(screen, field, camera, view_mode)
        if show_grid:
            self._draw_grid(screen, camera)
        self._draw_hud(screen, field, camera, view_mode, mouse_pos, fps, sim_stats)

    @staticmethod
    def _draw_grid(screen: pygame.Surface, camera: Camera) -> None:
        if camera.tile_size < 8.0:
            return
        width, height = screen.get_size()
        tile = camera.tile
        zoom = camera.zoom

        first = math.ceil(camera.offset_x / tile)
        x = first
        while True:
            sx = (x * tile - camera.offset_x) * zoom
            if sx > width:
                break
            if sx >= 0:
                pygame.draw.line(screen, config.GRID_COLOR, (int(sx), 0), (int(sx), height))
            x += 1

        first = math.ceil(camera.offset_y / tile)
        y = first
        while True:
            sy = (y * tile - camera.offset_y) * zoom
            if sy > height:
                break
            if sy >= 0:
                pygame.draw.line(screen, config.GRID_COLOR, (0, int(sy)), (width, int(sy)))
            y += 1

    def _draw_hud(
        self,
        screen: pygame.Surface,
        field: Field,
        camera: Camera,
        view_mode: int,
        mouse_pos: tuple[int, int],
        fps: float,
        sim_stats: dict | None = None,
    ) -> None:
        assert self._font is not None and self._small_font is not None

        mx, my = mouse_pos
        cx, cy = camera.screen_to_cell(mx, my, field)
        code = int(field.cell_type[cy, cx])
        if view_mode == config.VIEW_CELLS:
            hover = f"({cx}, {cy})  {config.CELL_NAMES[code]}"
        elif view_mode == config.VIEW_ORGANIC:
            hover = f"({cx}, {cy})  органика: {float(field.organic[cy, cx]):.2f}"
        elif view_mode == config.VIEW_ENERGY:
            hover = f"({cx}, {cy})  энергия почвы: {float(field.soil_energy[cy, cx]):.2f}"
        else:
            hover = (
                f"({cx}, {cy})  энергия клетки: {float(field.energy[cy, cx]):.2f}  "
                f"({config.CELL_NAMES[code]})"
            )

        zoom_text = (
            f"{camera.zoom:.3f}x" if camera.zoom < 0.1 else f"{camera.zoom:.2f}x"
        )
        lines = [
            (f"Слой: {config.VIEW_NAMES[view_mode]}   [1/2/3/4, Tab]", config.HUD_FG),
            (
                f"Поле: {field.width}x{field.height}   зум: {zoom_text}   "
                f"тайл: {camera.tile_size:.2f}px",
                config.HUD_MUTED,
            ),
            (f"FPS: {fps:.0f}   клеток: {field.cells:,}", config.HUD_MUTED),
            (hover, config.HUD_FG),
        ]

        if sim_stats is not None:
            counts = sim_stats.get("counts", {})
            paused = "да" if sim_stats.get("paused") else "нет"
            lines.extend(
                [
                    (
                        f"Тик: {sim_stats.get('tick', 0)}   "
                        f"сим: {sim_stats.get('sim_hz', 0.0):.1f}/с   пауза: {paused}",
                        config.HUD_FG,
                    ),
                    (
                        f"Энергия семени: {sim_stats.get('seed_energy', 0.0):.1f}   "
                        f"всего в клетках: {sim_stats.get('total_energy', 0.0):.1f}",
                        config.HUD_MUTED,
                    ),
                    (
                        f"зел: {counts.get(config.GREEN, 0)}   "
                        f"оранж: {counts.get(config.ORANGE, 0)}   "
                        f"син: {counts.get(config.BLUE, 0)}   "
                        f"стебель: {counts.get(config.STEM, 0)}   "
                        f"семя: {counts.get(config.SEED, 0)}",
                        config.HUD_MUTED,
                    ),
                ]
            )

        lines.append(
            (
                "ЛКМ — панорама, колесо — зум, G — сетка, R — сброс, F — вписать",
                config.HUD_MUTED,
            )
        )
        lines.append(
            (
                "Space — пауза, N — один тик, PageUp/PageDown — скорость",
                config.HUD_MUTED,
            )
        )

        padding = 10
        line_h = 21
        panel_w = 660
        panel_h = padding * 2 + line_h * len(lines)
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(config.HUD_BG)
        screen.blit(panel, (10, 10))

        y = 10 + padding
        for text, color in lines:
            label = self._font.render(text, True, color)
            screen.blit(label, (10 + padding, y))
            y += line_h

        self._draw_legend(screen, field, view_mode)

    def _draw_legend(self, screen: pygame.Surface, field: Field, view_mode: int) -> None:
        assert self._small_font is not None
        width, height = screen.get_size()
        x0 = 10
        y0 = height - 44
        panel_w = 500
        panel = pygame.Surface((panel_w, 34), pygame.SRCALPHA)
        panel.fill(config.HUD_BG)
        screen.blit(panel, (x0, y0))

        if view_mode == config.VIEW_CELLS:
            x = x0 + 10
            for code in (config.GREEN, config.ORANGE, config.BLUE, config.STEM, config.SEED):
                pygame.draw.rect(screen, config.CELL_COLORS[code], (x, y0 + 10, 14, 14))
                label = self._small_font.render(config.CELL_NAMES[code], True, config.HUD_FG)
                screen.blit(label, (x + 19, y0 + 11))
                x += 19 + label.get_width() + 14
            return

        if view_mode == config.VIEW_ORGANIC:
            lut = self.organic_lut
            caption_text = "органика в почве"
        elif view_mode == config.VIEW_CELL_ENERGY:
            lut = self.cell_energy_lut
            caption_text = "энергия клеток"
        else:
            lut = self.energy_lut
            caption_text = "энергия в почве"

        if view_mode not in self._gradients:
            gradient = lut[np.linspace(0, 255, 220).astype(np.uint8)]
            bar = np.repeat(gradient[:, None, :], 12, axis=1)
            self._gradients[view_mode] = pygame.surfarray.make_surface(bar)

        screen.blit(self._gradients[view_mode], (x0 + 10, y0 + 11))
        vmax = self._value_scale(field, view_mode)
        left = self._small_font.render("0", True, config.HUD_MUTED)
        right = self._small_font.render(f"{vmax:.1f}", True, config.HUD_MUTED)
        screen.blit(left, (x0 + 244, y0 + 10))
        screen.blit(right, (x0 + 280, y0 + 10))
        caption = self._small_font.render(caption_text, True, config.HUD_FG)
        screen.blit(caption, (x0 + 340, y0 + 10))
