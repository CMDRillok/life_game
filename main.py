"""Точка входа life_game.

Интерактивный режим — поле с панорамой, зумом, ресайзом и тремя слоями.
Headless — только создание/загрузка поля (симуляция появится позже).
"""

from __future__ import annotations

import argparse

import pygame

from game import config
from game.camera import Camera
from game.demo import seed_demo
from game.field import Field
from game.renderer import Renderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="life_game: поле и визуализация")
    parser.add_argument("--width", type=int, default=config.WIDTH, help="ширина поля")
    parser.add_argument("--height", type=int, default=config.HEIGHT, help="высота поля")
    parser.add_argument(
        "--demo",
        dest="demo",
        action="store_true",
        default=True,
        help="заполнить демо-данными (по умолчанию)",
    )
    parser.add_argument(
        "--empty",
        dest="demo",
        action="store_false",
        help="пустое поле",
    )
    parser.add_argument("--load", type=str, default=None, help="загрузить чекпоинт .npz")
    parser.add_argument("--save", type=str, default=None, help="сохранить поле при выходе")
    parser.add_argument("--headless", action="store_true", help="без окна")
    parser.add_argument("--fps", type=int, default=config.FPS, help="лимит FPS")
    return parser


def create_field(args: argparse.Namespace) -> Field:
    if args.load:
        field = Field.load(args.load)
        print(f"загружено: {field}")
    else:
        field = Field(args.width, args.height)
        if args.demo:
            seed_demo(field)
            print("демо-сцена заполнена")
    return field


def run_headless(field: Field, args: argparse.Namespace) -> None:
    print(field)
    print(f"память слоёв: {field.memory_bytes() / (1024 * 1024):.1f} MiB")
    print("headless-цикл симуляции появится на этапе numba-ядра")
    if args.save:
        target = field.save(args.save)
        print(f"сохранено: {target}")


def run_interactive(field: Field, args: argparse.Namespace) -> None:
    pygame.init()
    pygame.display.set_caption("life_game — поле")
    screen = pygame.display.set_mode(
        (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.RESIZABLE
    )
    clock = pygame.time.Clock()
    renderer = Renderer()
    camera = Camera()
    camera.fit(field, *screen.get_size())

    view_mode = config.VIEW_CELLS
    view_index = 0
    show_grid = True
    dragging = False
    running = True

    while running:
        dt = clock.tick(args.fps) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(
                    (max(320, event.w), max(240, event.h)), pygame.RESIZABLE
                )

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    view_mode, view_index = config.VIEW_CELLS, 0
                elif event.key == pygame.K_2:
                    view_mode, view_index = config.VIEW_ORGANIC, 1
                elif event.key == pygame.K_3:
                    view_mode, view_index = config.VIEW_ENERGY, 2
                elif event.key == pygame.K_TAB:
                    view_index = (view_index + 1) % len(config.VIEW_ORDER)
                    view_mode = config.VIEW_ORDER[view_index]
                elif event.key == pygame.K_g:
                    show_grid = not show_grid
                elif event.key == pygame.K_r:
                    camera.reset(field, *screen.get_size())
                elif event.key == pygame.K_f:
                    camera.fit(field, *screen.get_size())
                elif event.key == pygame.K_RIGHTBRACKET:
                    field.resize(field.width + config.RESIZE_STEP, field.height)
                elif event.key == pygame.K_LEFTBRACKET:
                    field.resize(max(16, field.width - config.RESIZE_STEP), field.height)
                elif event.key == pygame.K_EQUALS:
                    field.resize(field.width, field.height + config.RESIZE_STEP)
                elif event.key == pygame.K_MINUS:
                    field.resize(field.width, max(16, field.height - config.RESIZE_STEP))

            elif event.type == pygame.MOUSEWHEEL:
                factor = config.ZOOM_STEP ** event.y
                camera.zoom_at(*mouse_pos, factor)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 2, 3):
                    dragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 2, 3):
                    dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                camera.pan_pixels(event.rel[0], event.rel[1])

        keys = pygame.key.get_pressed()
        dx = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(
            keys[pygame.K_a] or keys[pygame.K_LEFT]
        )
        dy = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(
            keys[pygame.K_w] or keys[pygame.K_UP]
        )
        if dx or dy:
            camera.move_pixels(
                dx * config.KEY_PAN_SPEED * dt,
                dy * config.KEY_PAN_SPEED * dt,
            )

        renderer.draw(
            screen,
            field,
            camera,
            view_mode,
            show_grid,
            mouse_pos,
            clock.get_fps(),
        )
        pygame.display.flip()

    if args.save:
        target = field.save(args.save)
        print(f"сохранено: {target}")

    pygame.quit()


def main() -> None:
    args = build_parser().parse_args()
    field = create_field(args)
    if args.headless:
        run_headless(field, args)
    else:
        run_interactive(field, args)


if __name__ == "__main__":
    main()
