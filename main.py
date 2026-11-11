"""Точка входа life_game.

Интерактивный режим — поле с панорамой, зумом, ресайзом, слоями и симуляцией.
Headless — прогон заданного числа тиков без окна и сохранение чекпоинта.
"""

from __future__ import annotations

import argparse
import time

import pygame

from game import config
from game.camera import Camera
from game.demo import seed_demo
from game.field import Field
from game.renderer import Renderer
from game.sim import Simulation


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
    parser.add_argument(
        "--organisms",
        type=int,
        default=1,
        help="сколько организмов поставить в демо-сцене",
    )
    parser.add_argument(
        "--seeds-per-org",
        type=int,
        default=1,
        help="сколько семян ставить в каждом организме",
    )
    parser.add_argument("--load", type=str, default=None, help="загрузить чекпоинт .npz")
    parser.add_argument("--save", type=str, default=None, help="сохранить поле при выходе")
    parser.add_argument("--headless", action="store_true", help="без окна")
    parser.add_argument("--fps", type=int, default=config.FPS, help="лимит FPS")
    parser.add_argument(
        "--sim-hz",
        type=float,
        default=config.SIM_HZ,
        help="целевая скорость симуляции, тиков/с",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=0,
        help="сколько тиков прогнать в headless-режиме (0 — не запускать)",
    )
    return parser


def create_field(args: argparse.Namespace) -> Field:
    if args.load:
        field = Field.load(args.load)
        print(f"загружено: {field}")
    else:
        field = Field(args.width, args.height)
        if args.demo:
            seed_demo(
                field,
                organisms=max(1, args.organisms),
                seeds_per_org=max(1, args.seeds_per_org),
            )
            print(
                "демо-сцена заполнена: "
                f"организмов {max(1, args.organisms)}, "
                f"семян на организм {max(1, args.seeds_per_org)}"
            )
    return field


def _print_stats(sim: Simulation) -> None:
    stats = sim.stats()
    counts = stats["counts"]
    print(
        f"тик {stats['tick']}: "
        f"семян {stats['seed_count']} "
        f"(энергия {stats['seed_energy']:.1f}), "
        f"организмов {stats['organism_count']}, "
        f"поколение max {stats['generation_max']} "
        f"avg {stats['generation_avg']:.1f}, "
        f"всего в клетках {stats['total_energy']:.1f}, "
        f"зел {counts[config.GREEN]}, "
        f"оранж {counts[config.ORANGE]}, "
        f"син {counts[config.BLUE]}, "
        f"стебель {counts[config.STEM]}, "
        f"семя {counts[config.SEED]}"
    )


def run_headless(field: Field, args: argparse.Namespace) -> None:
    print(field)
    print(f"память слоёв: {field.memory_bytes() / (1024 * 1024):.1f} MiB")

    sim = Simulation(field)
    ticks = max(0, args.ticks)
    if ticks:
        # Первый тик прогревает numba и не входит в замер.
        sim.tick()
        remaining = ticks - 1
        started = time.perf_counter()
        if remaining > 0:
            for _ in range(remaining):
                sim.tick()
        elapsed = time.perf_counter() - started
        if remaining > 0:
            rate = remaining / elapsed
            print(
                f"прогнано {ticks} тиков: "
                f"{elapsed * 1000.0 / remaining:.2f} мс/тик, "
                f"{rate:.1f} тиков/с"
            )
        else:
            print("прогнан 1 тик (прогрев numba)")
    else:
        print("симуляция не запускалась (--ticks 0)")

    _print_stats(sim)

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

    sim = Simulation(field)
    print("прогрев numba (первый тик)...")
    sim.tick()

    sim_hz = max(0.1, float(args.sim_hz))
    accumulator = 0.0
    paused = False
    single_tick = False

    # Измеренная скорость и кэш статистики для HUD.
    measured_ticks = 0
    measured_time = 0.0
    measured_hz = 0.0
    stats_timer = 0.0
    sim_stats = sim.stats()
    sim_stats["sim_hz"] = 0.0
    sim_stats["paused"] = paused

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

            # VIDEORESIZE намеренно не обрабатываем: pygame-ce сам обновляет
            # display surface. Вызов set_mode здесь создаёт цикл
            # resize -> set_mode -> resize (особенно в тайловых WM вроде Hyprland).

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    view_mode, view_index = config.VIEW_CELLS, 0
                elif event.key == pygame.K_2:
                    view_mode, view_index = config.VIEW_ORGANIC, 1
                elif event.key == pygame.K_3:
                    view_mode, view_index = config.VIEW_ENERGY, 2
                elif event.key == pygame.K_4:
                    view_mode, view_index = config.VIEW_CELL_ENERGY, 3
                elif event.key == pygame.K_TAB:
                    view_index = (view_index + 1) % len(config.VIEW_ORDER)
                    view_mode = config.VIEW_ORDER[view_index]
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    accumulator = 0.0
                elif event.key == pygame.K_n:
                    single_tick = True
                elif event.key == pygame.K_PAGEUP:
                    sim_hz = min(1000.0, sim_hz * 1.25)
                elif event.key == pygame.K_PAGEDOWN:
                    sim_hz = max(0.1, sim_hz / 1.25)
                elif event.key == pygame.K_g:
                    show_grid = not show_grid
                elif event.key == pygame.K_r:
                    camera.reset(field, *screen.get_size())
                elif event.key == pygame.K_f:
                    camera.fit(field, *screen.get_size())
                elif event.key == pygame.K_RIGHTBRACKET:
                    field.resize(field.width + config.RESIZE_STEP, field.height)
                    sim.on_field_changed()
                elif event.key == pygame.K_LEFTBRACKET:
                    field.resize(max(16, field.width - config.RESIZE_STEP), field.height)
                    sim.on_field_changed()
                elif event.key == pygame.K_EQUALS:
                    field.resize(field.width, field.height + config.RESIZE_STEP)
                    sim.on_field_changed()
                elif event.key == pygame.K_MINUS:
                    field.resize(field.width, max(16, field.height - config.RESIZE_STEP))
                    sim.on_field_changed()

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

        ticks_done = 0
        if single_tick:
            sim.tick()
            ticks_done += 1
            single_tick = False
        elif not paused:
            accumulator += dt
            step = 1.0 / sim_hz
            while accumulator >= step and ticks_done < config.MAX_TICKS_PER_FRAME:
                sim.tick()
                accumulator -= step
                ticks_done += 1
            if ticks_done >= config.MAX_TICKS_PER_FRAME:
                # Не копим долг, иначе после лага симуляция будет догонять рывком.
                accumulator = min(accumulator, step)

        measured_ticks += ticks_done
        measured_time += dt
        stats_timer += dt
        if stats_timer >= config.STATS_INTERVAL:
            if measured_time > 0.0:
                measured_hz = measured_ticks / measured_time
            measured_ticks = 0
            measured_time = 0.0
            stats_timer = 0.0
            sim_stats = sim.stats()
            sim_stats["sim_hz"] = measured_hz
            sim_stats["paused"] = paused

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
            sim_stats,
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
