"""Бенчмарк поля: аллокация, проходы по памяти, сборка источника и кадра.

Запуск:
    .venv/bin/python benchmark.py
    .venv/bin/python benchmark.py --width 1024 --height 1024
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import pygame

from game import config
from game.camera import Camera
from game.demo import seed_demo
from game.field import Field
from game.renderer import Renderer
from game.sim import Simulation

SCREEN = (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)


def bench(label: str, fn: Callable[[], object], repeat: int = 10, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    for _ in range(repeat):
        fn()
    elapsed = (time.perf_counter() - started) / repeat
    print(f"{label:<44} {elapsed * 1000.0:9.2f} ms")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк поля life_game")
    parser.add_argument("--width", type=int, default=config.WIDTH)
    parser.add_argument("--height", type=int, default=config.HEIGHT)
    parser.add_argument("--passes", type=int, default=20, help="проходов на условный тик")
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    print(f"Поле: {args.width}x{args.height} = {args.width * args.height:,} клеток")
    print("-" * 64)

    alloc_time = bench(
        "аллокация Field",
        lambda: Field(args.width, args.height),
        repeat=3,
        warmup=0,
    )

    field = Field(args.width, args.height)
    seed_demo(field)
    print(f"память слоёв: {field.memory_bytes() / (1024 * 1024):.1f} MiB")
    print("-" * 64)

    arr = field.energy
    bytes_per_pass = arr.nbytes * 2  # чтение + запись

    full_pass = bench(
        "полный проход: multiply(out=)",
        lambda: np.multiply(arr, 0.999, out=arr),
        repeat=args.repeat,
    )
    print(
        f"  ~ {bytes_per_pass / full_pass / 1e9:.2f} GB/s, "
        f"оценка {1.0 / (full_pass * args.passes):.1f} тиков/с при {args.passes} проходах"
    )

    neighbor = bench(
        "соседи через np.roll (4 сдвига + сумма)",
        lambda: (
            np.roll(arr, 1, axis=0)
            + np.roll(arr, -1, axis=0)
            + np.roll(arr, 1, axis=1)
            + np.roll(arr, -1, axis=1)
        ),
        repeat=args.repeat,
    )
    print(f"  ~ {neighbor * 1000.0:.2f} ms/проход (roll аллоцирует копии)")
    print("-" * 64)

    print("numba: первый тик включает компиляцию ядер...")
    sim = Simulation(field)
    sim.tick()
    print(f"работников на поле: {sim.worker_count}")

    bfs_time = bench(
        "пересборка BFS (при изменении топологии)",
        sim.rebuild_bfs,
        repeat=args.repeat,
        warmup=1,
    )
    tick_time = bench(
        "тик симуляции (производство/переток/смерть)",
        sim.tick,
        repeat=args.repeat,
        warmup=1,
    )
    print(
        f"  BFS ~ {bfs_time * 1000.0:.2f} ms, "
        f"тик ~ {tick_time * 1e6:.1f} мкс (~{1.0 / tick_time:.0f} тиков/с)"
    )
    print("-" * 64)

    renderer = Renderer()
    target = pygame.Surface(SCREEN)
    cameras = {"zoom=1": Camera(), "fit": Camera()}
    cameras["zoom=1"].reset(field, *SCREEN)
    cameras["fit"].fit(field, *SCREEN)

    for cam_name, camera in cameras.items():
        print(f"камера {cam_name}: zoom={camera.zoom:.4f}, tile={camera.tile_size:.2f}px")
        for view_mode, view_name in config.VIEW_NAMES.items():
            source_time = bench(
                f"  источник {view_name}",
                lambda vm=view_mode: renderer.build_source(field, vm),
                repeat=args.repeat,
            )
            renderer.source(field, view_mode)
            frame_time = bench(
                f"  кадр {view_name} ({SCREEN[0]}x{SCREEN[1]})",
                lambda vm=view_mode, cam=camera: renderer.render_field(
                    target, field, cam, vm
                ),
                repeat=args.repeat,
            )
            print(
                f"    источник {source_time * 1000.0:.2f} ms, "
                f"кадр ~ {1.0 / frame_time:.0f} FPS"
            )

    print("-" * 64)
    print(f"аллокация: {alloc_time * 1000.0:.1f} ms")
    print("Готово.")


if __name__ == "__main__":
    main()
