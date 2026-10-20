#!/usr/bin/env python3
"""CS-48 clocks. One code path, two clocks.

``RealClock`` is wall-real-time (``time.monotonic_ns``) with real threads; it is what CS-49 uses against the live
cascade. ``VirtualClock`` is a cooperative deterministic scheduler: every task is a thread but exactly ONE runs at
a time; time advances only when every task is parked in ``sleep_until_ns``; ties break by park order. Same inputs
give a bit-identical event log, which is what lets the conformance suite assert exact (not statistical) results.
Anything that blocks must block through the clock (the harness code does; backends injected in virtual mode must too).
"""
from __future__ import annotations

import threading
import time
from typing import Callable

NS = 1_000_000_000


class RealClock:
    kind = "real"

    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []
        self.errors: list[BaseException] = []

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def sleep_until_ns(self, t_ns: int) -> None:
        # coarse sleep then a short spin: the player's frame deadlines are 20 ms apart, jitter is recorded not hidden
        while True:
            rem = t_ns - time.monotonic_ns()
            if rem <= 0:
                return
            if rem > 2_000_000:
                time.sleep((rem - 1_000_000) / NS)

    def spawn(self, name: str, fn: Callable[[], None]) -> None:
        def run() -> None:
            try:
                fn()
            except BaseException as e:  # noqa: BLE001 - surfaced by run_all
                self.errors.append(e)

        th = threading.Thread(target=run, name=name, daemon=True)
        self._threads.append(th)

    def run_all(self) -> None:
        for th in self._threads:
            th.start()
        for th in self._threads:
            th.join()
        if self.errors:
            raise self.errors[0]


class VirtualClock:
    kind = "virtual"

    def __init__(self, start_ns: int = 0) -> None:
        self._now = start_ns
        self._cv = threading.Condition()
        self._tasks: dict[str, dict] = {}
        self._order: list[str] = []
        self._seq = 0
        self._turn: str | None = None  # name of the task allowed to run
        self.errors: list[BaseException] = []
        self._by_thread: dict[int, str] = {}

    def now_ns(self) -> int:
        return self._now

    def spawn(self, name: str, fn: Callable[[], None]) -> None:
        if name in self._tasks:
            raise ValueError(f"duplicate task {name}")
        self._tasks[name] = {"fn": fn, "wake": self._now, "seq": self._next_seq(), "done": False, "thread": None}
        self._order.append(name)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def sleep_until_ns(self, t_ns: int) -> None:
        name = self._by_thread[threading.get_ident()]
        with self._cv:
            task = self._tasks[name]
            task["wake"] = max(int(t_ns), self._now)
            task["seq"] = self._next_seq()
            self._turn = None  # yield to the scheduler
            self._cv.notify_all()
            while self._turn != name:
                self._cv.wait()

    def _body(self, name: str) -> None:
        self._by_thread[threading.get_ident()] = name
        with self._cv:
            while self._turn != name:
                self._cv.wait()
        try:
            self._tasks[name]["fn"]()
        except BaseException as e:  # noqa: BLE001
            self.errors.append(e)
        finally:
            with self._cv:
                self._tasks[name]["done"] = True
                self._turn = None
                self._cv.notify_all()

    def run_all(self) -> None:
        for name in self._order:
            th = threading.Thread(target=self._body, args=(name,), name=f"vclk-{name}", daemon=True)
            self._tasks[name]["thread"] = th
            th.start()
        while True:
            with self._cv:
                live = [(t["wake"], t["seq"], n) for n, t in self._tasks.items() if not t["done"]]
                if not live or self.errors:
                    break
                wake, _, name = min(live)
                self._now = max(self._now, wake)
                self._turn = name
                self._cv.notify_all()
                while self._turn is not None:
                    self._cv.wait()
        for t in self._tasks.values():
            if t["done"]:
                t["thread"].join()
        if self.errors:
            raise self.errors[0]
