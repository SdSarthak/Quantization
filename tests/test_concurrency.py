"""Concurrent access to the model registry.

Every test here is bounded by an explicit timeout and driven by events rather
than sleeps, so a regression fails the suite instead of hanging it.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from .conftest import FakeModel, StubManager

TIMEOUT = 10.0


class BlockingManager(StubManager):
    """Load blocks on ``gate`` so a load can be held in flight deliberately."""

    def __init__(self, settings, block: str = "original"):
        super().__init__(settings)
        self.block = block
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.load_counts = {}

    def _load_variant(self, variant):
        self.load_counts[variant.name] = self.load_counts.get(variant.name, 0) + 1
        if variant.name == self.block:
            self.entered.set()
            if not self.gate.wait(TIMEOUT):
                raise AssertionError("gate was never released")
        return FakeModel()


def test_a_variant_is_loaded_once_under_concurrent_requests(settings):
    manager = BlockingManager(settings, block="original")
    manager.gate.set()  # do not block; just count

    with ThreadPoolExecutor(max_workers=8) as pool:
        models = [
            future.result(timeout=TIMEOUT)
            for future in [pool.submit(manager.ensure_loaded, "original") for _ in range(8)]
        ]

    assert manager.load_counts == {"original": 1}
    assert all(model is models[0] for model in models)
    manager.shutdown()


def test_loading_one_variant_does_not_block_a_resident_one(settings):
    """A slow download must not queue requests for an already loaded variant."""
    manager = BlockingManager(settings, block="dynamic_quant")
    resident = manager.ensure_loaded("original")

    slow = threading.Thread(target=manager.ensure_loaded, args=("dynamic_quant",))
    slow.start()
    assert manager.entered.wait(TIMEOUT), "the slow load never started"

    done = threading.Event()
    result = {}

    def fetch_resident():
        result["model"] = manager.ensure_loaded("original")
        done.set()

    threading.Thread(target=fetch_resident, daemon=True).start()
    assert done.wait(2.0), "a resident variant blocked behind an unrelated load"
    assert result["model"] is resident

    manager.gate.set()
    slow.join(TIMEOUT)
    assert not slow.is_alive()
    manager.shutdown()


def test_a_failed_load_does_not_poison_later_attempts(settings):
    class FlakyManager(StubManager):
        def __init__(self, s):
            super().__init__(s)
            self.attempts = 0

        def _load_variant(self, variant):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("transient hub failure")
            return FakeModel()

    manager = FlakyManager(settings)
    with pytest.raises(Exception):
        manager.ensure_loaded("original")
    assert manager.ensure_loaded("original") is not None
    assert manager.attempts == 2
    manager.shutdown()


def test_concurrent_generate_requests_all_succeed(client):
    payload = {"prompt": "hello", "max_length": 3, "temperature": 0.0}

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = [
            future.result(timeout=TIMEOUT)
            for future in [
                pool.submit(client.post, "/generate/original", json=payload)
                for _ in range(6)
            ]
        ]

    assert [r.status_code for r in responses] == [200] * 6
    assert {tuple(r.json()["generated_text"]) for r in responses} == {("10 11 12",)}
    # Six concurrent requests, one load.
    assert client.manager.loaded_variants == ["original"]
