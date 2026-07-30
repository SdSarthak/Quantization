"""Shared fixtures.

The tests never download a checkpoint. Instead they subclass ``ModelManager``
and replace only the two methods that touch the Hugging Face hub, so the real
encode/generate/decode/benchmark code paths are still exercised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from airavata_quant.config import Settings
from airavata_quant.manager import ModelManager

PROMPT_IDS = [5, 6, 7]
PAD_ID = 0
EOS_ID = 2


class FakeTokenizer:
    """Minimal stand-in for a Hugging Face tokenizer."""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = PAD_ID
        self.eos_token_id = EOS_ID
        self.padding_side = "right"
        self.saved_to: List[str] = []

    def __call__(self, prompt: str, return_tensors=None, truncation=None, max_length=None):
        ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        values = [int(t) for t in tokens]
        if skip_special_tokens:
            values = [v for v in values if v not in (PAD_ID, EOS_ID)]
        return " ".join(str(v) for v in values)

    def save_pretrained(self, path: str) -> None:
        self.saved_to.append(path)


class FakeModel(torch.nn.Module):
    """Echoes the prompt then emits ``max_new_tokens`` deterministic tokens."""

    def __init__(self, pad_tail: int = 0) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.pad_tail = pad_tail
        self.generate_calls: List[Dict[str, Any]] = []
        self.saved_to: List[str] = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.generate_calls.append(kwargs)
        new_tokens = kwargs["max_new_tokens"]
        sequences = kwargs.get("num_return_sequences", 1)
        prompt = input_ids[0]

        rows = []
        for offset in range(sequences):
            body = torch.arange(
                10 + offset * 100, 10 + offset * 100 + new_tokens - self.pad_tail,
                dtype=torch.long,
            )
            pad = torch.full((self.pad_tail,), PAD_ID, dtype=torch.long)
            rows.append(torch.cat([prompt, body, pad]))
        return torch.stack(rows)

    def save_pretrained(self, path: str) -> None:
        self.saved_to.append(path)
        (Path(path) / "model.bin").write_bytes(b"x" * 2048)


class StubManager(ModelManager):
    """Real manager with the hub calls swapped out."""

    def __init__(self, settings: Settings | None = None, pad_tail: int = 0) -> None:
        super().__init__(settings or Settings(device="cpu"))
        self.pad_tail = pad_tail
        self.loaded_variants: List[str] = []

    def load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = FakeTokenizer()
            self.tokenizer.padding_side = "left"
        return self.tokenizer

    def _load_variant(self, variant):
        self.loaded_variants.append(variant.name)
        return FakeModel(pad_tail=self.pad_tail)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        device="cpu",
        cache_dir=tmp_path / "cache",
        quantized_model_path=tmp_path / "quantized",
        benchmark_dir=tmp_path / "benchmarks",
        preload=[],
    )


@pytest.fixture
def manager(settings) -> StubManager:
    stub = StubManager(settings)
    yield stub
    stub.shutdown()


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    from airavata_quant.api import create_app

    stub = StubManager(settings)
    with TestClient(create_app(settings, manager=stub)) as test_client:
        test_client.manager = stub
        yield test_client
