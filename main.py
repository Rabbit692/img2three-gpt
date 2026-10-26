from __future__ import annotations

from m.pipeline import run


def generate(image: bytes, seed: int, witness, renderer) -> str | None:
    return run(image, seed, witness.responses, renderer)
