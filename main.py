from __future__ import annotations

import asyncio
import dataclasses
from concurrent.futures import ThreadPoolExecutor

from m.pipeline import RenderOptions, RenderRejected, run


def generate(image: bytes, seed: int, witness, renderer=None):
    """Synchronous with a renderer; returns an awaitable when called as generate(image, seed, witness)."""
    if renderer is not None:
        return run(image, seed, witness.responses, renderer)
    return _generate_async(image, seed, witness)


async def _generate_async(image: bytes, seed: int, witness) -> str | None:
    loop = asyncio.get_running_loop()
    bridge = _Bridge(witness, loop)
    # A dedicated thread, so concurrent prompts never wait on the loop's shared default executor.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return await loop.run_in_executor(pool, run, image, seed, bridge, bridge)
    finally:
        pool.shutdown(wait=False)


class _Bridge:
    """Runs the pipeline's blocking model and render calls on the async witness client's event loop."""

    def __init__(self, witness, loop: asyncio.AbstractEventLoop) -> None:
        self.witness, self.loop = witness, loop
        import witness_client
        self.options = getattr(witness_client, "RenderOptions", RenderOptions)

    def wait(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def create(self, **kw):
        return self.wait(self.witness.responses.create(**kw))

    def render(self, source: str, options: RenderOptions) -> bytes:
        if self.options is not RenderOptions:
            options = self.options(**{f.name: getattr(options, f.name) for f in dataclasses.fields(RenderOptions)})
        try:
            return self.wait(self.witness.render(source, options))
        except RenderRejected:
            raise
        except Exception as e:
            if type(e).__name__ == "RenderRejected" and hasattr(e, "detail_json"):
                raise RenderRejected(e.detail_json) from e
            raise
