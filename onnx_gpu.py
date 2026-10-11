"""ONNX Runtime on a GPU shared with Ollama: the STT and TTS models' runs and memory (AD-053).

Per session, ONNX Runtime keeps an arena that grows to fit each new input length and never hands
memory back. With requests and replies of a new length every time, the voice server went from 4.6
to 7.2 GB, and beside Ollama's 8.6 GB the 16 GB card ran out (`CUBLAS failure 3`).

A model that finds no room on the card runs on the CPU, slower, until CUDA works again (AD-055).
"""

import asyncio
import functools
import logging
import weakref
from concurrent.futures import ThreadPoolExecutor

import onnxruntime

logger = logging.getLogger(__name__)

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"

# Every model run on one thread: two runs never add up their memory, and ONNX Runtime makes its
# per-thread CUDA handles once, not whenever a request lands on a new thread of asyncio's pool
_executor = ThreadPoolExecutor(1, thread_name_prefix="onnx-gpu")

_SHRINK = onnxruntime.RunOptions()
_SHRINK.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")


def has_cuda() -> bool:
    return CUDA in onnxruntime.get_available_providers()


def session_kwargs(gpu: bool) -> dict:
    """InferenceSession's arguments, on CUDA or the CPU.

    On CUDA, its arena grown by what a run asks, not to the next power of two, weights allocated
    outside it, so that it holds only what runs need, all freed after them. And no fallback: when
    CUDA failed (the card full), ONNX Runtime's Python API retried on the CPU, printing an
    `EP Error` and nothing else, and STT and TTS ran on the CPU for two hours unnoticed."""
    options = onnxruntime.SessionOptions()
    options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    if gpu:
        providers = [(CUDA, {"arena_extend_strategy": "kSameAsRequested"}), CPU]
        return {"sess_options": options, "providers": providers, "enable_fallback": 0}
    return {"sess_options": options, "providers": [CPU]}


def device(session: onnxruntime.InferenceSession) -> str:
    return "cuda" if session.get_providers()[0] == CUDA else "cpu"


def lost_cuda(engine) -> bool:
    """The engine's model on the CPU though onnxruntime has CUDA: CUDA failed when it loaded."""
    return engine.device == "cpu" and has_cuda()


def _on_cuda(build, check):
    """build(True), then check(model), a run: on a thread where CUDA failed, the first run fails
    too, once (a stale `out of memory`). The model, or None and CUDA's error. A function of its
    own: a failed model is freed when it returns, before the CPU's loads."""
    try:
        model = build(True)
        check(model)
        return model, None
    # Not only what ONNX Runtime's own fallback catches (ValueError, RuntimeError): weights that
    # find no room raise its `Fail`, an Exception, and the server stopped at startup
    except Exception as e:
        return None, str(e)


def load(build, check, name: str):
    """build(gpu): on CUDA when onnxruntime has it and it works, else on the CPU, logged: there, STT
    takes 5 times as long, TTS 10 times."""
    if has_cuda():
        model, error = _on_cuda(build, check)
        if model is not None:
            return model
        logger.error(f"{name} on the CPU, slower, until CUDA works again: {error}")
    return build(False)


def retry(build, check, name: str):
    """The model on CUDA if it works now, else None."""
    model, error = _on_cuda(build, check)
    if model is None:
        logger.debug(f"{name} still on the CPU: {error}")
    else:
        logger.info(f"{name} back on CUDA")
    return model


def shrinking(session: onnxruntime.InferenceSession) -> onnxruntime.InferenceSession:
    """The session, its arena handing back after each run the memory the run needed. For inputs that
    vary in length; not for Parakeet's decoder: it runs once per audio frame, where allocating again
    each time cost 32 ms per transcription, and its arena does not grow."""
    if device(session) == "cuda":
        # Through a proxy: session.run bound in the partial, the session held itself, and a model
        # dropped after a failed retry stayed on the card until the garbage collector's next pass
        session.run = functools.partial(type(session).run, weakref.proxy(session), run_options=_SHRINK)
    return session


async def run(fn, *args):
    """fn(*args) on the models' thread."""
    return await asyncio.get_running_loop().run_in_executor(_executor, fn, *args)
