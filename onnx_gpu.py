"""ONNX Runtime on a GPU shared with Ollama: the STT and TTS models' runs and memory (AD-053).

Per session, ONNX Runtime keeps an arena that grows to fit each new input length and never hands
memory back. With requests and replies of a new length every time, the voice server went from 4.6
to 7.2 GB, and beside Ollama's 8.6 GB the 16 GB card ran out (`CUBLAS failure 3`).
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

import onnxruntime

CUDA = "CUDAExecutionProvider"

# Every model run on one thread: two runs never add up their memory, and ONNX Runtime makes its
# per-thread CUDA handles once, not whenever a request lands on a new thread of asyncio's pool
_executor = ThreadPoolExecutor(1, thread_name_prefix="onnx-gpu")

_SHRINK = onnxruntime.RunOptions()
_SHRINK.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")


def providers() -> list:
    """CUDA when onnxruntime has it, its arena grown by what a run asks, not to the next power of two."""
    if CUDA in onnxruntime.get_available_providers():
        return [(CUDA, {"arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def session_options() -> onnxruntime.SessionOptions:
    """Weights allocated outside the arena, which then holds only what runs need, all freed after them."""
    options = onnxruntime.SessionOptions()
    options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    return options


def shrinking(session: onnxruntime.InferenceSession) -> onnxruntime.InferenceSession:
    """The session, its arena handing back after each run the memory the run needed. For inputs that
    vary in length; not for Parakeet's decoder: it runs once per audio frame, where allocating again
    each time cost 32 ms per transcription, and its arena does not grow."""
    if session.get_providers()[0] == CUDA:
        session.run = functools.partial(session.run, run_options=_SHRINK)
    return session


async def run(fn, *args):
    """fn(*args) on the models' thread."""
    return await asyncio.get_running_loop().run_in_executor(_executor, fn, *args)
