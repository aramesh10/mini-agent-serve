"""
nvtx.py
NVTX annotations for Nsight Systems. Everything here is a no-op unless MAS_NVTX=1,
so the instrumented paths cost nothing when we are not profiling.

    MAS_NVTX=1 nsys profile -t cuda,nvtx --cuda-graph-trace=node --resolve-symbols=false \
        --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
        -o engine -f true python benchmark/profile_engine.py

The capture range starts once warmup is done, so torch.compile and CUDA graph
capture stay out of the report.

Ranges are coloured by what they tell you (see the palette below): warm for the
GPU-bound model path, cool for the CPU work between steps, red for the one place
the step actually waits on the GPU. torch.cuda.nvtx only wraps the ASCII entry
points, which carry no colour, so the event-attribute variants are bound directly.
"""
import ctypes
import functools
import os
from contextlib import contextmanager, nullcontext
from typing import Callable

import torch

ENABLED = os.environ.get("MAS_NVTX", "0") == "1"

# 0xAARRGGBB. Grouped by what a wide band of it means.
STEP      = 0xFF546E7A  # slate   - one engine iteration, the container for everything below
SCHEDULE  = 0xFFAB47BC  # purple  - admission and block allocation, pure Python
PREPARE   = 0xFFFDD835  # yellow  - building input tensors and copying them to the device
PREFILL   = 0xFFFF8F00  # orange  - prompt forward pass
DECODE    = 0xFF1E88E5  # blue    - single-token forward pass
REPLAY    = 0xFF4FC3F7  # cyan    - CUDA graph replay, nested inside decode
FORWARD   = 0xFF7E57C2  # violet  - eager or compiled forward, the non-graph path
SAMPLE    = 0xFF26A69A  # teal    - sampling kernels
SYNC      = 0xFFE53935  # red     - blocking on the GPU; this band is your real GPU time
EMIT      = 0xFFEC407A  # pink    - detokenize and hand the token to the client
IDLE      = 0xFF90A4AE  # grey    - engine had no work
REQUEST   = 0xFF66BB6A  # green   - one request, from arrival to its last token

_NVTX_VERSION = 3
_ATTR_SIZE = 48           # sizeof(nvtxEventAttributes_t) on 64-bit
_COLOR_ARGB = 1
_MESSAGE_ASCII = 1


def _load_library():
    for name in ("libnvToolsExt.so.1", "libnvToolsExt.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class _Payload(ctypes.Union):
    _fields_ = [("ullValue", ctypes.c_uint64), ("llValue", ctypes.c_int64), ("dValue", ctypes.c_double)]


class _Message(ctypes.Union):
    _fields_ = [("ascii", ctypes.c_char_p), ("unicode", ctypes.c_void_p)]


class _EventAttributes(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint16),
        ("size", ctypes.c_uint16),
        ("category", ctypes.c_uint32),
        ("colorType", ctypes.c_int32),
        ("color", ctypes.c_uint32),
        ("payloadType", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
        ("payload", _Payload),
        ("messageType", ctypes.c_int32),
        ("message", _Message),
    ]


_lib = _load_library() if ENABLED else None

if _lib is not None:
    _lib.nvtxRangePushEx.argtypes = [ctypes.POINTER(_EventAttributes)]
    _lib.nvtxRangePushEx.restype = ctypes.c_int
    _lib.nvtxRangeStartEx.argtypes = [ctypes.POINTER(_EventAttributes)]
    _lib.nvtxRangeStartEx.restype = ctypes.c_uint64
    _lib.nvtxRangeEnd.argtypes = [ctypes.c_uint64]
    _lib.nvtxRangeEnd.restype = None
    _lib.nvtxMarkEx.argtypes = [ctypes.POINTER(_EventAttributes)]
    _lib.nvtxMarkEx.restype = None
    _lib.nvtxRangePop.argtypes = []
    _lib.nvtxRangePop.restype = ctypes.c_int


def _attributes(msg: str, color: int) -> _EventAttributes:
    attrs = _EventAttributes()
    attrs.version = _NVTX_VERSION
    attrs.size = _ATTR_SIZE
    attrs.colorType = _COLOR_ARGB
    attrs.color = color
    attrs.messageType = _MESSAGE_ASCII
    attrs._encoded = msg.encode()   # the struct only holds a pointer: keep the bytes alive
    attrs.message.ascii = attrs._encoded
    return attrs


if ENABLED and _lib is not None:

    @contextmanager
    def range(msg: str, color: int = STEP):
        """Scoped range on the calling thread; nests in the timeline."""
        _lib.nvtxRangePushEx(ctypes.byref(_attributes(msg, color)))
        try:
            yield
        finally:
            _lib.nvtxRangePop()

    def annotate(color: int = STEP) -> Callable:
        """Ranges the whole call, labelled with the qualified name."""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                _lib.nvtxRangePushEx(ctypes.byref(_attributes(fn.__qualname__, color)))
                try:
                    return fn(*args, **kwargs)
                finally:
                    _lib.nvtxRangePop()
            return wrapper
        return decorator

    def mark(msg: str, color: int = STEP) -> None:
        _lib.nvtxMarkEx(ctypes.byref(_attributes(msg, color)))

    def start_range(msg: str, color: int = REQUEST) -> int:
        """Free-standing range: unlike `range`, it may outlive the frame that opened it."""
        return _lib.nvtxRangeStartEx(ctypes.byref(_attributes(msg, color)))

    def end_range(handle: int | None) -> None:
        if handle is not None:
            _lib.nvtxRangeEnd(handle)

elif ENABLED:   # no libnvToolsExt to bind: fall back to torch's uncoloured ranges

    @contextmanager
    def range(msg: str, color: int = STEP):
        torch.cuda.nvtx.range_push(msg)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def annotate(color: int = STEP) -> Callable:
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                torch.cuda.nvtx.range_push(fn.__qualname__)
                try:
                    return fn(*args, **kwargs)
                finally:
                    torch.cuda.nvtx.range_pop()
            return wrapper
        return decorator

    def mark(msg: str, color: int = STEP) -> None:
        torch.cuda.nvtx.mark(msg)

    def start_range(msg: str, color: int = REQUEST) -> int:
        return torch.cuda.nvtx.range_start(msg)

    def end_range(handle: int | None) -> None:
        if handle is not None:
            torch.cuda.nvtx.range_end(handle)

else:

    def range(msg: str, color: int = STEP):
        return nullcontext()

    def annotate(color: int = STEP) -> Callable:
        return lambda fn: fn

    def mark(msg: str, color: int = STEP) -> None:
        pass

    def start_range(msg: str, color: int = REQUEST) -> None:
        return None

    def end_range(handle: int | None) -> None:
        pass


def profiler_start() -> None:
    if ENABLED:
        torch.cuda.profiler.start()


def profiler_stop() -> None:
    if ENABLED:
        torch.cuda.profiler.stop()
