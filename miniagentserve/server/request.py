"""
request.py

What a client sends to the server. The engine turns it into a `Sequence`.
"""
from dataclasses import dataclass

from miniagentserve.engine.sequence import SamplingParams


@dataclass(kw_only=True)
class Request(SamplingParams):
    prompt: str
