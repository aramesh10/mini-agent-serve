import itertools
import time
from collections import deque

import torch

from miniagentserve.engine.serving_engine import ServingEngine

class MetricsSampler:

    def __init__(self, engine: ServingEngine, history: int = 60, interval_s: float = 0.5):
        self.engine = engine
        self.interval_s = interval_s
        self.samples: deque[dict] = deque(maxlen=history)  # most recent engine samples

    def run(self):
        last_time, last_tokens = time.perf_counter(), self.engine.num_generated_tokens
        for i in itertools.count():
            time.sleep(self.interval_s)
            now, tokens, waiting = time.perf_counter(), self.engine.num_generated_tokens, list(self.engine.waiting)
            self.samples.append({
                "id": i,
                "output_tokens_per_s": round((tokens - last_tokens) / (now - last_time), 2),
                "gpu_utilization": torch.cuda.utilization(self.engine.model_runner.device) / 100,
                "kv_cache_utilization": round(self.engine.block_manager.utilization(), 4),
                "queueing_latency_ms": round(max((now - seq.enqueue_time for seq in waiting), default=0) * 1e3, 2),
            })
            last_time, last_tokens = now, tokens
