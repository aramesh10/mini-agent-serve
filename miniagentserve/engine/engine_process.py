"""
engine_process.py

Entry point for the engine process: receives requests on `in_q`, sends tokens and metrics back on `out_q`.
"""
import os
import threading

from miniagentserve.engine.metrics import MetricsSampler
from miniagentserve.engine.serving_engine import ServingEngine


def run_engine(path: str, in_q, out_q):
    # KV cache size and the longest sequence the CUDA graphs cover; longer sequences run eagerly
    engine = ServingEngine(path,
                           num_blocks=int(os.environ.get("MAS_NUM_BLOCKS", 4096)),
                           max_model_len=int(os.environ.get("MAS_MAX_MODEL_LEN", 8192)))
    sampler = MetricsSampler(engine, out_q)
    threading.Thread(target=engine.run, daemon=True).start()
    threading.Thread(target=sampler.run, daemon=True).start()
    while True:
        rid, prompt, params = in_q.get()
        try:
            engine.add_request(prompt, params, lambda text, finished, rid=rid: out_q.put(("token", rid, text, finished)))
        except ValueError as e:
            out_q.put(("error", rid, str(e)))
