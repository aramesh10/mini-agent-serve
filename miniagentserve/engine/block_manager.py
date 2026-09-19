"""
block_manager.py
prefix sharing not supported yet
"""
from collections import deque

from miniagentserve.engine.sequence import Sequence

class KVBlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.total_blocks = num_blocks
        self.free_block_ids = deque(range(num_blocks))

    def num_blocks(self, num_tokens: int):
        return (num_tokens + self.block_size - 1) // self.block_size

    def utilization(self):
        return 1 - len(self.free_block_ids) / self.total_blocks

    def can_allocate(self, seq: Sequence):
        return len(self.free_block_ids) >= self.num_blocks(len(seq)) - len(seq.block_table)

    def allocate(self, seq: Sequence):
        while len(seq.block_table) < self.num_blocks(len(seq)):
            seq.block_table.append(self.free_block_ids.popleft())

    def deallocate(self, seq: Sequence):
        self.free_block_ids.extend(seq.block_table)
        seq.block_table.clear()
        seq.num_cached_tokens = 0
