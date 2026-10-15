"""
block_manager.py
"""
from array import array

import xxhash

from miniagentserve.engine.sequence import Sequence

class KVBlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.total_blocks = num_blocks
        self.free_block_ids = dict.fromkeys(range(num_blocks))    # insertion ordered: front is least recently freed
        self.ref_count = [0] * num_blocks
        self.block_hash: list[int | None] = [None] * num_blocks   # kept after a block is freed, so it can still be hit
        self.hash_to_block: dict[int, int] = {}
        self.num_query_tokens = 0
        self.num_hit_tokens = 0

    def num_blocks(self, num_tokens: int):
        return (num_tokens + self.block_size - 1) // self.block_size

    def utilization(self):
        return 1 - len(self.free_block_ids) / self.total_blocks

    def hash_block(self, parent: int, token_ids: list[int]) -> int:
        h = xxhash.xxh3_128()
        h.update(parent.to_bytes(16, "little"))
        h.update(array("I", token_ids))
        return h.intdigest()
    
    def match_prefix(self, seq: Sequence) -> list[tuple[int, int]]:
        hits, h = [], 0
        for i in range((len(seq) - 1) // self.block_size):
            h = self.hash_block(h, seq.token_ids[i * self.block_size:(i + 1) * self.block_size])
            block = self.hash_to_block.get(h)
            if block is None: break
            hits.append((h, block))
        return hits

    def can_allocate(self, seq: Sequence):
        needed = self.num_blocks(len(seq)) - len(seq.block_table)
        if not seq.block_table:
            # hits on in-use blocks cost nothing; hits on free blocks still take them off the free list
            hits = self.match_prefix(seq)
            needed -= sum(self.ref_count[block] > 0 for _, block in hits)
        return len(self.free_block_ids) >= needed

    def allocate(self, seq: Sequence):
        if not seq.block_table:
            hits = self.match_prefix(seq)
            for h, block in hits:
                if self.ref_count[block] == 0:
                    del self.free_block_ids[block]
                self.ref_count[block] += 1
                seq.block_table.append(block)
                seq.block_hashes.append(h)
            seq.num_cached_tokens = len(hits) * self.block_size
            self.num_query_tokens += len(seq)
            self.num_hit_tokens += seq.num_cached_tokens
            
        while len(seq.block_table) < self.num_blocks(len(seq)):
            block = next(iter(self.free_block_ids))
            del self.free_block_ids[block]
            h = self.block_hash[block]
            if h is not None:   # evict its cached contents
                if self.hash_to_block.get(h) == block:
                    del self.hash_to_block[h]
                self.block_hash[block] = None
            self.ref_count[block] = 1
            seq.block_table.append(block)

    def commit(self, seq: Sequence):
        """Registers the full blocks whose K/V are now written, so later sequences can hit them."""
        while (i := len(seq.block_hashes)) < seq.num_cached_tokens // self.block_size:
            token_ids = seq.token_ids[i * self.block_size:(i + 1) * self.block_size]
            if min(token_ids) < 0: break    # sampled token not back yet, retry next step
            h = self.hash_block(seq.block_hashes[-1] if i else 0, token_ids)
            block = seq.block_table[i]
            self.block_hash[block] = h
            self.hash_to_block.setdefault(h, block)
            seq.block_hashes.append(h)

    def deallocate(self, seq: Sequence):
        for block in reversed(seq.block_table):     # tail blocks are evicted first, since they are the least shared
            self.ref_count[block] -= 1
            if self.ref_count[block] == 0:
                self.free_block_ids[block] = None
        seq.block_table.clear()
        seq.block_hashes.clear()
        seq.num_cached_tokens = 0
