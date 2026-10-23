"""Run vLLM v0.11.0's CPU block metadata classes without importing the engine.

Read the two upstream files from --source-dir, or download their pinned contents.
Only the selected AST definitions are executed, unchanged; GPU tensors, engine
imports, logging setup and KV events are outside this experiment.
"""
import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, NewType, Optional, Union
from urllib.request import urlopen


FILES = {
    "kv_cache_utils.py": "0824beae09a3a8faeb54e8e0605d1661045e34a33a2026de537e6849a4c78ee4",
    "block_pool.py": "8cd4b2fe35a26dcf8aa7d8e8ec16922b7cd3e67e02c800026902dad9b1274368",
}
BASE = "https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/vllm/v1/core/"


def load_classes(source_dir):
    module = ModuleType("_vllm_block_experiment")
    sys.modules[module.__name__] = module
    module.__dict__.update({name: globals()[name] for name in
        ("Any", "NewType", "Optional", "Union", "Iterable", "dataclass")})
    module.logger = logging.getLogger(__name__)
    selected = {"BlockHash", "BlockHashWithGroupId", "ExternalBlockHash",
                "make_block_hash_with_group_id", "KVCacheBlock",
                "FreeKVCacheBlockQueue", "BlockHashToBlockMap", "BlockPool"}
    for filename, digest in FILES.items():
        if source_dir:
            data = (Path(source_dir) / filename).read_bytes()
        else:
            with urlopen(BASE + filename, timeout=30) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"upstream source checksum mismatch: {filename}")
        nodes = []
        for node in ast.parse(data).body:
            name = getattr(node, "name", None)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            if name in selected:
                nodes.append(node)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
        exec(compile(tree, filename, "exec"), module.__dict__)
    return module


def experiment(source_dir):
    upstream = load_classes(source_dir)
    pool = upstream.BlockPool(num_gpu_blocks=9, enable_caching=True)
    snapshots = []
    hashes = [hashlib.sha256(f"prefix-{i}".encode()).digest() for i in range(2)]
    a = b = []

    def snapshot(stage):
        snapshots.append({"stage": stage, "free_blocks": pool.get_num_free_blocks(),
            "shared_refs": [pool.blocks[i].ref_cnt for i in (1, 2)],
            "prefix_cached": [pool.get_cached_block(h, [0]) is not None for h in hashes]})

    snapshot("initial")
    a = pool.get_new_blocks(3)  # A: 37 tokens, 3 blocks of 16.
    pool.cache_full_blocks(SimpleNamespace(block_hashes=hashes), a, 0, 2, 16, 0)
    snapshot("A allocated and first 32 tokens cached")
    shared = [pool.get_cached_block(h, [0])[0] for h in hashes]
    pool.touch((shared,))
    b = shared + pool.get_new_blocks(1)  # B shares 32 tokens, adds 8.
    assert [x.block_id for x in a] == [1, 2, 3]
    assert [x.block_id for x in b] == [1, 2, 4]
    snapshot("B shares first two blocks")
    pool.free_blocks(reversed(a))
    snapshot("A finished")
    assert [x.ref_cnt for x in shared] == [1, 1]
    pool.free_blocks(reversed(b))
    snapshot("B finished")
    assert pool.get_num_free_blocks() == 8
    assert all(pool.get_cached_block(h, [0]) is not None for h in hashes)
    pool.touch((shared,))
    snapshot("cached prefix touched again")
    assert [x.block_id for x in shared] == [1, 2]
    pool.free_blocks(reversed(shared))
    reused = pool.get_new_blocks(8)  # Reassignment evicts cached hash entries.
    snapshot("all eight blocks reassigned")
    assert all(pool.get_cached_block(h, [0]) is None for h in hashes)
    pool.free_blocks(reversed(reused))
    assert pool.get_num_free_blocks() == 8

    lengths = [37, 40]
    reservation = 128  # Deliberate fixed-maximum allocation baseline.
    return {"vllm_version": "v0.11.0", "source_sha256": FILES,
        "python": sys.version.split()[0], "block_size": 16,
        "physical_blocks": 9, "usable_blocks": 8,
        "block_tables": {"A": [x.block_id for x in a], "B": [x.block_id for x in b]},
        "slot_example": {"position": 35, "block_id": a[2].block_id,
                         "offset": 3, "slot": a[2].block_id * 16 + 3},
        "capacity_comparison": {
            "request_lengths": lengths, "shared_prefix_tokens": 32,
            "fixed_max_per_request": reservation,
            "fixed_reserved_slots": len(lengths) * reservation,
            "paged_without_sharing_slots": sum((n + 15) // 16 * 16 for n in lengths),
            "paged_with_sharing_slots": 4 * 16,
            "unique_valid_tokens": sum(lengths) - 32},
        "snapshots": snapshots}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", help="directory containing the two vLLM source files")
    args = parser.parse_args()
    print(json.dumps(experiment(args.source_dir), indent=2))
