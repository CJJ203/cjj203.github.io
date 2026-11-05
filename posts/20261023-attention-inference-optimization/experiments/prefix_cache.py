"""Small compressed radix cache, standard library only.

Token IDs are synthetic; slot IDs stand in for already computed GPU KV entries.
Single model/namespace, page size 1, single threaded, no model execution.
"""
from dataclasses import dataclass, field
import json


@dataclass(eq=False)
class Node:
    key: tuple = ()
    slots: tuple = ()
    parent: "Node | None" = None
    children: dict = field(default_factory=dict)
    refs: int = 0
    touched: int = 0


class PrefixCache:
    def __init__(self, capacity):
        self.root = Node()
        self.capacity = capacity
        self.free = list(range(capacity))
        self.clock = 0
        self.evicted_tokens = 0

    def touch(self, node):
        self.clock += 1
        node.touched = self.clock

    def split(self, child, count):
        parent = child.parent
        mid = Node(child.key[:count], child.slots[:count], parent,
                   refs=child.refs, touched=child.touched)
        parent.children[child.key[0]] = mid
        child.key, child.slots = child.key[count:], child.slots[count:]
        child.parent = mid
        mid.children[child.key[0]] = child
        return mid

    def match(self, tokens):
        """Return longest prefix slots and a node ending exactly at the match."""
        rest, slots, node = tuple(tokens), (), self.root
        while rest and rest[0] in node.children:
            child = node.children[rest[0]]
            count = 0
            for a, b in zip(rest, child.key):
                if a != b:
                    break
                count += 1
            if count < len(child.key):
                child = self.split(child, count)
            slots += child.slots
            rest = rest[count:]
            node = child
            self.touch(node)
        return slots, node

    def pin(self, node, delta):
        while node is not self.root:
            node.refs += delta
            assert node.refs >= 0
            node = node.parent

    def nodes(self):
        pending = list(self.root.children.values())
        while pending:
            node = pending.pop()
            yield node
            pending.extend(node.children.values())

    def evict(self, needed):
        # Whole leaves only: a surviving descendant always needs its prefix.
        while len(self.free) < needed:
            leaves = [n for n in self.nodes() if not n.children and n.refs == 0]
            if not leaves:
                raise MemoryError("all remaining cache paths are pinned")
            victim = min(leaves, key=lambda n: n.touched)
            del victim.parent.children[victim.key[0]]
            self.free.extend(victim.slots)
            self.evicted_tokens += len(victim.slots)

    def acquire(self, tokens):
        tokens = tuple(tokens)
        if len(tokens) > self.capacity:
            raise ValueError("request exceeds cache capacity")
        slots, prefix = self.match(tokens)
        hit = len(slots)
        self.pin(prefix, 1)
        try:
            self.evict(len(tokens) - hit)
        except MemoryError:
            self.pin(prefix, -1)
            raise
        if hit < len(tokens):
            fresh = tuple(self.free.pop() for _ in range(len(tokens) - hit))
            leaf = Node(tokens[hit:], fresh, prefix)
            prefix.children[tokens[hit]] = leaf
            self.touch(leaf)
            self.pin(leaf, 1)
            self.pin(prefix, -1)
            slots, last = slots + fresh, leaf
        else:
            last = prefix
        # Caller would fill only the new slots with actual KV here.
        return hit, slots, last

    def release(self, last):
        self.pin(last, -1)

    def validate(self):
        occupied = []
        for n in self.nodes():
            assert len(n.key) == len(n.slots) > 0
            assert n.parent.children[n.key[0]] is n
            assert n.refs >= 0
            occupied.extend(n.slots)
        all_slots = occupied + self.free
        assert len(all_slots) == self.capacity
        assert len(set(all_slots)) == self.capacity


def workload(kind, count=100):
    for i in range(count):
        suffix = tuple(range(100000 + i * 64, 100000 + (i + 1) * 64))
        if kind == "unique":
            yield tuple(range(1000000 + i * 256, 1000000 + (i + 1) * 256))
        elif kind == "shared192":
            yield tuple(range(192)) + suffix
        elif kind == "four_groups":
            group = i % 4
            yield tuple(range(48)) + tuple(range(1000 + group * 144, 1000 + (group + 1) * 144)) + suffix
        else:
            raise ValueError(kind)


def experiment(kind, capacity):
    cache = PrefixCache(capacity)
    hits = total = requests_hit = 0
    for tokens in workload(kind):
        hit, slots, last = cache.acquire(tokens)
        hits += hit
        total += len(tokens)
        requests_hit += hit > 0
        cache.release(last)
        cache.validate()
    return {"workload": kind, "capacity_tokens": capacity, "requests": 100,
            "input_tokens": total, "reused_tokens": hits,
            "token_reuse_rate": hits / total, "requests_with_hit": requests_hit,
            "evicted_tokens": cache.evicted_tokens,
            "resident_tokens": capacity - len(cache.free)}


if __name__ == "__main__":
    print(json.dumps([experiment(kind, cap) for kind, cap in
        [("unique", 32768), ("shared192", 32768), ("four_groups", 32768),
         ("shared192", 512), ("four_groups", 512)]], indent=2))
