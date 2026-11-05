import random
import unittest
from prefix_cache import PrefixCache


class PrefixCacheTests(unittest.TestCase):
    def test_split_preserves_slot_identity(self):
        cache = PrefixCache(32)
        _, old, node = cache.acquire([1, 2, 3, 4])
        cache.release(node)
        hit, new, node = cache.acquire([1, 2, 8])
        self.assertEqual(hit, 2)
        self.assertEqual(old[:2], new[:2])
        cache.release(node)
        self.assertEqual(cache.match([1, 2, 3, 4])[0], old)
        cache.validate()

    def test_pinned_split_and_eviction(self):
        cache = PrefixCache(6)
        _, slots, active = cache.acquire([1, 2, 3, 4])
        _, split = cache.match([1, 2, 9])
        self.assertEqual(split.refs, 1)
        with self.assertRaises(MemoryError):
            cache.acquire([8, 9, 10])
        self.assertEqual(cache.match([1, 2, 3, 4])[0], slots)
        cache.release(active)
        _, _, new = cache.acquire([8, 9, 10])
        cache.release(new)
        cache.validate()

    def test_shorter_and_exact_prefix(self):
        cache = PrefixCache(8)
        _, slots, last = cache.acquire([1, 2, 3])
        cache.release(last)
        hit, short, last = cache.acquire([1, 2])
        self.assertEqual((hit, short), (2, slots[:2]))
        cache.release(last)
        hit, exact, last = cache.acquire([1, 2, 3])
        self.assertEqual((hit, exact), (3, slots))
        cache.release(last)
        cache.validate()

    def test_randomized_lcp_oracle(self):
        rng = random.Random(42)
        cache = PrefixCache(8192)
        inserted = []
        for _ in range(400):
            tokens = tuple(rng.randrange(8) for _ in range(rng.randrange(1, 20)))
            expected = 0
            for old in inserted:
                count = 0
                for a, b in zip(old, tokens):
                    if a != b:
                        break
                    count += 1
                expected = max(expected, count)
            hit, _, node = cache.acquire(tokens)
            self.assertEqual(hit, expected)
            cache.release(node)
            inserted.append(tokens)
            cache.validate()

    def test_capacity_and_live_payloads(self):
        rng = random.Random(17)
        cache = PrefixCache(40)
        payload = {}
        for _ in range(400):
            tokens = tuple(rng.randrange(5) for _ in range(12))
            hit, slots, last = cache.acquire(tokens)
            for i in range(hit):
                self.assertEqual(payload[slots[i]], tokens[:i + 1])
            for i in range(hit, len(tokens)):
                payload[slots[i]] = tokens[:i + 1]
            cache.release(last)
            cache.validate()


if __name__ == "__main__":
    unittest.main()
