"""Deterministic hashed bag-of-words embedder for the dev/test index. Not for production.

An index built with it records model="fake-hash-embed"; retrieval then uses the same
embedder, so a fake index needs no embedding API calls.
"""

import re
import zlib

import numpy as np

HASH_EMBED_MODEL = "fake-hash-embed"


class HashEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[zlib.crc32(word.encode()) % self.dim] += 1.0
            out.append(v.tolist())
        return out
