"""
Helper utilities for VL-Checklist integration.

This module provides utility functions used by the VL-Checklist wrapper.
"""

from collections import OrderedDict
from typing import Any, Iterator, List, TypeVar

from PIL import Image
from torchvision import transforms

T = TypeVar("T")


class MinMaxResize:
    def __init__(self, shorter=800, longer=1333):
        self.min = shorter
        self.max = longer

    def __call__(self, x):
        w, h = x.size
        scale = self.min / min(w, h)
        if h < w:
            newh, neww = self.min, scale * w
        else:
            newh, neww = scale * h, self.min

        if max(newh, neww) > self.max:
            scale = self.max / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        newh, neww = int(newh + 0.5), int(neww + 0.5)
        newh, neww = newh // 32 * 32, neww // 32 * 32

        return x.resize((neww, newh), resample=Image.BICUBIC)


# This is simple maximum entropy normalization performed in Inception paper
inception_normalize = transforms.Compose(
    [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)


class LRUCache:
    """
    Simple LRU (Least Recently Used) cache implementation.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str) -> Any:
        """Get value by key and mark as recently used."""
        if key in self.cache:
            # Move to end (mark as recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put key-value pair into cache."""
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)

        self.cache[key] = value

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache


def chunks(lst: List[T], n: int) -> Iterator[List[T]]:
    """
    Yield successive n-sized chunks from lst.

    Args:
        lst: List to chunk
        n: Chunk size

    Yields:
        Lists of size n (or smaller for the last chunk)
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
