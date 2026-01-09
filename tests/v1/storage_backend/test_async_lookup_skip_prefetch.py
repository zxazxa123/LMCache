# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import pytest

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.event_manager import EventStatus, EventType


class _DummyPinAllocator:
    def __init__(self, used: int, capacity: int):
        self.total_allocated_size = used
        # mimic TensorMemoryAllocator API
        self.buffer_size = capacity


class _DummyMixedMemAllocator:
    def __init__(self, used: int, capacity: int):
        self.pin_allocator = _DummyPinAllocator(used, capacity)


class _DummyAllocatorBackend:
    def __init__(self, used: int, capacity: int):
        self._mem = _DummyMixedMemAllocator(used, capacity)

    def get_memory_allocator(self):
        return self._mem


class _MockBackend:
    def __init__(self, hit_chunks: int):
        self.hit_chunks = hit_chunks
        self.prefetch_called = False

    async def batched_async_contains(self, lookup_id, keys, pin=False):
        # ensure we are not pinning when skipping prefetch
        assert pin is False
        return min(self.hit_chunks, len(keys))

    async def batched_get_non_blocking(self, *args, **kwargs):
        self.prefetch_called = True
        raise AssertionError("prefetch should not be called when skipping")


@pytest.mark.asyncio
async def test_async_lookup_skips_prefetch_when_dram_full(storage_manager):
    # Make CPU utilization look high so prefetch is skipped
    storage_manager.allocator_backend = _DummyAllocatorBackend(used=95, capacity=100)

    # Ensure config threshold triggers skipping
    storage_manager.config.async_loading_prefetch_max_cpu_utilization = 0.9
    storage_manager.config.enable_async_loading = True

    # Replace active backends with a single mock backend
    backend = _MockBackend(hit_chunks=2)

    def _iter_backends(*args, **kwargs):
        yield "LocalDiskBackend", backend

    storage_manager.get_active_storage_backends = _iter_backends  # type: ignore

    lookup_id = "l1"
    keys = [CacheEngineKey("m", i, 0, 0) for i in range(4)]
    # cum lengths: 0,10,20,30,40
    cum = [0, 10, 20, 30, 40]

    await storage_manager.async_lookup_and_prefetch(lookup_id, keys, cum)

    # Should respond immediately with hit tokens=20 (2 chunks)
    assert storage_manager.async_lookup_server.responses == [(lookup_id, 20)]

    # Should not register loading event
    assert (
        storage_manager.event_manager.get_event_status(EventType.LOADING, lookup_id)
        == EventStatus.NOT_FOUND
    )

    assert backend.prefetch_called is False
