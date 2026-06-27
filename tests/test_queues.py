from sentinel_vision.queues import LatestValueQueue


def test_latest_queue_drops_oldest() -> None:
    drops = 0

    def on_drop() -> None:
        nonlocal drops
        drops += 1

    queue: LatestValueQueue[int] = LatestValueQueue(2, on_drop)
    queue.put_latest(1)
    queue.put_latest(2)
    queue.put_latest(3)
    assert queue.dropped == 1
    assert drops == 1
    assert queue.get_nowait() == 2
    assert queue.get_nowait() == 3
