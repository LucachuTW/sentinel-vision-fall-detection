#!/usr/bin/env python3
"""Download pinned local development weights and verify their digests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

ARTIFACTS = {
    "models/yolo26n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt",
        "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
    ),
    "models/sam2.1_t.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt",
        "3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3",
    ),
    "artifacts/bus.jpg": (
        "https://ultralytics.com/images/bus.jpg",
        "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63",
    ),
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> None:
    for filename, (url, expected) in ARTIFACTS.items():
        destination = Path(filename)
        if destination.is_file() and digest(destination) == expected:
            print(f"verified {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)
        if digest(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"checksum mismatch for {destination}")
        temporary.replace(destination)
        print(f"downloaded and verified {destination}")


if __name__ == "__main__":
    main()
