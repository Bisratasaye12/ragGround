"""Blob storage for uploaded originals. Local disk now; an S3 implementation can satisfy the same
protocol for deployment (M8)."""

import asyncio
import re
from pathlib import Path
from typing import Protocol

_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/_.-]*$")


class BlobStorage(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        # Keys are generated server-side, but refuse anything that could escape the root anyway.
        if not _SAFE_KEY.match(key) or ".." in key.split("/"):
            raise ValueError(f"unsafe storage key: {key!r}")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"unsafe storage key: {key!r}")
        return path

    async def put(self, key: str, data: bytes) -> None:
        path = self._path(key)

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)  # atomic: readers never see a half-written file

        await asyncio.to_thread(write)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._path(key).unlink, missing_ok=True)
