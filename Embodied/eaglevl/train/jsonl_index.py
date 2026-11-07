# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Compact byte-offset indexes for random JSONL access.

The training recipes can reference hundreds of large JSONL files.  Scanning all
of them independently in every distributed rank is both slow and wasteful, so
this module builds one versioned index per source file and memory maps it in all
readers. Index files contain a flat little-endian uint64 array of byte offsets.
"""

from __future__ import annotations

import bisect
import fcntl
import hashlib
import json
import mmap
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


INDEX_DTYPE = np.dtype("<u8")
INDEX_VERSION = 1
DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class JsonlIndexInfo:
    source_path: str
    index_path: str
    rows: int
    expected_rows: Optional[int]
    elapsed_seconds: float


def _normalized_source_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _path_digest(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]


def _source_signature(path: str) -> Tuple[int, int, int]:
    stat_result = os.stat(path)
    if not os.path.isfile(path):
        raise ValueError(f"JSONL source is not a regular file: {path}")
    return stat_result.st_size, stat_result.st_mtime_ns, stat_result.st_ctime_ns


def _index_path(index_dir: str, source_path: str, signature: Tuple[int, int, int]) -> str:
    size, mtime_ns, ctime_ns = signature
    filename = (
        f"v{INDEX_VERSION}-{_path_digest(source_path)}-"
        f"{size:x}-{mtime_ns:x}-{ctime_ns:x}.idx"
    )
    return os.path.join(index_dir, filename)


def _write_nonblank_starts(
    output_handle,
    mapping: mmap.mmap,
    starts: np.ndarray,
    ends: np.ndarray,
    chunk_start: int,
    chunk_view: np.ndarray,
) -> int:
    """Write starts for non-empty/non-whitespace lines and return their count."""
    nonempty = ends > starts
    if not np.any(nonempty):
        return 0

    candidate_starts = starts[nonempty]
    candidate_ends = ends[nonempty]
    leading = np.empty(candidate_starts.size, dtype=np.uint8)

    in_chunk = candidate_starts >= chunk_start
    if np.any(in_chunk):
        relative_starts = (candidate_starts[in_chunk] - chunk_start).astype(
            np.intp, copy=False
        )
        leading[in_chunk] = chunk_view[relative_starts]
    if np.any(~in_chunk):
        for position in np.flatnonzero(~in_chunk):
            leading[position] = mapping[int(candidate_starts[position])]

    maybe_blank = (
        (leading == ord(" "))
        | ((leading >= ord("\t")) & (leading <= ord("\r")))
    )
    keep = np.ones(candidate_starts.size, dtype=bool)
    for position in np.flatnonzero(maybe_blank):
        start = int(candidate_starts[position])
        end = int(candidate_ends[position])
        if not mapping[start:end].strip():
            keep[position] = False

    selected = candidate_starts[keep].astype(INDEX_DTYPE, copy=False)
    output_handle.write(selected.tobytes())
    return int(selected.size)


def _build_index_file(
    source_path: str,
    index_path: str,
    signature: Tuple[int, int, int],
    chunk_size: int,
) -> int:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    temporary_path = (
        f"{index_path}.tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    )
    rows = 0
    try:
        with open(source_path, "rb") as source_handle, open(
            temporary_path, "wb", buffering=1024 * 1024
        ) as output_handle:
            file_size = signature[0]
            if file_size:
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
                    try:
                        os.posix_fadvise(
                            source_handle.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL
                        )
                    except OSError:
                        pass

                mapping = mmap.mmap(source_handle.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_SEQUENTIAL"):
                        try:
                            mapping.madvise(mmap.MADV_SEQUENTIAL)
                        except (OSError, ValueError):
                            pass

                    line_start = 0
                    for chunk_start in range(0, file_size, chunk_size):
                        chunk_end = min(file_size, chunk_start + chunk_size)
                        chunk_view = np.frombuffer(
                            mapping,
                            dtype=np.uint8,
                            count=chunk_end - chunk_start,
                            offset=chunk_start,
                        )
                        newline_positions = np.flatnonzero(chunk_view == ord("\n"))
                        if newline_positions.size:
                            ends = newline_positions.astype(np.uint64, copy=False)
                            ends += np.uint64(chunk_start)
                            starts = np.empty(ends.size, dtype=np.uint64)
                            starts[0] = line_start
                            if ends.size > 1:
                                starts[1:] = ends[:-1] + 1
                            rows += _write_nonblank_starts(
                                output_handle,
                                mapping,
                                starts,
                                ends,
                                chunk_start,
                                chunk_view,
                            )
                            line_start = int(ends[-1]) + 1

                        # mmap cannot be closed while a NumPy view still references it.
                        del chunk_view

                    if line_start < file_size and mapping[line_start:file_size].strip():
                        output_handle.write(
                            np.asarray([line_start], dtype=INDEX_DTYPE).tobytes()
                        )
                        rows += 1
                finally:
                    mapping.close()

            output_handle.flush()
            os.fsync(output_handle.fileno())

        if _source_signature(source_path) != signature:
            raise RuntimeError(f"JSONL source changed while it was being indexed: {source_path}")
        os.replace(temporary_path, index_path)
        return rows
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


def build_jsonl_index(
    source_path: str,
    index_dir: str,
    expected_rows: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> JsonlIndexInfo:
    """Rebuild a source-versioned index, atomically replacing any prior file."""
    started = time.monotonic()
    source_path = _normalized_source_path(source_path)
    index_dir = os.path.abspath(os.path.expanduser(index_dir))
    os.makedirs(index_dir, exist_ok=True)

    lock_path = os.path.join(index_dir, f"{_path_digest(source_path)}.lock")
    with open(lock_path, "a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        signature = _source_signature(source_path)
        index_path = _index_path(index_dir, source_path, signature)
        _build_index_file(source_path, index_path, signature, chunk_size)

    rows = os.path.getsize(index_path) // INDEX_DTYPE.itemsize
    return JsonlIndexInfo(
        source_path=source_path,
        index_path=index_path,
        rows=rows,
        expected_rows=expected_rows,
        elapsed_seconds=time.monotonic() - started,
    )


def prepare_jsonl_indices(
    sources: Mapping[str, Optional[int]],
    index_dir: str,
    workers: int = 16,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, JsonlIndexInfo]:
    """Rebuild indexes concurrently and return results keyed by input path."""
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    unique_sources: Dict[str, Optional[int]] = {}
    original_to_normalized: Dict[str, str] = {}
    for source_path, expected_rows in sources.items():
        normalized = _normalized_source_path(source_path)
        original_to_normalized[source_path] = normalized
        if normalized in unique_sources:
            existing = unique_sources[normalized]
            if existing is not None and expected_rows is not None and existing != expected_rows:
                raise ValueError(
                    f"Conflicting expected row counts for {normalized}: "
                    f"{existing} and {expected_rows}"
                )
            if existing is None:
                unique_sources[normalized] = expected_rows
        else:
            unique_sources[normalized] = expected_rows

    by_normalized: Dict[str, JsonlIndexInfo] = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(unique_sources)))) as executor:
        futures = {
            executor.submit(
                build_jsonl_index,
                source_path,
                index_dir,
                expected_rows,
                chunk_size,
            ): source_path
            for source_path, expected_rows in unique_sources.items()
        }
        for future in as_completed(futures):
            source_path = futures[future]
            by_normalized[source_path] = future.result()

    return {
        original_path: by_normalized[normalized]
        for original_path, normalized in original_to_normalized.items()
    }


def load_offset_array(index_path: str) -> np.ndarray:
    size = os.path.getsize(index_path)
    if size % INDEX_DTYPE.itemsize:
        raise ValueError(f"Corrupt JSONL index (invalid byte size): {index_path}")
    if size == 0:
        return np.empty(0, dtype=INDEX_DTYPE)
    return np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")


def _active_index_spec(
    index_dir: str,
    source_identity: str,
    total_rows: int,
    sample_rows: int,
    seed: int,
) -> Tuple[str, np.dtype]:
    if not 0 <= sample_rows <= total_rows:
        raise ValueError(
            f"sample_rows must be in [0, total_rows], got {sample_rows}/{total_rows}"
        )
    dtype = np.dtype("<u4") if total_rows <= np.iinfo(np.uint32).max else INDEX_DTYPE
    identity = f"{source_identity}\0{total_rows}\0{sample_rows}\0{seed}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    suffix = "u32" if dtype.itemsize == 4 else "u64"
    return os.path.join(index_dir, f"active-{digest}-{sample_rows:x}.{suffix}"), dtype


def build_active_index(
    index_dir: str,
    source_identity: str,
    total_rows: int,
    sample_rows: int,
    seed: int,
) -> Tuple[str, np.dtype]:
    """Rebuild the exact sorted ``random.sample`` selection used by training."""
    index_dir = os.path.abspath(os.path.expanduser(index_dir))
    os.makedirs(index_dir, exist_ok=True)
    path, dtype = _active_index_spec(
        index_dir, source_identity, total_rows, sample_rows, seed
    )
    lock_path = f"{path}.lock"
    with open(lock_path, "a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        selected = random.Random(seed).sample(range(total_rows), sample_rows)
        selected.sort()
        compact = np.asarray(selected, dtype=dtype)
        temporary_path = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            with open(temporary_path, "wb", buffering=1024 * 1024) as handle:
                handle.write(compact.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    return path, dtype


class MappedIndexSequence:
    """Small pickle payload that reopens a compact integer mmap after spawn."""

    def __init__(self, path: str, dtype: np.dtype, length: Optional[int] = None):
        self.path = os.path.abspath(path)
        self.dtype = np.dtype(dtype)
        self._array: Optional[np.ndarray] = None
        size = os.path.getsize(self.path)
        if size % self.dtype.itemsize:
            raise ValueError(f"Corrupt compact index (invalid byte size): {self.path}")
        actual_length = size // self.dtype.itemsize
        if length is not None and length != actual_length:
            raise ValueError(
                f"Compact index length mismatch for {self.path}: "
                f"expected {length}, found {actual_length}"
            )
        self._length = actual_length

    def _open(self) -> None:
        if self._length:
            self._array = np.memmap(self.path, dtype=self.dtype, mode="r")
        else:
            self._array = np.empty(0, dtype=self.dtype)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> int:
        if self._array is None:
            self._open()
        assert self._array is not None
        return int(self._array[index])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._array = None


class IndexedJsonlReader:
    """Random-access JSONL reader backed by compact on-disk indexes."""

    def __init__(
        self,
        paths: Sequence[str],
        index_paths: Sequence[str],
        row_counts: Optional[Sequence[int]] = None,
    ):
        if isinstance(paths, (str, os.PathLike)):
            paths = [str(paths)]
        if isinstance(index_paths, (str, os.PathLike)):
            index_paths = [str(index_paths)]
        if len(paths) != len(index_paths):
            raise ValueError(
                f"paths and index_paths must have equal length: {len(paths)} != {len(index_paths)}"
            )
        self.paths = [_normalized_source_path(str(path)) for path in paths]
        self.index_paths = [os.path.abspath(str(path)) for path in index_paths]
        self._file_handles: Dict[Tuple[int, int, int], object] = {}
        self._offset_arrays: List[np.ndarray] = []
        if row_counts is not None:
            if len(row_counts) != len(paths):
                raise ValueError(
                    f"row_counts and paths must have equal length: "
                    f"{len(row_counts)} != {len(paths)}"
                )
            total = 0
            self._cumulative_lengths = []
            for row_count in row_counts:
                total += int(row_count)
                self._cumulative_lengths.append(total)
        else:
            self._cumulative_lengths = []
            self._open_indexes()

    def _open_indexes(self) -> None:
        self._offset_arrays = [load_offset_array(path) for path in self.index_paths]
        total = 0
        actual_cumulative_lengths = []
        for offsets in self._offset_arrays:
            total += len(offsets)
            actual_cumulative_lengths.append(total)
        if self._cumulative_lengths and self._cumulative_lengths != actual_cumulative_lengths:
            raise ValueError(
                f"JSONL index row counts changed for {self.index_paths}: "
                f"{self._cumulative_lengths} != {actual_cumulative_lengths}"
            )
        self._cumulative_lengths = actual_cumulative_lengths

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def _get_file_handle(self, file_idx: int):
        key = (os.getpid(), threading.get_ident(), file_idx)
        if key not in self._file_handles:
            self._file_handles[key] = open(self.paths[file_idx], "rb")
        return self._file_handles[key]

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        file_idx = bisect.bisect_right(self._cumulative_lengths, idx)
        previous_total = 0 if file_idx == 0 else self._cumulative_lengths[file_idx - 1]
        local_idx = idx - previous_total
        if not self._offset_arrays:
            self._open_indexes()
        offset = int(self._offset_arrays[file_idx][local_idx])
        file_handle = self._get_file_handle(file_idx)
        file_handle.seek(offset)
        return json.loads(file_handle.readline())

    def close(self) -> None:
        for file_handle in getattr(self, "_file_handles", {}).values():
            try:
                file_handle.close()
            except Exception:
                pass
        self._file_handles = {}

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handles"] = {}
        state["_offset_arrays"] = []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file_handles = {}
        self._offset_arrays = []
