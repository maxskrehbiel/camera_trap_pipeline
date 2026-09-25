"""Resume machinery: stage flags in ``status.json`` and append-only per-item checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Generic, TypeVar, cast

from .atomic_io import atomic_write_bytes, atomic_write_json

RecordT = TypeVar("RecordT", bound=Mapping[str, object])


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class StageRow:
    """One line of the status table.

    Attributes:
        stage: Stage name.
        state: ``done``, ``incomplete`` (started but not finished) or ``pending``.
        info: Counts and timestamps the stage recorded.
    """

    stage: str
    state: str
    info: dict[str, Any]


class RunStatus:
    """Stage completion flags persisted to a JSON file after every change.

    A stage is marked started before it runs and done after its outputs are written, so a
    crash leaves it visibly incomplete and the next run starts it again.
    """

    def __init__(self, path: Path, stages: Sequence[str]) -> None:
        """Load the status file if it exists.

        Args:
            path: Location of ``status.json``.
            stages: Stage names in execution order.
        """
        self.path = path
        self.stages = tuple(stages)
        stored: dict[str, Any] = {}
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
        self._stages: dict[str, dict[str, Any]] = dict(stored.get("stages", {}))

    def info(self, stage: str) -> dict[str, Any]:
        """Return what a stage recorded.

        Args:
            stage: Stage name.

        Returns:
            A copy of the stage's entry, or an empty dict if it never ran.
        """
        return dict(self._stages.get(stage, {}))

    def is_done(self, stage: str) -> bool:
        """Tell whether a stage finished and has not been invalidated since.

        Args:
            stage: Stage name.

        Returns:
            True when the stage is marked done.
        """
        return bool(self._stages.get(stage, {}).get("done", False))

    def mark_started(self, stage: str) -> None:
        """Record that a stage began.

        Args:
            stage: Stage name.
        """
        self._check(stage)
        self._stages[stage] = {"done": False, "started_at": _now()}
        self.save()

    def mark_done(self, stage: str, **info: Any) -> None:
        """Record that a stage finished, with any counts worth keeping.

        Args:
            stage: Stage name.
            **info: JSON-serializable facts about the run, such as row counts.
        """
        self._check(stage)
        started = self._stages.get(stage, {}).get("started_at")
        self._stages[stage] = {"done": True, "started_at": started, "finished_at": _now(), **info}
        self.save()

    def invalidate_from(self, stage: str) -> list[str]:
        """Clear the flags of a stage and every stage after it.

        Args:
            stage: First stage to clear.

        Returns:
            The stages whose flags were removed.
        """
        self._check(stage)
        cleared = [s for s in self.stages[self.stages.index(stage) :] if s in self._stages]
        for name in cleared:
            del self._stages[name]
        self.save()
        return cleared

    def rows(self) -> list[StageRow]:
        """Summarize every stage.

        Returns:
            One row per stage, in execution order.
        """
        rows = []
        for stage in self.stages:
            info = self._stages.get(stage)
            if info is None:
                state = "pending"
            elif info.get("done"):
                state = "done"
            else:
                state = "incomplete"
            rows.append(StageRow(stage=stage, state=state, info=dict(info or {})))
        return rows

    def save(self) -> None:
        """Write the status file atomically."""
        atomic_write_json(self.path, {"stages": self._stages})

    def _check(self, stage: str) -> None:
        if stage not in self.stages:
            raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(self.stages)}")


class JsonlCheckpoint(Generic[RecordT]):
    """Append-only JSON Lines file of records, each identified by a key.

    Each record is flushed as soon as it is written. If the process dies mid-write, the
    torn last line is dropped (and the file rewritten without it) the next time the
    checkpoint is opened, so appends always start on a clean line.
    """

    def __init__(self, path: Path, key: Callable[[RecordT], str]) -> None:
        """Open a checkpoint, loading and repairing any existing file.

        Args:
            path: Location of the ``.jsonl`` file.
            key: Returns a record's identifier; a later record replaces an earlier one.
        """
        self.path = path
        self.key = key
        self.records: dict[str, RecordT] = {}
        self.dropped_lines = 0
        self._handle: IO[str] | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        kept: list[bytes] = []
        for line in raw.split(b"\n"):
            if not line.strip():
                continue
            try:
                record = cast(RecordT, json.loads(line))
                item_key = self.key(record)
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
                self.dropped_lines += 1
                continue
            kept.append(line)
            self.records[item_key] = record
        if self.dropped_lines or (raw and not raw.endswith(b"\n")):
            atomic_write_bytes(self.path, b"".join(line + b"\n" for line in kept))

    def __contains__(self, item_key: object) -> bool:
        return item_key in self.records

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[RecordT]:
        return iter(self.records.values())

    def get(self, item_key: str) -> RecordT | None:
        """Look up a record.

        Args:
            item_key: Record identifier.

        Returns:
            The stored record, or None.
        """
        return self.records.get(item_key)

    def needs(self, item_key: str) -> bool:
        """Tell whether an item still has to be processed.

        Records carrying an ``error`` field are retried on the next run, so a transient
        failure (a file still being copied, for example) does not stick.

        Args:
            item_key: Record identifier.

        Returns:
            True when there is no record yet or the last attempt failed.
        """
        record = self.records.get(item_key)
        return record is None or "error" in record

    def append(self, record: RecordT, flush: bool = True) -> None:
        """Append one record, by default flushing it to disk straight away.

        Args:
            record: JSON-serializable record.
            flush: Set False when appending several records that belong together, then
                call ``flush`` once.
        """
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.records[self.key(record)] = record
        if flush:
            self.flush()

    def flush(self) -> None:
        """Push buffered records to the operating system."""
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        """Close the file handle if one is open."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> JsonlCheckpoint[RecordT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
