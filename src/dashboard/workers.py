"""Background-thread bridge for expensive full-history scans (e.g. the
Universe Over Time tab's basket-count-over-time scan). Tk is not thread-safe:
results/progress from the worker thread are placed on a queue.Queue and
drained from the Tk main loop via root.after(), never touched directly from
the worker thread.

Only one job runs "in front" at a time -- starting a new job supersedes
(cancels) whichever job was previously running, and any of its stale queue
messages are dropped on arrival rather than applied to the UI."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class _JobHandle:
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self.cancel_event.set()

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()


class BackgroundJobRunner:
    def __init__(self, root):
        self._root = root
        self._queue: "queue.Queue[tuple[str, _JobHandle, Any]]" = queue.Queue()
        self._current_handle: _JobHandle | None = None
        self._on_progress: Callable[[int, int], None] | None = None
        self._on_done: Callable[[Any], None] | None = None
        self._on_error: Callable[[BaseException], None] | None = None
        self._poll_scheduled = False

    def start(
        self,
        work: Callable[[Callable[[int, int], None], Callable[[], bool]], Any],
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        if self._current_handle is not None:
            self._current_handle.cancel()

        handle = _JobHandle()
        self._current_handle = handle
        self._on_progress = on_progress
        self._on_done = on_done
        self._on_error = on_error

        def _progress(done: int, total: int) -> None:
            self._queue.put(("progress", handle, (done, total)))

        def _cancel_check() -> bool:
            return handle.is_cancelled()

        def _run() -> None:
            try:
                result = work(_progress, _cancel_check)
                if not handle.is_cancelled():
                    self._queue.put(("done", handle, result))
            except BaseException as exc:  # noqa: BLE001 -- surfaced to the UI, not swallowed
                if not handle.is_cancelled():
                    self._queue.put(("error", handle, exc))

        threading.Thread(target=_run, daemon=True).start()
        self._ensure_polling()

    def _ensure_polling(self) -> None:
        if self._poll_scheduled:
            return
        self._poll_scheduled = True

        def _poll() -> None:
            try:
                while True:
                    kind, handle, payload = self._queue.get_nowait()
                    if handle is not self._current_handle:
                        continue  # stale job (superseded by a newer one), drop silently
                    if kind == "progress" and self._on_progress:
                        self._on_progress(*payload)
                    elif kind == "done" and self._on_done:
                        self._on_done(payload)
                    elif kind == "error" and self._on_error:
                        self._on_error(payload)
            except queue.Empty:
                pass
            self._root.after(100, _poll)

        self._root.after(100, _poll)
