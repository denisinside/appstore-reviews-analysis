"""Optional persistence and job dispatch hooks for hosted execution.

Local Uvicorn and CLI use the ordinary filesystem and FastAPI background tasks.
The Modal entrypoint installs Volume and worker hooks in its own containers.
"""

from __future__ import annotations

from collections.abc import Callable


def _noop() -> None:
    pass


reload: Callable[[], None] = _noop
commit: Callable[[], None] = _noop
dispatch_analysis: Callable[[str, dict], None] | None = None


def configure(
    *,
    reload_volume: Callable[[], None] = _noop,
    commit_volume: Callable[[], None] = _noop,
    dispatch: Callable[[str, dict], None] | None = None,
) -> None:
    global reload, commit, dispatch_analysis
    reload = reload_volume
    commit = commit_volume
    dispatch_analysis = dispatch
