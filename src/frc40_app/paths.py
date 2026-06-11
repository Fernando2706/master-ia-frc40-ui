"""Centralized filesystem layout for the FRC40 desktop app.

Resolves where the application should store its data (training runs, saved
models, logs) in a way that is independent of the executable's location.

Goals
-----
* On Windows, use ``%LOCALAPPDATA%\\FRC40\\Quimicos`` by default.
* On macOS, use ``~/Library/Application Support/FRC40/Quimicos``.
* On Linux, use ``$XDG_DATA_HOME/FRC40/Quimicos`` (falling back to
  ``~/.local/share/FRC40/Quimicos``).
* Fall back to a folder under the user's home directory if the platform's
  preferred location is unavailable (read-only filesystem, missing env var...).
* Stay backwards compatible with installations that still carry data next to
  the executable (``<exe_dir>/app_outputs``).

The functions below are pure (no side effects beyond creating the directory
when explicitly asked) so the UI can call them from the main thread or from
background workers without surprises.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "FRC40"
APP_SUBDIR_NAME = "Quimicos"
LEGACY_DIR_NAME = "app_outputs"


def _windows_default_data_dir() -> Path | None:
    """Return ``%LOCALAPPDATA%\\FRC40\\Quimicos`` if it can be resolved."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_DIR_NAME / APP_SUBDIR_NAME
    # Roaming AppData is a reasonable secondary option.
    roaming = os.environ.get("APPDATA")
    if roaming:
        return Path(roaming) / APP_DIR_NAME / APP_SUBDIR_NAME
    return None


def _macos_default_data_dir() -> Path | None:
    return Path.home() / "Library" / "Application Support" / APP_DIR_NAME / APP_SUBDIR_NAME


def _linux_default_data_dir() -> Path | None:
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / APP_DIR_NAME / APP_SUBDIR_NAME
    return Path.home() / ".local" / "share" / APP_DIR_NAME / APP_SUBDIR_NAME


def _home_fallback_data_dir() -> Path:
    """Last-resort fallback that always works: a folder under the user's home."""
    return Path.home() / f"{APP_DIR_NAME}_{APP_SUBDIR_NAME}"


def get_default_data_dir(create: bool = False) -> Path:
    """Return the platform-specific default data directory.

    The function never raises: if no platform-specific location is available
    it returns a folder under the user's home directory. The directory is
    only created on disk when ``create=True``.
    """
    candidates: list[Path | None]
    if sys.platform.startswith("win"):
        candidates = [_windows_default_data_dir(), _home_fallback_data_dir()]
    elif sys.platform == "darwin":
        candidates = [_macos_default_data_dir(), _home_fallback_data_dir()]
    else:
        candidates = [_linux_default_data_dir(), _home_fallback_data_dir()]

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if create:
                candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            # Permission denied, read-only volume, etc. Try the next one.
            continue

    # This should be unreachable because the home fallback always exists,
    # but returning a sensible default keeps the type checker happy.
    fallback = _home_fallback_data_dir()
    if create:
        fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_legacy_data_dir() -> Path | None:
    """Return the legacy data directory next to the executable, if it exists.

    The legacy layout stored everything under ``<project>/app_outputs`` and,
    for frozen builds, that folder lived next to the ``.exe``. We still want
    to discover those folders so that existing users keep their references.
    """
    # The project layout is `<root>/app/app.py` and `<root>/app/src/frc40_app/`.
    # `parents[2]` is the project root, but when frozen, the executable lives
    # directly in the project's parent and the `app/` subfolder does not exist.
    try:
        here = Path(__file__).resolve()
        for ancestor in here.parents:
            legacy = ancestor / LEGACY_DIR_NAME
            if legacy.is_dir():
                return legacy
            if (ancestor / "app").is_dir() and (ancestor / "app" / "app.py").is_file():
                project_root = ancestor
                break
        else:
            project_root = None
        if project_root is not None:
            legacy = project_root / LEGACY_DIR_NAME
            if legacy.is_dir():
                return legacy
    except OSError:
        return None
    return None
