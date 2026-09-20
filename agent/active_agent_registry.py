"""Thread-safe registry of the currently-active AIAgent per session_id.

Plugin stream-hook callbacks run on their own dispatcher threads, so a
contextvar set on the streaming thread is not visible to them. This registry
lets an out-of-band observer (e.g. the reasoning-loop watchdog) reach the live
agent for a session and set flags such as ``_interrupt_requested``.

Thin on purpose: register at the start of the streaming call, unregister at the
end. Kept additive and small to minimize update-conflict surface.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_agents: Dict[str, Any] = {}


def register(session_id: str, agent: Any) -> None:
    if not session_id:
        return
    with _lock:
        _agents[session_id] = agent


def unregister(session_id: str) -> None:
    if not session_id:
        return
    with _lock:
        _agents.pop(session_id, None)


def get(session_id: str) -> Optional[Any]:
    if not session_id:
        return None
    with _lock:
        return _agents.get(session_id)


__all__ = ["register", "unregister", "get"]