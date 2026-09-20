"""Reasoning-loop watchdog.

Detects a dead loop in a model's streaming reasoning (the thinking block),
collapses it to a single copy, and hands a clean version to the next call so
the loop does not re-seed itself into the context.

Conservative by design: only a LONG segment (>= ``min_segment_len``) repeated
back-to-back (>= ``min_repeats``) with high similarity trips it. Legitimate
long reasoning that revisits ideas with intervening new content should NOT
trip it — the requirement is 宁可漏报早期, 不误杀 xhigh 的合法长推理.

This module is pure / self-contained (no core imports) so it is immune to
update conflicts and trivial to rebase. The thin core integration (stream-hook
feed + interrupt + re-prompt) lives elsewhere and stays small.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

DEFAULT_MIN_SEGMENT_LEN = 400
DEFAULT_MIN_REPEATS = 3
DEFAULT_SIMILARITY = 0.97
DEFAULT_MAX_BUFFER = 4000
DEFAULT_CHECK_EVERY = 200

LOOP_MARKER = "\n[reasoning loop collapsed: {n} repeated segments removed]"


def find_loop_segment(
    text: str,
    *,
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    similarity: float = DEFAULT_SIMILARITY,
) -> Optional[Tuple[str, int]]:
    """Return ``(segment, repeat_count)`` if the TAIL of ``text`` is back-to-back
    near-verbatim repetition of a segment of length >= ``min_segment_len``, else
    None.

    Models the dead-loop shape: the trailing region is periodic with period ``p``
    (``p >= min_segment_len``) at >= ``similarity`` over ``min_repeats`` copies.
    Scans ``p`` small-to-large and returns the tightest qualifying period. The
    positional periodicity comparison (``text[i]`` vs ``text[i+p]``) is tolerant
    to a few characters of drift, unlike exact non-overlapping-copy matching.
    """
    n = len(text)
    if n < min_segment_len * min_repeats:
        return None
    max_p = n // min_repeats
    if max_p < min_segment_len:
        return None
    for p in range(min_segment_len, max_p + 1):
        base = text[n - p:]
        # Cheap pre-screen: a 30-char anchor of the base must recur earlier in
        # the buffer (before the last copy) before we pay for the scan.
        anchor = base[:30]
        if anchor not in text[: n - p]:
            continue
        region_lo = n - min_repeats * p
        hi = n - p
        if hi <= region_lo:
            continue
        a = text[region_lo:hi]
        b = text[region_lo + p:hi + p]
        match = sum(1 for x, y in zip(a, b) if x == y)
        if match / len(a) >= similarity:
            return base, min_repeats
    return None


def is_reasoning_loop(
    text: str,
    *,
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    similarity: float = DEFAULT_SIMILARITY,
) -> bool:
    """True when ``text`` ends in a detected reasoning loop."""
    return find_loop_segment(
        text,
        min_segment_len=min_segment_len,
        min_repeats=min_repeats,
        similarity=similarity,
    ) is not None


def collapse_reasoning(
    text: str,
    *,
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    similarity: float = DEFAULT_SIMILARITY,
) -> Tuple[str, int]:
    """Collapse a detected loop to ``prefix + one copy of the segment + marker``.

    Returns ``(collapsed, repeat_count)``. When no loop is detected, returns
    ``(text, 0)`` unchanged. The collapsed result is deliberately no longer a
    loop, so replaying it into the next call does not re-seed the repetition.
    """
    found = find_loop_segment(
        text,
        min_segment_len=min_segment_len,
        min_repeats=min_repeats,
        similarity=similarity,
    )
    if not found:
        return text, 0
    segment, count = found
    seg_len = len(segment)
    n = len(text)
    prefix_end = n - count * seg_len
    prefix = text[:prefix_end]
    collapsed = prefix + segment + LOOP_MARKER.format(n=count)
    return collapsed, count


@dataclass
class _SessionState:
    buffer: str = ""
    last_checked_len: int = 0
    fired: bool = False


class ReasoningLoopWatchdog:
    """Per-session streaming-reasoning loop detector.

    ``feed(session_id, delta)`` accumulates reasoning deltas and, once enough
    new text has arrived since the last check, runs the conservative detector.
    It returns a loop-info dict on a confirmed loop (only once per generation),
    else None. ``reset(session_id)`` clears state for a new generation — call it
    on stream start.

    The loop-info dict carries: ``session_id``, ``segment``, ``repeat_count``,
    ``collapsed`` (cleaned reasoning), and ``raw_tail`` (the buffered tail).
    """

    def __init__(
        self,
        *,
        min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
        min_repeats: int = DEFAULT_MIN_REPEATS,
        similarity: float = DEFAULT_SIMILARITY,
        max_buffer: int = DEFAULT_MAX_BUFFER,
        check_every: int = DEFAULT_CHECK_EVERY,
    ):
        self.min_segment_len = min_segment_len
        self.min_repeats = min_repeats
        self.similarity = similarity
        self.max_buffer = max_buffer
        self.check_every = check_every
        self._states: Dict[str, _SessionState] = {}
        self._lock = threading.Lock()

    def reset(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._states.pop(session_id, None)

    def feed(
        self, session_id: str, delta: str, now: Optional[float] = None
    ) -> Optional[dict]:
        if not session_id or not delta:
            return None
        del now  # reserved for future throttling; kept for call-site stability
        with self._lock:
            st = self._states.get(session_id)
            if st is None:
                st = _SessionState()
                self._states[session_id] = st
            st.buffer += delta
            if len(st.buffer) > self.max_buffer:
                st.buffer = st.buffer[-self.max_buffer:]
            if len(st.buffer) - st.last_checked_len < self.check_every:
                return None
            st.last_checked_len = len(st.buffer)
            if st.fired:
                return None
            found = find_loop_segment(
                st.buffer,
                min_segment_len=self.min_segment_len,
                min_repeats=self.min_repeats,
                similarity=self.similarity,
            )
            if not found:
                return None
            st.fired = True
            segment, count = found
            collapsed, _ = collapse_reasoning(
                st.buffer,
                min_segment_len=self.min_segment_len,
                min_repeats=self.min_repeats,
                similarity=self.similarity,
            )
            return {
                "session_id": session_id,
                "segment": segment,
                "repeat_count": count,
                "collapsed": collapsed,
                "raw_tail": st.buffer,
            }


__all__ = [
    "ReasoningLoopWatchdog",
    "find_loop_segment",
    "is_reasoning_loop",
    "collapse_reasoning",
    "DEFAULT_MIN_SEGMENT_LEN",
    "DEFAULT_MIN_REPEATS",
    "DEFAULT_SIMILARITY",
    "DEFAULT_MAX_BUFFER",
    "DEFAULT_CHECK_EVERY",
    "LOOP_MARKER",
]