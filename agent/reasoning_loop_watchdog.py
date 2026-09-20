"""Reasoning-loop watchdog.

Detects a dead loop in a model's streaming reasoning (the thinking block),
collapses it to a single copy, and hands a clean version to the next call so
the loop does not re-seed itself into the context.

Definition (per the user): a dead loop is the thinking chain being generated
repeatedly and EXACTLY — a single changed character disqualifies it. Detection
is therefore exact-match periodicity, not fuzzy similarity: a long segment
(>= ``min_segment_len``) repeated back-to-back (>= ``min_repeats``) byte-for-byte.
This is maximally conservative — no false positives from near-matches on xhigh
reasoning that merely revisits phrasing.

This module is pure / self-contained (no core imports) so it is immune to
update conflicts and trivial to rebase. The thin core integration (stream-hook
feed + interrupt + re-prompt) lives elsewhere and stays small.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

DEFAULT_MIN_SEGMENT_LEN = 400
DEFAULT_MIN_REPEATS = 3
DEFAULT_MAX_BUFFER = 4000
DEFAULT_CHECK_EVERY = 200

LOOP_MARKER = "\n[reasoning loop collapsed: {n} repeated segments removed]"


def find_loop_segment(
    text: str,
    *,
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
    min_repeats: int = DEFAULT_MIN_REPEATS,
) -> Optional[Tuple[str, int]]:
    """Return ``(segment, repeat_count)`` if the TAIL of ``text`` is an EXACT
    back-to-back repetition of a segment of length >= ``min_segment_len``, else
    None.

    Models the dead-loop shape: the trailing region is byte-for-byte periodic
    with period ``p`` (``p >= min_segment_len``) over ``min_repeats`` copies.
    Scans ``p`` small-to-large and returns the tightest (smallest) exact period.
    A single changed character anywhere in the repeated region disqualifies it.
    """
    n = len(text)
    if n < min_segment_len * min_repeats:
        return None
    max_p = n // min_repeats
    if max_p < min_segment_len:
        return None
    for p in range(min_segment_len, max_p + 1):
        # Cheap necessary check: the last two p-copies must be identical.
        if text[n - 2 * p : n - p] != text[n - p : n]:
            continue
        # Full check: the entire trailing region is periodic with period p.
        if text[n - min_repeats * p : n - p] == text[n - min_repeats * p + p : n]:
            return text[n - p :], min_repeats
    return None


def is_reasoning_loop(
    text: str,
    *,
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
    min_repeats: int = DEFAULT_MIN_REPEATS,
) -> bool:
    """True when ``text`` ends in an exact reasoning loop."""
    return find_loop_segment(
        text, min_segment_len=min_segment_len, min_repeats=min_repeats
    ) is not None


def collapse_reasoning(
    text: str,
    *,
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
    min_repeats: int = DEFAULT_MIN_REPEATS,
) -> Tuple[str, int]:
    """Collapse a detected loop to ``prefix + one copy of the segment + marker``.

    Returns ``(collapsed, repeat_count)``. When no loop is detected, returns
    ``(text, 0)`` unchanged. The collapsed result is deliberately no longer a
    loop, so replaying it into the next call does not re-seed the repetition.
    """
    found = find_loop_segment(
        text, min_segment_len=min_segment_len, min_repeats=min_repeats
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
    """Per-session streaming-reasoning loop detector (exact-match).

    ``feed(session_id, delta)`` accumulates reasoning deltas and, once enough
    new text has arrived since the last check, runs the detector. Returns a
    loop-info dict on a confirmed loop (only once per generation), else None.
    ``reset(session_id)`` clears state for a new generation — call it on stream
    start.

    The loop-info dict carries: ``session_id``, ``segment``, ``repeat_count``,
    ``collapsed`` (cleaned reasoning), and ``raw_tail`` (the buffered tail).
    """

    def __init__(
        self,
        *,
        min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
        min_repeats: int = DEFAULT_MIN_REPEATS,
        max_buffer: int = DEFAULT_MAX_BUFFER,
        check_every: int = DEFAULT_CHECK_EVERY,
    ):
        self.min_segment_len = min_segment_len
        self.min_repeats = min_repeats
        self.max_buffer = max_buffer
        self.check_every = check_every
        self._states: Dict[str, _SessionState] = {}
        self._lock = threading.Lock()

    def reset(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._states.pop(session_id, None)

    def feed(self, session_id: str, delta: str) -> Optional[dict]:
        if not session_id or not delta:
            return None
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
            )
            if not found:
                return None
            st.fired = True
            segment, count = found
            collapsed, _ = collapse_reasoning(
                st.buffer,
                min_segment_len=self.min_segment_len,
                min_repeats=self.min_repeats,
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
    "DEFAULT_MAX_BUFFER",
    "DEFAULT_CHECK_EVERY",
    "LOOP_MARKER",
]