"""Reasoning-loop watchdog.

Detects a dead loop in a model's streaming reasoning (the thinking block),
collapses it to a single copy, and hands a clean version to the next call so
the loop does not re-seed itself into the context.

Definition (per the user): a dead loop is the thinking chain being generated
repeatedly and EXACTLY — a single changed character disqualifies it. Detection
is therefore exact-match periodicity, not fuzzy similarity. Two tiers, either
of which qualifies:

  * a segment of >= 20 chars repeated back-to-back >= 5 times, OR
  * a segment of >= 100 chars repeated back-to-back >= 3 times.

The tightest (smallest) qualifying period wins, so the collapse keeps the
atomic loop unit (20 or 100 chars worth), not a bloated block that already
contains several sub-loops. This is maximally conservative — no false positives
from near-matches on xhigh reasoning that merely revisits phrasing.

This module is pure / self-contained (no core imports) so it is immune to
update conflicts and trivial to rebase. The thin core integration (stream-hook
feed + interrupt + re-prompt) lives elsewhere and stays small.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# (min_segment_len, min_repeats) tiers. Sorted ascending by min_segment_len.
DEFAULT_TIERS: Tuple[Tuple[int, int], ...] = ((20, 5), (100, 3))
DEFAULT_MAX_BUFFER = 4000
DEFAULT_CHECK_EVERY = 200

LOOP_MARKER = "\n[reasoning loop collapsed: {n} repeated segments removed]"


def find_loop_segment(
    text: str,
    *,
    tiers: Tuple[Tuple[int, int], ...] = DEFAULT_TIERS,
) -> Optional[Tuple[str, int]]:
    """Return ``(segment, repeat_count)`` if the TAIL of ``text`` is an EXACT
    back-to-back repetition matching any tier, else None.

    For each tier ``(min_len, min_repeats)`` the trailing region must be
    byte-for-byte periodic with period ``p >= min_len`` over at least
    ``min_repeats`` copies. The smallest qualifying ``p`` across all tiers wins
    (the atomic loop unit). ``repeat_count`` is the ACTUAL number of trailing
    copies (>= the tier minimum), so the collapse keeps exactly one copy.
    A single changed character anywhere in the repeated region disqualifies it.
    """
    n = len(text)
    best_p: Optional[int] = None
    for min_len, min_repeats in tiers:
        if n < min_len * min_repeats:
            continue
        max_p = n // min_repeats
        if max_p < min_len:
            continue
        for p in range(min_len, max_p + 1):
            base = text[n - p:]
            # Cheap necessary check: the immediately-preceding p-block is
            # identical to the last one (find is C-speed, no full-region alloc).
            if text.find(base, n - 2 * p, n - p) != n - 2 * p:
                continue
            # Full check: the trailing region is exactly min_repeats copies.
            if text[n - min_repeats * p : n] == base * min_repeats:
                if best_p is None or p < best_p:
                    best_p = p
                break  # smallest p for this tier
    if best_p is None:
        return None
    base = text[n - best_p:]
    # Count the ACTUAL number of trailing copies (may exceed the tier minimum).
    count = 1
    pos = n - best_p
    while pos - best_p >= 0 and text[pos - best_p : pos] == base:
        count += 1
        pos -= best_p
    return base, count


def is_reasoning_loop(
    text: str,
    *,
    tiers: Tuple[Tuple[int, int], ...] = DEFAULT_TIERS,
) -> bool:
    """True when ``text`` ends in an exact reasoning loop."""
    return find_loop_segment(text, tiers=tiers) is not None


def collapse_reasoning(
    text: str,
    *,
    tiers: Tuple[Tuple[int, int], ...] = DEFAULT_TIERS,
) -> Tuple[str, int]:
    """Collapse a detected loop to ``prefix + one copy of the segment + marker``.

    Returns ``(collapsed, repeat_count)``. When no loop is detected, returns
    ``(text, 0)`` unchanged. Because ``repeat_count`` is the actual number of
    trailing copies, the prefix excludes every copy and exactly one copy is
    retained — so replaying the result into the next call does not re-seed the
    repetition.
    """
    found = find_loop_segment(text, tiers=tiers)
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
    """Per-session streaming-reasoning loop detector (exact-match, tiered).

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
        tiers: Tuple[Tuple[int, int], ...] = DEFAULT_TIERS,
        max_buffer: int = DEFAULT_MAX_BUFFER,
        check_every: int = DEFAULT_CHECK_EVERY,
    ):
        self.tiers = tiers
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
            found = find_loop_segment(st.buffer, tiers=self.tiers)
            if not found:
                return None
            st.fired = True
            segment, count = found
            collapsed, _ = collapse_reasoning(st.buffer, tiers=self.tiers)
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
    "DEFAULT_TIERS",
    "DEFAULT_MAX_BUFFER",
    "DEFAULT_CHECK_EVERY",
    "LOOP_MARKER",
]