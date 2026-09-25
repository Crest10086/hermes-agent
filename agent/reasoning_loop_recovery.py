"""Reasoning-loop recovery — core-side wiring.

Turns a detected reasoning loop (from :mod:`agent.reasoning_loop_watchdog`)
into an early interrupt plus a clean re-prompt, so a dead-looping thinking
block does not burn the whole budget and does not re-seed itself into the
context.

Policy (adaptive A/B escalation, per the user):
  * breaks 1..N (default N=3) -> mode B: keep thinking (xhigh) but on the
    cleaned (collapsed) context — re-think.
  * break N+1                 -> mode A: turn thinking off (``_ephemeral_reasoning_off``)
    and force the final answer.
  * break N+2 and beyond      -> exhausted: just end the turn.

The escalation counter lives on the agent (``_reasoning_loop_breaks``) so it
persists across the re-think generations within a turn and is reset once per
turn (see :func:`reset_for_turn`).

Pure glue on top of the watchdog — no heavy core imports, so it stays
rebase-friendly and is immune to update conflicts on the detection logic.
"""
from __future__ import annotations

import threading
from typing import Optional

from agent.reasoning_loop_watchdog import (
    DEFAULT_CHECK_EVERY,
    DEFAULT_MAX_BUFFER,
    DEFAULT_TIERS,
    ReasoningLoopWatchdog,
)

_MAX_B_BREAKS_DEFAULT = 3

_lock = threading.Lock()
_watchdog: Optional[ReasoningLoopWatchdog] = None
_cached_cfg: Optional[dict] = None


def _cfg() -> dict:
    """The ``reasoning_loop`` config section (cached), falling back to defaults.

    Shape::

        reasoning_loop:
          enabled: true
          tiers: [[20, 5], [100, 3]]   # (min_segment_len, min_repeats)
          max_buffer: 4000
          check_every: 200
          max_b_breaks: 3
    """
    global _cached_cfg
    if _cached_cfg is not None:
        return _cached_cfg
    section = {}
    try:
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("reasoning_loop", {}) or {}
    except Exception:
        section = {}
    _cached_cfg = section
    return section


def is_enabled() -> bool:
    return bool(_cfg().get("enabled", True))


def _get_watchdog() -> ReasoningLoopWatchdog:
    global _watchdog
    with _lock:
        if _watchdog is None:
            cfg = _cfg()
            tiers = tuple(tuple(t) for t in (cfg.get("tiers") or DEFAULT_TIERS))
            _watchdog = ReasoningLoopWatchdog(
                tiers=tiers,
                max_buffer=int(cfg.get("max_buffer", DEFAULT_MAX_BUFFER)),
                check_every=int(cfg.get("check_every", DEFAULT_CHECK_EVERY)),
            )
    return _watchdog


def _sid(agent) -> str:
    return getattr(agent, "session_id", "") or ""


def generation_start(agent) -> None:
    """Reset the detection buffer for a fresh generation (called at stream start).

    Deliberately does NOT touch the escalation counter — that persists across
    re-think generations and is reset per turn via :func:`reset_for_turn`.
    """
    if not is_enabled():
        return
    _get_watchdog().reset(_sid(agent))


def feed_reasoning(agent, delta: str) -> Optional[dict]:
    """Feed a reasoning delta; returns a loop-info dict on a confirmed loop."""
    if not is_enabled() or not delta:
        return None
    return _get_watchdog().feed(_sid(agent), delta)


def reset_for_turn(agent) -> None:
    """Reset the escalation counter at the start of a turn."""
    try:
        agent._reasoning_loop_breaks = 0
    except Exception:
        pass


def arm_recovery(agent, loop_info: dict) -> Optional[str]:
    """Decide mode A/B from the escalation counter, set the interrupt + thinking
    flags, and store the recovery state on the agent.

    Returns the mode (``"A"``/``"B"``) on success, or ``None`` when exhausted
    (the caller should just let the interrupt end the turn).
    """
    max_b = int(_cfg().get("max_b_breaks", _MAX_B_BREAKS_DEFAULT))
    breaks = int(getattr(agent, "_reasoning_loop_breaks", 0) or 0)
    if breaks < max_b:
        mode = "B"
        agent._reasoning_loop_breaks = breaks + 1
    elif breaks == max_b:
        mode = "A"
        agent._reasoning_loop_breaks = breaks + 1
    else:
        mode = None  # exhausted: interrupt but no re-prompt
        agent._reasoning_loop_breaks = breaks + 1
    agent._interrupt_requested = True  # always break the stream early
    if mode is not None:
        agent._ephemeral_reasoning_off = (mode == "A")
        agent._reasoning_loop_recovery = {
            "collapsed": loop_info["collapsed"],
            "mode": mode,
            "repeat_count": loop_info.get("repeat_count", 0),
        }
    else:
        agent._reasoning_loop_recovery = None
    # Observability: one greppable line per break. The recovery branch in
    # handle_api_interrupt returns before the generic "⚡ Interrupted" print,
    # so without this a loop event would leave no trace in the gateway log.
    try:
        seg = loop_info.get("segment", "") or ""
        note = {
            "B": "re-thinking on collapsed context (thinking stays on)",
            "A": "disabling thinking to force the final answer",
            None: "loop persists; ending the turn",
        }[mode]
        agent._vprint(
            f"{agent.log_prefix}🔁 [reasoning-loop] break {breaks + 1} "
            f"(mode {mode or 'exhausted'}): {len(seg)}-char segment repeated "
            f"{loop_info.get('repeat_count', '?')}x exactly -> {note}",
            force=True,
        )
    except Exception:
        pass
    return mode


_NUDGE_B = (
    "Your reasoning fell into an exact loop and has been collapsed to a single "
    "copy. Keep thinking and reach a conclusion."
)
_NUDGE_A = (
    "Your reasoning fell into an exact loop and has been collapsed to a single "
    "copy. Stop deliberating and give the final answer now."
)


def handle_in_interrupt(agent, messages: list) -> Optional[str]:
    """Called from ``handle_api_interrupt``. If a recovery is armed, append the
    collapsed reasoning (as the assistant row) plus a nudge user row and return
    ``"fallthrough"`` so the retry loop re-enters for a clean re-prompt.

    Returns ``None`` to defer to the normal interrupt handling (no recovery
    armed, e.g. a plain user escape).
    """
    rec = getattr(agent, "_reasoning_loop_recovery", None)
    if not rec:
        return None
    mode = rec.get("mode", "B")
    messages.append(
        {"role": "assistant", "content": "", "reasoning": rec.get("collapsed", "")}
    )
    messages.append(
        {"role": "user", "content": _NUDGE_A if mode == "A" else _NUDGE_B}
    )
    # Consume so a later interrupt in the same turn doesn't re-fire this.
    agent._reasoning_loop_recovery = None
    if mode == "A":
        agent._ephemeral_reasoning_off = True
    return "fallthrough"


__all__ = [
    "is_enabled",
    "generation_start",
    "feed_reasoning",
    "reset_for_turn",
    "arm_recovery",
    "handle_in_interrupt",
]