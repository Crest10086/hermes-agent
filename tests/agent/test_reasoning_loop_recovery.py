"""Integration tests for the reasoning-loop recovery: A/B escalation + re-prompt.

Drives the recovery module with a fake agent to verify the full policy:
breaks 1..3 -> mode B (re-think, thinking on), break 4 -> mode A (thinking off),
break 5 -> exhausted (interrupt, no re-prompt); plus per-turn counter reset and
the config-disabled short-circuit.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import reasoning_loop_recovery as rec  # noqa: E402
from agent.reasoning_loop_watchdog import ReasoningLoopWatchdog  # noqa: E402


class FakeAgent:
    def __init__(self, session_id="sess-1"):
        self.session_id = session_id
        self._interrupt_requested = False
        self._ephemeral_reasoning_off = False
        self._reasoning_loop_breaks = 0
        self._reasoning_loop_recovery = None
        self.log_prefix = ""

    def _vprint(self, *a, **k):
        pass

    def _strip_think_blocks(self, s):
        return s

    def _persist_session(self, messages, history):
        pass

    def _has_pending_redirect(self):
        return False

    def clear_interrupt(self, preserve_redirect=False):
        return True


def _atomic(n: int) -> str:
    return "".join(chr(33 + ((i * 7) % 94)) for i in range(n))


def _loop_info():
    wd = ReasoningLoopWatchdog(check_every=1)
    seg = _atomic(100)
    info = wd.feed("sess-1", "pre " + seg * 3)
    assert info is not None
    return info


def _reset_cfg():
    rec._cached_cfg = None


def test_escalation_sequence_B_B_B_A_exhausted():
    _reset_cfg()
    agent = FakeAgent()
    info = _loop_info()

    # Break 1 -> B (re-think, thinking stays on)
    mode = rec.arm_recovery(agent, info)
    assert mode == "B"
    assert agent._interrupt_requested is True
    assert agent._reasoning_loop_breaks == 1
    assert agent._ephemeral_reasoning_off is False
    assert agent._reasoning_loop_recovery["mode"] == "B"

    # handle_in_interrupt re-prompts: collapsed reasoning + nudge, fallthrough.
    messages = []
    action = rec.handle_in_interrupt(agent, messages)
    assert action == "fallthrough"
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[0]["reasoning"] == info["collapsed"]
    assert messages[1]["role"] == "user"
    assert "loop" in messages[1]["content"].lower()
    assert agent._reasoning_loop_recovery is None  # consumed

    # Breaks 2 and 3 -> still B
    for expected in (2, 3):
        mode = rec.arm_recovery(agent, info)
        assert mode == "B"
        assert agent._reasoning_loop_breaks == expected
        assert agent._ephemeral_reasoning_off is False

    # Break 4 -> A (thinking off, force answer)
    mode = rec.arm_recovery(agent, info)
    assert mode == "A"
    assert agent._reasoning_loop_breaks == 4
    assert agent._ephemeral_reasoning_off is True
    assert agent._reasoning_loop_recovery["mode"] == "A"

    # mode-A re-prompt nudges for the final answer
    msgs_a = []
    assert rec.handle_in_interrupt(agent, msgs_a) == "fallthrough"
    assert "final answer" in msgs_a[1]["content"].lower()

    # Break 5 -> exhausted: interrupt set, but no recovery armed
    mode = rec.arm_recovery(agent, info)
    assert mode is None
    assert agent._interrupt_requested is True
    assert agent._reasoning_loop_recovery is None
    assert rec.handle_in_interrupt(agent, []) is None  # defers to normal interrupt


def test_reset_for_turn_clears_counter():
    _reset_cfg()
    agent = FakeAgent()
    agent._reasoning_loop_breaks = 4
    rec.reset_for_turn(agent)
    assert agent._reasoning_loop_breaks == 0


def test_generation_start_resets_buffer_not_counter():
    _reset_cfg()
    agent = FakeAgent()
    agent._reasoning_loop_breaks = 2
    seg = _atomic(100)
    rec.generation_start(agent)
    # Buffer reset: feeding a fresh non-loop paragraph does not fire.
    assert rec.feed_reasoning(agent, "totally fresh reasoning text here ") is None
    # Counter persists across generations (re-think streams).
    assert agent._reasoning_loop_breaks == 2


def test_disabled_by_config_short_circuits():
    rec._cached_cfg = {"enabled": False}
    try:
        agent = FakeAgent()
        assert rec.is_enabled() is False
        assert rec.feed_reasoning(agent, _atomic(100) * 3) is None
    finally:
        _reset_cfg()