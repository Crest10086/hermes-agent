"""Tests for the reasoning-loop watchdog: detector, collapse, state machine."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reasoning_loop_watchdog import (  # noqa: E402
    ReasoningLoopWatchdog,
    collapse_reasoning,
    find_loop_segment,
    is_reasoning_loop,
)


_TOPIC = [
    "inventory", "migration", "transaction", "rollback", "locking",
    "indexing", "caching", "quota", "ledger", "audit",
]


def _unique_paragraph(seed: int = 0, chars: int = 550) -> str:
    # Distinct reasoning content: topic words rotate per clause and are offset
    # by seed, so adjacent paragraphs share little text. This models legitimate
    # reasoning that covers different facets — which must NOT trip the detector.
    parts = []
    i = seed
    while sum(len(p) for p in parts) < chars:
        t = _TOPIC[(seed + i) % len(_TOPIC)]
        parts.append(
            f"consider {t} behavior case {i} using wording gamma{i} "
            f"delta{seed} epsilon{i} zeta{seed} for this distinct facet. "
        )
        i += 1
    return "".join(parts)


def test_detects_back_to_back_loop():
    seg = _unique_paragraph()
    assert len(seg) >= 400
    text = "Initial analysis before the loop begins. " + seg + seg + seg
    found = find_loop_segment(text)
    assert found is not None
    _segment, count = found
    assert count == 3
    assert is_reasoning_loop(text)


def test_no_loop_on_distinct_content():
    text = "".join(_unique_paragraph(seed=k) for k in range(5))
    assert find_loop_segment(text) is None
    assert not is_reasoning_loop(text)


def test_short_repeats_do_not_trip():
    # Smallest period (~200) is below min_segment_len (400) -> no loop.
    seg = _unique_paragraph(chars=200)
    text = seg * 4
    assert find_loop_segment(text) is None


def test_insufficient_length_does_not_trip():
    seg = _unique_paragraph()
    text = "some intro. " + seg + seg  # only two copies, need three
    assert find_loop_segment(text) is None


def test_one_char_changed_is_not_a_loop():
    # The user's definition: completely identical repetition. Flip a single
    # character in the middle copy -> no longer an exact loop.
    seg = _unique_paragraph()
    idx = len(seg) // 2
    flipped = ("X" if seg[idx] != "X" else "Y")
    corrupted = seg[:idx] + flipped + seg[idx + 1 :]
    text = "PFX " + seg + corrupted + seg
    assert find_loop_segment(text) is None
    assert not is_reasoning_loop(text)


def test_collapse_removes_the_loop():
    seg = _unique_paragraph()
    text = "PFX " + seg + seg + seg
    collapsed, count = collapse_reasoning(text)
    assert count == 3
    assert len(collapsed) < len(text)
    assert "collapsed" in collapsed
    # Behaviour contract: the collapsed result is no longer a loop.
    assert not is_reasoning_loop(collapsed)


def test_collapse_no_loop_returns_unchanged():
    text = "".join(_unique_paragraph(seed=k) for k in range(3))
    collapsed, count = collapse_reasoning(text)
    assert count == 0
    assert collapsed == text


def test_watchdog_fires_once_then_resets():
    wd = ReasoningLoopWatchdog(check_every=1)  # check on every feed
    seg = _unique_paragraph()
    full = "intro. " + seg + seg + seg
    sid = "s1"
    fired = None
    for i in range(0, len(full), 50):
        r = wd.feed(sid, full[i:i + 50])
        if r is not None:
            fired = r
            break
    assert fired is not None
    assert fired["repeat_count"] == 3
    assert not is_reasoning_loop(fired["collapsed"])
    # Further feeds in the same generation do not re-fire.
    assert wd.feed(sid, "more") is None
    # After reset, a fresh loop can fire again.
    wd.reset(sid)
    fired2 = None
    for i in range(0, len(full), 50):
        r = wd.feed(sid, full[i:i + 50])
        if r is not None:
            fired2 = r
            break
    assert fired2 is not None


def test_watchdog_no_fire_on_non_loop():
    wd = ReasoningLoopWatchdog(check_every=1)
    text = "".join(_unique_paragraph(seed=k) for k in range(6))
    for i in range(0, len(text), 50):
        assert wd.feed("s2", text[i:i + 50]) is None