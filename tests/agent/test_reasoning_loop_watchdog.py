"""Tests for the reasoning-loop watchdog: detector, collapse, state machine."""
import random
import sys
import time
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


def _atomic(n: int) -> str:
    # Exactly n chars with no exact period shorter than n (multiplicative hash
    # over printable ASCII keeps positions distinct for n <= 94).
    return "".join(chr(33 + ((i * 7) % 94)) for i in range(n))


_TOPIC = [
    "inventory", "migration", "transaction", "rollback", "locking",
    "indexing", "caching", "quota", "ledger", "audit",
]


def _unique_paragraph(seed: int = 0, chars: int = 550) -> str:
    # Distinct reasoning content: topic words rotate per clause and are offset
    # by seed, so adjacent paragraphs share little text. Models legitimate
    # reasoning covering different facets — must NOT trip the detector.
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


# --- Tier boundary tests --------------------------------------------------

def test_tier1_20char_x5_detected():
    seg = _atomic(20)
    text = "pre " + seg * 5
    found = find_loop_segment(text)
    assert found is not None
    segment, count = found
    assert len(segment) == 20
    assert count >= 5


def test_tier1_20char_x4_not_detected():
    seg = _atomic(20)
    text = "pre " + seg * 4  # four copies, tier 1 needs five
    assert find_loop_segment(text) is None


def test_tier2_100char_x3_detected():
    seg = _atomic(100)
    text = "pre " + seg * 3
    found = find_loop_segment(text)
    assert found is not None
    segment, count = found
    assert len(segment) == 100
    assert count >= 3


def test_tier2_100char_x2_not_detected():
    seg = _atomic(100)
    text = "pre " + seg * 2  # two copies, tier 2 needs three
    assert find_loop_segment(text) is None


def test_below_20char_not_detected():
    seg = _atomic(15)  # 15-char unit, below the 20-char floor
    text = "pre " + seg * 8
    assert find_loop_segment(text) is None


def _randomish(n: int, seed: int = 20260921) -> str:
    # n-char string with (astronomically unlikely) no exact period — for
    # large-scale tests where _atomic's 94-cycle would introduce a sub-period.
    rnd = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rnd.choice(alphabet) for _ in range(n))


def test_tiny_scale_20char_x6_detected():
    # Extremely small scale: a 20-char unit repeated 6x (more than the tier-1
    # minimum of 5). Actual-count must report 6, and collapse keeps one copy.
    seg = _atomic(20)
    text = "pre " + seg * 6
    found = find_loop_segment(text)
    assert found is not None
    segment, count = found
    assert len(segment) == 20
    assert count == 6
    collapsed, cc = collapse_reasoning(text)
    assert cc == 6
    assert collapsed.count(seg) == 1


def test_large_scale_1000char_x3_detected():
    # Extremely large length: a 1000-char unit repeated 3x (3000 chars total).
    # Verifies the detector finds the large period and stays fast enough for a
    # background thread.
    seg = _randomish(1000)
    text = "pre " + seg * 3
    t0 = time.perf_counter()
    found = find_loop_segment(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert found is not None
    segment, count = found
    assert len(segment) == 1000
    assert count == 3
    assert elapsed_ms < 250, f"detection too slow: {elapsed_ms:.1f}ms"


# --- Realistic long-reasoning loop ----------------------------------------

def test_detects_back_to_back_loop():
    seg = _unique_paragraph()
    assert len(seg) >= 100
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


def test_insufficient_length_does_not_trip():
    seg = _unique_paragraph()
    text = "some intro. " + seg + seg  # only two copies, tier 2 needs three
    assert find_loop_segment(text) is None


def test_one_char_changed_is_not_a_loop():
    # The user's definition: completely identical repetition. Flip a single
    # character in the middle copy -> no longer an exact loop.
    seg = _unique_paragraph()
    idx = len(seg) // 2
    flipped = ("X" if seg[idx] != "X" else "Y")
    corrupted = seg[:idx] + flipped + seg[idx + 1:]
    text = "PFX " + seg + corrupted + seg
    assert find_loop_segment(text) is None
    assert not is_reasoning_loop(text)


# --- Collapse -------------------------------------------------------------

def test_collapse_removes_the_loop():
    seg = _unique_paragraph()
    text = "PFX " + seg + seg + seg
    collapsed, count = collapse_reasoning(text)
    assert count == 3
    assert len(collapsed) < len(text)
    assert "collapsed" in collapsed
    # Behaviour contract: the collapsed result is no longer a loop.
    assert not is_reasoning_loop(collapsed)


def test_collapse_keeps_exactly_one_copy():
    seg = _atomic(100)
    text = "PFX " + seg * 6  # six copies
    collapsed, count = collapse_reasoning(text)
    assert count == 6
    # Exactly one copy of the atomic unit remains, plus the prefix and marker.
    assert collapsed.count(seg) == 1
    assert collapsed.startswith("PFX ")


def test_collapse_no_loop_returns_unchanged():
    text = "".join(_unique_paragraph(seed=k) for k in range(3))
    collapsed, count = collapse_reasoning(text)
    assert count == 0
    assert collapsed == text


# --- State machine ---------------------------------------------------------

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