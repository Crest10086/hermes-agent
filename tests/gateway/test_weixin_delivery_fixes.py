import asyncio

from gateway.config import PlatformConfig
from gateway.platforms import weixin as wx
from gateway.platforms.weixin import WeixinAdapter

MODEL_BLOCK = "**OpenAI**: `gpt-4o`, `gpt-4o-mini`\n" \
              "**Anthropic**: `claude-3-5-sonnet`\n" \
              "**Local**: `qwen3.8-27b`\n" \
              "**DeepSeek**: `deepseek-v4-flash`\n" \
              "**Google**: `gemini-2.0-flash`"

CHAT_BLOCK = "好的\n收到\n谢谢"


def _make_adapter(**extra):
    return WeixinAdapter(PlatformConfig(enabled=True, token="***", extra=extra))


# ---- Item B: local circuit open_seconds 30 -> 90 + configurable/coerce ----

def test_circuit_open_seconds_defaults_to_90():
    adapter = _make_adapter()
    assert adapter._rate_limit_circuit_open_seconds == 90.0


def test_circuit_open_seconds_extra_override():
    adapter = _make_adapter(rate_limit_circuit_open_seconds=45)
    assert adapter._rate_limit_circuit_open_seconds == 45.0


def test_circuit_open_seconds_env_override(monkeypatch):
    monkeypatch.setenv("WEIXIN_RATE_LIMIT_CIRCUIT_OPEN_SECONDS", "120")
    adapter = _make_adapter()
    assert adapter._rate_limit_circuit_open_seconds == 120.0


def test_circuit_config_coerces_bad_values_to_default():
    # A non-numeric value must not crash __init__ — falls back to 90s.
    adapter = _make_adapter(rate_limit_circuit_open_seconds="not-a-number")
    assert adapter._rate_limit_circuit_open_seconds == 90.0


# ---- Item C: per-user_id 5s send gate ----

def test_wait_for_user_slot_waits_until_interval_elapsed(monkeypatch):
    adapter = _make_adapter()
    adapter._send_user_min_interval_seconds = 5.0
    clock = {"t": 1000.0}
    monkeypatch.setattr(wx.time, "monotonic", lambda: clock["t"])
    sleeps = []

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(wx.asyncio, "sleep", _fake_sleep)

    # Last send was 1s ago (t=1000, last=999): must wait 4s to reach 5s.
    adapter._user_last_send["user-1"] = 999.0
    asyncio.run(adapter._wait_for_user_slot("user-1"))
    assert sleeps == [4.0]

    # Last send was 10s ago: no wait.
    sleeps.clear()
    adapter._user_last_send["user-1"] = 990.0
    asyncio.run(adapter._wait_for_user_slot("user-1"))
    assert sleeps == []


def test_wait_for_user_slot_zero_interval_never_sleeps(monkeypatch):
    adapter = _make_adapter()
    adapter._send_user_min_interval_seconds = 0.0
    clock = {"t": 1000.0}
    monkeypatch.setattr(wx.time, "monotonic", lambda: clock["t"])
    sleeps = []

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(wx.asyncio, "sleep", _fake_sleep)
    adapter._user_last_send["user-1"] = 1000.0
    asyncio.run(adapter._wait_for_user_slot("user-1"))
    assert sleeps == []


def test_send_gate_is_per_user_not_global(monkeypatch):
    # Two different users must get DISTINCT locks, so one user's send never
    # blocks the other (requirement C: no bare sleep blocking other users).
    adapter = _make_adapter()
    g1 = adapter._send_gates.setdefault("user-1", asyncio.Lock())
    g2 = adapter._send_gates.setdefault("user-2", asyncio.Lock())
    assert g1 is not g2
    assert adapter._send_gates["user-1"] is g1


# ---- Item E: /model list must not split into chatty bubbles ----

def test_model_list_ships_as_single_chunk():
    chunks = wx._split_text_for_weixin_delivery(MODEL_BLOCK, 2000)
    assert chunks == [MODEL_BLOCK]


def test_chat_block_still_splits_per_line():
    # Short plain chat lines (no markdown) keep the legacy bubble split.
    chunks = wx._split_text_for_weixin_delivery(CHAT_BLOCK, 2000)
    assert chunks == ["好的", "收到", "谢谢"]


def test_markdown_line_votes_block_non_chatty():
    # Even one markdown line makes the whole block ship as one message.
    block = "收到\n```content```\n好的"
    assert wx._should_split_short_chat_block_for_weixin(block) is False


def test_looks_like_chatty_line_rejects_markdown_and_backticks():
    assert wx._looks_like_chatty_line_for_weixin("**OpenAI**: `gpt-4o`") is False
    assert wx._looks_like_chatty_line_for_weixin("`qwen3.8-27b`") is False
    assert wx._looks_like_chatty_line_for_weixin("好的") is True
    assert wx._looks_like_chatty_line_for_weixin("谢谢") is True
