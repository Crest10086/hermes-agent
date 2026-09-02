import asyncio

from gateway.config import Platform
from gateway.kanban_watchers import _fileify_long_notice
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb

# A real research-completion summary, extended past 1200 chars so the
# file-ification threshold triggers (mirrors the t_82ffe3e3 ~1500-char notice).
LONG_SUMMARY = (
    "调研完成：Windows/WSL2 vLLM 跑 Qwen3.8-27B (hybrid GatedDeltaNet) 的经验与坑，报告落盘 "
    "E:\\hermes-mem\\researcher\\vllm-windows-qwen38-27b-recon.md（36KB，含 Q1-Q7 全覆盖 + "
    "TL;DR 5 件事到参数级 + 启动参数片段 + 差异表 + 存疑清单 + 分层信源）。核心增量："
    "(1) WSL≥2.7.0 + mask nvidia-cdi-refresh + sleep45 是单卡 Blackwell CUDA graph 稳定的总开关"
    "（microsoft/WSL#14452，同卡 RTX5090 修好 full graph ~140tok/s）；"
    "(2) SM120 必设 FLASHINFER_CUDA_ARCH_LIST=12.0f 止血 illegal-instruction；"
    "(3) MTP 接受率 eager 59.5% / cudagraph 67-100%，num_speculative_tokens=2 保留；"
    "(4) nvfp4_4over6 在 sm_120 全部 5 backend 不 support kv_cache_dtype，KV 量化走 fp8。"
    "结论：今晚用 WSL2 + cudagraph 试 4K/16K，eager 保 249K，cudagraph 保 158K。"
    "补充：Windows native 下 flashinfer JIT 需 nvcc + CUDA_HOME，vLLM≥0.24 已移除 GGUF；"
    "hybrid Mamba 的 KV 只来自 16 层 full attention，故 KV 量化收益有限。"
    "信源：vllm-project/vllm issue、HuggingFace 讨论区、Reddit r/LocalLLaMA、知乎/B站；"
    "每条结论均带日期与可信度标注，存疑项单独列清单。"
    "分项：Q1 平台选型（WSL2 vs native）Q2 hybrid 架构在 vLLM 的成熟度 Q3 显存与上下文取舍 "
    "Q4 KV 量化 Q5 MTP 投机调优 Q6 Blackwell attention backend 选择 Q7 启动参数片段汇总。"
    "差异/修正表（相对既有认知）：① \"PIECEWISE 崩溃\"实为 max_model_len=262144 时 KV OOM"
    "（需 9.13GiB > 可用 8.75GiB），非 graph 兼容 bug；② flashinfer attention JIT 需要 nvcc"
    "（已装 cuda-nvcc-13.0 + CUDA_HOME=/usr/local/cuda-13.0）；③ vLLM≥0.24 已移除 GGUF 支持"
    "（ConfigFormat 仅 auto/hf/mistral）。下一步建议：按 TL;DR 优先级实测，先 WSL2 + cudagraph"
    "跑 4K/16K 各 3 次取中位数，再比较 eager 249K 下 decode 6-8 tok/s 是否可接受；"
    "若不可接受则回退 cudagraph 158K。存疑待验证项单列在报告末节，含 CUDA graph/inductor 编译"
    "工作区再压缩的可能性与 multi-user 并发下的稳定表现。"
)


class FailingAdapter:
    """Records sends and raises a chosen error on every text send."""

    def __init__(self, exc):
        self.sent = []
        self.docs = []
        self.exc = exc

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        raise self.exc

    async def send_document(self, chat_id, file_path, metadata=None, **kwargs):
        self.docs.append(file_path)
        return None

    async def handle_message(self, event):
        pass


class RecordingAdapter(FailingAdapter):
    """Like FailingAdapter but a successful text send (no raise)."""

    def __init__(self):
        super().__init__(exc=None)

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


async def _drive_notifier(monkeypatch, runner, done, max_iters=1000):
    """Run the real notifier loop continuously until ``done()`` is True, then
    cancel it. The continuous loop is what actually retries a rewound event —
    a fresh watcher call per tick would not re-claim it (see the rewind path)."""
    real_sleep = asyncio.sleep

    async def fast_sleep(delay):
        if delay == 5:
            return None  # skip the boot delay but KEEP the loop running
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    task = asyncio.create_task(runner._kanban_notifier_watcher(interval=0))
    iters = 0
    while not done() and not task.done() and iters < max_iters:
        await real_sleep(0.01)
        iters += 1
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Let any in-flight to_thread (rewind/unsub) land before we assert on the DB.
    await real_sleep(0.2)


def _create_completed_sub(db_path, monkeypatch, summary):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _sub_count(conn, tid):
    return len(kb.list_notify_subs(conn, task_id=tid))


# ---- Item A: rate-limited failures must NOT drop the subscription ----

def test_rate_limited_failures_do_not_drop_subscription(tmp_path, monkeypatch):
    tid = _create_completed_sub(tmp_path / "rate-limit.db", monkeypatch, summary="done")
    adapter = FailingAdapter(
        RuntimeError(
            "adapter send() reported failure: iLink sendmessage rate limited; "
            "cooldown active for 30.0s"
        )
    )
    runner = _make_runner(adapter)

    def done():
        # Stop once the loop has clearly tried >MAX_SEND_FAILURES times and
        # the subscription is still alive (rate-limited never counts down).
        return len(adapter.sent) >= 13

    asyncio.run(_drive_notifier(monkeypatch, runner, done))
    conn = kb.connect()
    try:
        assert _sub_count(conn, tid) == 1, "rate-limited sub must not be dropped"
        assert len(adapter.sent) >= 13
    finally:
        conn.close()


def test_permanent_failures_drop_subscription(tmp_path, monkeypatch):
    tid = _create_completed_sub(tmp_path / "permanent.db", monkeypatch, summary="done")
    adapter = FailingAdapter(RuntimeError("adapter send() reported failure: chat not found"))
    runner = _make_runner(adapter)

    def done():
        c = kb.connect()
        try:
            return len(kb.list_notify_subs(c, task_id=tid)) == 0
        finally:
            c.close()

    asyncio.run(_drive_notifier(monkeypatch, runner, done))
    conn = kb.connect()
    try:
        assert _sub_count(conn, tid) == 0, "permanent-failure sub must drop after 12"
        assert len(adapter.sent) >= 12
    finally:
        conn.close()


# ---- Item D: long notification -> short text + .md document ----

def test_fileify_long_notice_writes_document_matching_original(tmp_path):
    assert len(LONG_SUMMARY) > 1200
    send_text, path = _fileify_long_notice(
        LONG_SUMMARY,
        "t_82ffe3e3",
        1788357673.0,
        threshold=1200,
        cache_dir=str(tmp_path),
    )
    assert path is not None
    assert path.endswith(".md")
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == LONG_SUMMARY
    assert send_text.startswith(LONG_SUMMARY[:500])
    assert "…完整报告见附件" in send_text
    assert len(send_text) < len(LONG_SUMMARY)


def test_fileify_long_notice_keeps_short_text_unchanged(tmp_path):
    send_text, path = _fileify_long_notice(
        "short done", "t_x", 123.0, threshold=1200, cache_dir=str(tmp_path)
    )
    assert send_text == "short done"
    assert path is None


def test_long_notice_delivered_as_short_text_plus_document(tmp_path, monkeypatch):
    tid = _create_completed_sub(tmp_path / "long-notice.db", monkeypatch, summary=LONG_SUMMARY)
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    def done():
        return len(adapter.docs) >= 1

    asyncio.run(_drive_notifier(monkeypatch, runner, done))
    assert len(adapter.sent) == 1, "one short text notification"
    assert "…完整报告见附件" in adapter.sent[0]["text"]
    assert len(adapter.docs) == 1, "one .md document attachment"
    with open(adapter.docs[0], encoding="utf-8") as fh:
        assert fh.read() == LONG_SUMMARY
