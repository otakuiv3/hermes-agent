"""Local regressions: real memory files and transport, never model requests."""
from types import SimpleNamespace

import pytest

from agent.agent_init import _parse_compression_config, _init_memory
from agent.chat_request_budget import dashboard_request_budget
from gateway.dashboard_bridge import DashboardStore, PREFIX
from gateway.dashboard_notices import dashboard_notice
from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource
from tools.memory_scope import bind_memory_namespace, memory_namespace
from tools.memory_tool import MemoryStore

A, B = "a" * 64, "b" * 64


def test_memory_two_homes_and_accounts_roundtrip(tmp_path, monkeypatch):
    stores = []
    for home_name, account in (("home-a", A), ("home-b", B), ("home-a", A)):
        home = tmp_path / home_name
        monkeypatch.setenv("HERMES_HOME", str(home))
        (home / "memories").mkdir(parents=True, exist_ok=True)
        (home / "memories" / "USER.md").write_text("legacy personal profile", encoding="utf-8")
        (home / "memories" / "MEMORY.md").write_text("legacy private notes", encoding="utf-8")
        (home / "memories" / "SHARED.md").write_text("General timezone: Baghdad", encoding="utf-8")
        token = bind_memory_namespace(account)
        try:
            store = MemoryStore()
            store.load_from_disk()
            prompt = (store.format_for_system_prompt("memory") or "") + (store.format_for_system_prompt("user") or "")
            assert "legacy" not in prompt
            assert "General timezone" in prompt
            assert ("private A" in prompt) == (len(stores) == 2)
            assert store.add("memory", f"private {account[0].upper()}")["success"]
            assert (store.format_for_system_prompt("memory") or "") in prompt  # frozen prefix
            stores.append(store)
        finally:
            memory_namespace.reset(token)
    # A cached A store must remain bound to home A/account A under home B/context B.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home-b"))
    token = bind_memory_namespace(B)
    try:
        assert stores[0].add("user", "A prefers short replies")["success"]
        other = tmp_path / "home-b" / "dashboard-memories" / B / "USER.md"
        assert not other.exists() or not other.read_text(encoding="utf-8")
    finally:
        memory_namespace.reset(token)
    assert "A prefers" in (tmp_path / "home-a" / "dashboard-memories" / A / "USER.md").read_text(encoding="utf-8")
    native = MemoryStore()
    native.load_from_disk()
    assert "legacy personal" in native.format_for_system_prompt("user")


@pytest.mark.parametrize("value", ["../outside", "", "a" * 63, "A" * 64])
def test_namespace_rejects_untrusted_values(value):
    with pytest.raises(ValueError):
        bind_memory_namespace(value)
    assert memory_namespace.get() is None


def test_admins_in_same_home_have_independent_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for account, expected in ((A, ""), (B, ""), (A, "private A")):
        token = bind_memory_namespace(account)
        try:
            store = MemoryStore()
            store.load_from_disk()
            prompt = store.format_for_system_prompt("memory") or ""
            assert ("private A" in prompt) == bool(expected)
            assert "private B" not in prompt
            assert store.add("memory", f"private {account[0].upper()}")["success"]
        finally:
            memory_namespace.reset(token)


def test_oversized_shared_block_does_not_disable_private_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "SHARED.md").write_text("x" * 2201, encoding="utf-8")
    token = bind_memory_namespace(A)
    try:
        store = MemoryStore()
        store.load_from_disk()
        assert store.add("memory", "private knowledge")["success"]
        restored = MemoryStore()
        restored.load_from_disk()
        assert "private knowledge" in restored.format_for_system_prompt("memory")
        assert "xxx" not in restored.format_for_system_prompt("memory")
    finally:
        memory_namespace.reset(token)


def test_scoped_compression_bounded_without_mutating_global():
    agent = SimpleNamespace(model="m", provider="openrouter", api_mode="chat_completions", quiet_mode=True)
    config = {"compression": {"max_attempts": 3, "threshold": .85}}
    token = bind_memory_namespace(A)
    try:
        scoped = _parse_compression_config(agent, config)
        assert scoped.max_attempts == 1
        assert scoped.threshold <= .5
        assert scoped.proactive_prune_tokens == 10000
    finally:
        memory_namespace.reset(token)
    assert _parse_compression_config(agent, config).max_attempts == 3
    assert config["compression"]["threshold"] == .85


def test_dashboard_does_not_reuse_external_global_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace(enabled_toolsets=["memory"], disabled_toolsets=[], tools=[])
    token = bind_memory_namespace(A)
    try:
        _init_memory(agent, {}, False, "api_server", memory_manager=object())
        assert agent._memory_store is not None
        assert agent._memory_manager is None
    finally:
        memory_namespace.reset(token)


def test_explicit_budget_bound_to_endpoint_and_model():
    config = {"dashboard": {"request_token_limits": {"api.groq.com": {"chosen": 6000}}}}
    assert dashboard_request_budget(config, "chosen", "https://api.groq.com/openai/v1") == 6000
    assert dashboard_request_budget(config, "other", "https://api.groq.com/openai/v1") is None
    assert dashboard_request_budget(config, "chosen", "https://other.example/v1") is None
    assert dashboard_request_budget(config, "chosen", "https://[") is None
    for invalid in (True, 0, "6000", -1):
        config["dashboard"]["request_token_limits"]["api.groq.com"]["chosen"] = invalid
        assert dashboard_request_budget(config, "chosen", "https://api.groq.com") is None


def test_runtime_allowlist_isolated_and_persistent(tmp_path):
    path = tmp_path / "chat.db"
    store = DashboardStore(path)
    store.record_runtime(A, {"model": "openai/gpt-oss-120b", "provider": "custom", "last_prompt_tokens": 5000,
                             "context_length": 131072, "tool_count": 38, "api_key": "secret", "messages": "private"})
    store.record_runtime(B, {"model": "https://secret.example", "tool_count": True})
    assert not store.snapshot(B)["runtime"]
    store.db.close()
    store = DashboardStore(path)
    assert store.snapshot(A)["runtime"]["tool_count"] == 38
    assert "secret" not in str(store.snapshot(A)["runtime"])
    store.db.close()


def test_notices_preserve_real_answers_and_final_errors():
    assert dashboard_notice("📬 No home channel is set for Api_Server. hello")[1] == "skip"
    assert dashboard_notice("🗜️ Compacting context — summarizing")[1] == "activity"
    assert dashboard_notice("⚠️  Request payload too large (413) — compression attempt 1/3")[1] == "activity"
    for text in ("**Hello**", "This conversation has grown too large", "HTTP 413 final error"):
        assert dashboard_notice(text) == (text, "message")


@pytest.mark.asyncio
async def test_exhaustion_preserves_dashboard_session():
    runner = GatewayTurnMixin()
    entry = SimpleNamespace(session_id="existing")
    source = SessionSource(platform=Platform.API_SERVER, chat_id=PREFIX + A)
    response, same_entry = await runner._hmwa_compression_exhaustion_reset(
        {"compression_exhausted": True}, "old error", entry, "key", source)
    assert same_entry is entry
    assert "/new" in response and "/compress" in response
    assert "Session auto-reset" not in response
