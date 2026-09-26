import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.dashboard_bridge import (
    BusyError, ConflictError, DashboardBridge, DashboardStore, PREFIX,
    dashboard_http, guarded_gateway_message,
)
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.wake import WakeNotAccepted, adapter_supports_push

A = "a" * 64
B = "b" * 64
KEY = "local-test-key-never-production"


@pytest.fixture
def transport(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY}))
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[Platform.API_SERVER] = adapter
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    bridge = DashboardBridge(adapter, tmp_path / "chat.db")
    adapter._dashboard_bridge = runner._dashboard_bridge = bridge
    yield adapter, runner, bridge
    bridge.store.db.close()


async def settle(bridge):
    await asyncio.gather(*tuple(bridge.tasks))
    await asyncio.sleep(0)


def test_presentation_isolation_and_restart(tmp_path):
    path = tmp_path / "messages.db"
    store = DashboardStore(path)
    store.admit(A, "id-a", "private A")
    store.admit(B, "id-b", "private B")
    message = store.add(A, "**bold**")
    assert not store.edit(B, message, "replace")
    assert not store.delete(B, message)
    assert [row["content"] for row in store.snapshot(B)["messages"]] == ["private B"]
    store.db.close()
    restored = DashboardStore(path)
    assert restored.snapshot(A)["request"]["status"] == "interrupted"
    assert restored.snapshot(A)["messages"][-1]["content"] == "**bold**"
    restored.db.close()


@pytest.mark.asyncio
async def test_atomic_admission_idempotency_and_release(transport):
    adapter, runner, bridge = transport
    release = asyncio.Event()
    async def handler(event):
        await release.wait()
        return "done"
    adapter.set_message_handler(handler)
    first = bridge.submit(A, "request-first-0001", "hello")
    assert bridge.submit(A, "request-first-0001", "hello")["id"] == first["id"]
    with pytest.raises(ConflictError):
        bridge.submit(A, "request-first-0001", "different")
    with pytest.raises(BusyError):
        bridge.submit(B, "request-second-001", "other")
    assert bridge.snapshot(A)["ownBusy"]
    assert bridge.snapshot(B)["busy"] and not bridge.snapshot(B)["ownBusy"]
    release.set()
    await settle(bridge)
    assert not bridge.snapshot(B)["busy"]
    assert bridge.snapshot(A)["request"]["status"] == "completed"
    assert [row["content"] for row in bridge.snapshot(A)["messages"]] == ["hello", "done"]


@pytest.mark.asyncio
async def test_telegram_lease_blocks_dashboard_and_preserves_control(transport):
    adapter, runner, bridge = transport
    started, release = asyncio.Event(), asyncio.Event()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test-private")
    event = MessageEvent(text="hello", source=source)
    async def invoke(event):
        started.set()
        await release.wait()
        return "done"
    task = asyncio.create_task(guarded_gateway_message(runner, event, invoke))
    await started.wait()
    assert bridge.snapshot(A)["busy"] and not bridge.snapshot(A)["ownBusy"]
    with pytest.raises(BusyError):
        bridge.submit(A, "request-test-00001", "hello")
    async def control(event):
        return "approved"
    assert await guarded_gateway_message(runner, MessageEvent(text="/approve", source=source), control) == "approved"
    release.set()
    assert await task == "done"
    assert not bridge.snapshot(A)["busy"]


@pytest.mark.asyncio
async def test_cancel_before_first_step_does_not_stick_busy(transport):
    _, _, bridge = transport
    bridge.submit(A, "cancel-before-00001", "hello")
    task = next(iter(bridge.tasks))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert not bridge.snapshot(A)["busy"]
    assert bridge.snapshot(A)["request"]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_native_commands_and_model_global_scope(transport):
    _, runner, bridge = transport
    bridge.submit(A, "native-whoami-00001", "/whoami")
    await settle(bridge)
    snapshot = bridge.snapshot(A)
    assert snapshot["request"]["status"] == "completed"
    assert len(snapshot["messages"]) >= 2
    assert "api_server" in snapshot["messages"][-1]["content"].lower()
    bridge.submit(A, "native-model-000001", "/model anything --global")
    await settle(bridge)
    assert "--global" in bridge.snapshot(A)["messages"][-1]["content"]
    assert not runner._session_model_overrides
    bridge.submit(B, "native-resume-00001", "/resume " + bridge.store.sessions(A)[0])
    await settle(bridge)
    assert "حسابك فقط" in bridge.snapshot(B)["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_dashboard_async_delivery_and_busy_internal_retry(transport):
    adapter, _, bridge = transport
    dashboard_source = SessionSource(platform=Platform.API_SERVER, chat_id=PREFIX + A)
    ordinary_api_source = SessionSource(platform=Platform.API_SERVER, chat_id="ordinary-api-session")
    assert adapter_supports_push(adapter, dashboard_source)
    assert not adapter_supports_push(adapter, ordinary_api_source)
    token = bridge.admission.claim(("telegram", "another-chat"))
    try:
        event = MessageEvent(text="background completion", source=dashboard_source, internal=True)
        with pytest.raises(WakeNotAccepted):
            await adapter.handle_message(event)
        assert not event._gateway_accepted
    finally:
        bridge.admission.release(token)


@pytest.mark.asyncio
async def test_startup_recovery_never_replays_dashboard_tools(transport):
    _, runner, bridge = transport
    source = SessionSource(platform=Platform.API_SERVER, chat_id=PREFIX + A, user_id=PREFIX + A)
    entry = await runner.async_session_store.get_or_create_session(source)
    entry.origin = source
    entry.resume_pending = True
    entry.resume_reason = next(iter(runner._AUTO_RESUME_REASONS))
    assert not await runner._mark_durable_active_turn(MessageEvent(text="hello", source=source), entry.session_key)
    assert not runner._resume_pending_candidates()
    bridge.adapter._inflight_agent_runs = 1
    assert bridge.snapshot(A)["busy"]
    with pytest.raises(BusyError):
        bridge.submit(A, "existing-api-run-001", "hello")
    bridge.adapter._inflight_agent_runs = 0


@pytest.mark.asyncio
async def test_storage_failure_still_releases_the_global_lease(transport):
    adapter, _, bridge = transport
    release = asyncio.Event()
    async def handler(event):
        await release.wait()
        return "final"
    adapter.set_message_handler(handler)
    bridge.submit(A, "disk-failure-000001", "hello")
    await asyncio.sleep(0)
    bridge.store.db.execute("PRAGMA query_only=ON")
    release.set()
    await asyncio.gather(*tuple(bridge.tasks), return_exceptions=True)
    await asyncio.sleep(0)
    assert not bridge.snapshot(A)["busy"]
    assert bridge.snapshot(A)["request"]["status"] == "failed"
    bridge.store.db.execute("PRAGMA query_only=OFF")


@pytest.mark.asyncio
async def test_http_auth_disconnect_message_boundaries_and_reconnect(transport):
    adapter, _, bridge = transport
    release = asyncio.Event()
    async def handler(event):
        await adapter.send(event.source.chat_id, "short first")
        await adapter.send(event.source.chat_id, "short second")
        await release.wait()
        return "final **answer**"
    adapter.set_message_handler(handler)
    async def endpoint(request):
        return await dashboard_http(adapter, request)
    app = web.Application()
    app.router.add_get('/api/dashboard/{account}', endpoint)
    app.router.add_post('/api/dashboard/{account}', endpoint)
    app.router.add_get('/api/dashboard/{account}/events', endpoint)
    async with TestClient(TestServer(app)) as client:
        assert (await client.get(f'/api/dashboard/{A}')).status == 401
        headers = {"Authorization": f"Bearer {KEY}"}
        assert (await client.get('/api/dashboard/invalid', headers=headers)).status == 400
        response = await client.post(f'/api/dashboard/{A}', headers=headers, json={"content": "hello", "requestId": "http-request-00001"})
        assert response.status == 202
        stream = await client.get(f'/api/dashboard/{A}/events', headers=headers)
        assert (await stream.content.readline()).startswith(b'event: snapshot')
        stream.close()
        # Disconnect a subscriber; the accepted agent job remains active.
        assert bridge.tasks and bridge.snapshot(A)["busy"]
        release.set()
        await settle(bridge)
        restored = await (await client.get(f'/api/dashboard/{A}', headers=headers)).json()
        assert [row["content"] for row in restored["messages"]] == ["hello", "short first", "short second", "final **answer**"]
        other = await (await client.get(f'/api/dashboard/{B}', headers=headers)).json()
        assert other["messages"] == []
        duplicate = await client.post(f'/api/dashboard/{A}', headers=headers, json={"content": "hello", "requestId": "http-request-00001"})
        assert duplicate.status == 202
        assert len((await duplicate.json())["messages"]) == 4
