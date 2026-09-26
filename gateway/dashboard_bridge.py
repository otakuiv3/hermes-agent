"""Authenticated dashboard transport over the gateway's normal messaging pipeline.

No browser owns a turn. SQLite stores presentation messages and idempotency receipts;
the existing gateway owns agent history, commands, tools and review hooks.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
import hashlib
import json
import re
import sqlite3
import time
import uuid

PREFIX = "dashboard-"
ACCOUNT_RE = re.compile(r"^[a-f0-9]{64}$")
_reservation = ContextVar("dashboard_turn_reservation", default=None)


def is_dashboard_source(source):
    platform = getattr(getattr(source, "platform", None), "value", "")
    chat_id = str(getattr(source, "chat_id", ""))
    return platform == "api_server" and chat_id.startswith(PREFIX) and bool(ACCOUNT_RE.fullmatch(chat_id[len(PREFIX):]))


class BusyError(Exception):
    pass


class ConflictError(Exception):
    pass


class GlobalAdmission:
    """One event-loop-owned lease. Admission has no awaits, so races cannot pass."""
    def __init__(self, external_busy=lambda: False):
        self.owner = None
        self.token = None
        self.external_busy = external_busy

    def claim(self, owner):
        if self.owner is not None or self.external_busy():
            raise BusyError()
        self.owner, self.token = owner, uuid.uuid4().hex
        return self.token

    def release(self, token):
        if token == self.token:
            self.owner = self.token = None

    def owns(self, owner):
        return self.owner == owner and self.token == _reservation.get()


class DashboardStore:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL,
                id TEXT NOT NULL UNIQUE, role TEXT NOT NULL, content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'message', created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS messages_account ON messages(account, seq);
            CREATE TABLE IF NOT EXISTS requests (
                account TEXT NOT NULL, id TEXT NOT NULL, digest TEXT NOT NULL,
                status TEXT NOT NULL, activity TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(account, id));
            CREATE TABLE IF NOT EXISTS sessions (
                account TEXT NOT NULL, session_id TEXT NOT NULL,
                PRIMARY KEY(account, session_id));
        """)
        # A restart must not silently repeat tools with side effects.
        with self.db:
            self.db.execute("UPDATE requests SET status='interrupted', activity='' WHERE status='running'")

    def receipt(self, account, request_id, content):
        row = self.db.execute("SELECT * FROM requests WHERE account=? AND id=?", (account, request_id)).fetchone()
        if row and row["digest"] != hashlib.sha256(content.encode()).hexdigest():
            raise ConflictError()
        return dict(row) if row else None

    def admit(self, account, request_id, content):
        with self.db:
            self.db.execute("INSERT INTO requests(account,id,digest,status) VALUES(?,?,?,'running')",
                            (account, request_id, hashlib.sha256(content.encode()).hexdigest()))
            self.db.execute("INSERT INTO messages(account,id,role,content,created) VALUES(?,?,'user',?,?)",
                            (account, uuid.uuid4().hex, content, time.time()))

    def add(self, account, content, kind="message"):
        message_id = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO messages(account,id,role,content,kind,created) VALUES(?,?,'assistant',?,?,?)",
                            (account, message_id, content, kind, time.time()))
        return message_id

    def edit(self, account, message_id, content):
        with self.db:
            result = self.db.execute("UPDATE messages SET content=? WHERE account=? AND id=? AND role='assistant'",
                                     (content, account, message_id))
        return result.rowcount == 1

    def delete(self, account, message_id):
        with self.db:
            result = self.db.execute("DELETE FROM messages WHERE account=? AND id=? AND role='assistant'", (account, message_id))
        return result.rowcount == 1

    def status(self, account, request_id, status, activity=""):
        with self.db:
            self.db.execute("UPDATE requests SET status=?, activity=? WHERE account=? AND id=?",
                            (status, activity, account, request_id))

    def activity(self, account, text):
        with self.db:
            self.db.execute("UPDATE requests SET activity=? WHERE account=? AND status='running'", (text, account))

    def snapshot(self, account):
        messages = self.db.execute("SELECT id,role,content,kind,created FROM (SELECT * FROM messages WHERE account=? ORDER BY seq DESC LIMIT 500) ORDER BY seq", (account,)).fetchall()
        request = self.db.execute("SELECT id,status,activity FROM requests WHERE account=? ORDER BY rowid DESC LIMIT 1", (account,)).fetchone()
        return {"messages": [dict(row) for row in messages], "request": dict(request) if request else None}

    def remember_session(self, account, session_id):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO sessions(account,session_id) VALUES(?,?)", (account, session_id))

    def sessions(self, account):
        return [row[0] for row in self.db.execute("SELECT session_id FROM sessions WHERE account=? ORDER BY rowid DESC", (account,))]


def command_catalog():
    from hermes_cli.commands import COMMAND_REGISTRY, _is_gateway_available, _resolve_config_gates
    gates = _resolve_config_gates()
    return [{"name": item.name, "description": item.description, "args": item.args_hint,
             "aliases": list(item.aliases), "available": _is_gateway_available(item, gates) and item.name not in {"topic", "platform", "sethome", "branch"},
             "reason": "Requires a different transport" if item.name in {"topic", "platform", "sethome", "branch"} else "CLI only" if not _is_gateway_available(item, gates) else ""} for item in COMMAND_REGISTRY]


class DashboardBridge:
    def __init__(self, adapter, path):
        self.adapter = adapter
        self.store = DashboardStore(path)
        self.admission = GlobalAdmission(self._external_busy)
        self.tasks = set()
        self.outcomes = {}
        self.commands = command_catalog()

    def _external_busy(self):
        # A listener may be enabled/reconnected while another native turn is
        # already running. Do not admit a competing job into that startup gap.
        runner = self.adapter.gateway_runner
        active = getattr(runner, "_running_agents", {})
        return bool(active) or self.adapter._inflight_agent_runs > 0

    def snapshot(self, account):
        data = self.store.snapshot(account)
        owner = self.admission.owner
        if data["request"] and data["request"]["status"] == "running" and owner != ("api_server", PREFIX + account):
            # A failed terminal SQLite write cannot leave a false typing state.
            # The durable receipt still prevents duplicate tool execution.
            data["request"]["status"] = "failed"
            data["request"]["activity"] = ""
        data.update(busy=owner is not None or self._external_busy(),
                    ownBusy=owner == ("api_server", PREFIX + account),
                    commands=self.commands)
        return data

    def submit(self, account, request_id, content):
        previous = self.store.receipt(account, request_id, content)
        if previous:
            return previous
        token = self.admission.claim(("api_server", PREFIX + account))
        try:
            self.store.admit(account, request_id, content)
            task = asyncio.create_task(self._run(account, request_id, content, token))
            self.tasks.add(task)
            self.adapter._track_background_task(task)
            def settled(completed):
                self.tasks.discard(completed)
                if self.admission.token == token:
                    try:
                        self.store.status(account, request_id, "interrupted" if completed.cancelled() else "failed")
                    finally:
                        self.admission.release(token)
                if not completed.cancelled():
                    completed.exception()  # Consume an unexpected task exception.
            task.add_done_callback(settled)
        except BaseException:
            self.admission.release(token)
            raise
        return {"id": request_id, "status": "running"}

    async def _run(self, account, request_id, content, token):
        from gateway.config import Platform
        from gateway.platforms.event import MessageEvent, MessageType
        from gateway.session import SessionSource
        context_token = _reservation.set(token)
        status = "failed"
        try:
            if content.startswith("/"):
                command = content.split()[0][1:].lower().replace("_", "-")
                item = next((item for item in self.commands if command == item["name"] or command in item["aliases"]), None)
                if item and not item["available"]:
                    self.store.add(account, "هذا الأمر غير متاح عبر الموقع؛ يحتاج الطرفية أو منصة تدعم وظيفته.")
                    status = "completed"
                    return
                if item and item["name"] == "model" and "--global" in content.split():
                    self.store.add(account, "تغيير الموديل من الموقع يخص محادثتك فقط. استخدم /model بدون --global.")
                    status = "completed"
                    return
                if item and item["name"] == "sessions":
                    self.store.add(account, "جلسات محادثتك فقط:\n" + "\n".join(f"`{sid}`" for sid in self.store.sessions(account)))
                    status = "completed"
                    return
                if item and item["name"] == "resume":
                    args = content.split(maxsplit=1)
                    if len(args) != 2 or args[1].strip() not in self.store.sessions(account):
                        self.store.add(account, "يمكنك استئناف جلسات حسابك فقط. استخدم /sessions لعرض معرّفاتها.")
                        status = "completed"
                        return
            source = SessionSource(platform=Platform.API_SERVER, chat_id=PREFIX + account,
                                   user_id=PREFIX + account, role_authorized=True)
            # This trust signal is created only after API authentication, never read from JSON.
            event = MessageEvent(text=content, source=source, user_id=source.user_id,
                                 message_id=request_id,
                                 message_type=MessageType.COMMAND if content.startswith("/") else MessageType.TEXT)
            runner = self.adapter.gateway_runner
            key = runner._session_key_for_source(source)
            # Match the existing API's model-route alias on a NEW dashboard session only.
            entry = await runner.async_session_store.get_or_create_session(source)
            self.store.remember_session(account, entry.session_id)
            if not entry.model_override and not self.store.db.execute(
                    "SELECT 1 FROM requests WHERE account=? AND id<>?", (account, request_id)).fetchone():
                route = self.adapter._model_routes.get("hermes-agent")
                if route:
                    await runner.async_session_store.set_model_override(key, route)
                    runner._session_model_overrides[key] = dict(route)
            # Actual BasePlatformAdapter delivery handles streaming edits, commentary,
            # final response and post-delivery background-review release exactly as IM.
            await self.adapter._process_message_background(event, key)
            entry = await runner.async_session_store.get_or_create_session(source)
            self.store.remember_session(account, entry.session_id)
            status = self.outcomes.pop(account, "completed")
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception:
            # Do not persist exception strings containing providers, keys or private paths.
            self.store.add(account, "تعذر إكمال الطلب. تحقق من اتصال حمودي وحاول مرة أخرى.", "error")
        finally:
            try:
                self.store.status(account, request_id, status)
            finally:
                self.admission.release(token)
                _reservation.reset(context_token)


async def guarded_gateway_message(runner, event, invoke):
    """Use the same admission point for Telegram and every other gateway adapter."""
    bridge = getattr(runner, "_dashboard_bridge", None)
    if bridge is None:
        return await invoke(event)
    source = event.source
    owner = (source.platform.value, str(source.chat_id))
    if bridge.admission.owns(owner):
        return await invoke(event)
    # Preserve controls required to finish a blocked turn; other chat input is refused.
    command = event.text.strip().split(" ", 1)[0].lower()
    if bridge.admission.owner == owner and command in {"/approve", "/deny", "/stop"}:
        return await invoke(event)
    try:
        token = bridge.admission.claim(owner)
    except BusyError:
        return "حمودي مشغول حاليًا. انتظر انتهاء الطلب ثم أرسل رسالتك."
    context_token = _reservation.set(token)
    try:
        return await invoke(event)
    finally:
        bridge.admission.release(token)
        _reservation.reset(context_token)


async def dashboard_http(adapter, request):
    """Private transport endpoints. Dashboard authenticates admins and derives account IDs."""
    from aiohttp import web
    auth_error = adapter._check_auth(request)
    if auth_error:
        return auth_error
    from gateway.platforms.api_server import _api_request_profile
    if _api_request_profile.get() is not None:
        return web.json_response({"error": "Dashboard transport currently serves the default profile only"}, status=503)
    bridge = getattr(adapter, "_dashboard_bridge", None)
    if bridge is None or adapter.gateway_runner is None:
        return web.json_response({"error": "Dashboard transport is not enabled"}, status=503)
    account = request.match_info["account"]
    if not ACCOUNT_RE.fullmatch(account):
        return web.json_response({"error": "Invalid account"}, status=400)
    if request.method == "POST":
        draining = adapter._draining_response()
        if draining is not None:
            return draining
        try:
            body = await request.json()
            content, request_id = body.get("content"), body.get("requestId")
            if not isinstance(content, str) or not content.strip() or len(content) > 8000:
                raise ValueError()
            if not isinstance(request_id, str) or not re.fullmatch(r"[a-zA-Z0-9-]{16,80}", request_id):
                raise ValueError()
            bridge.submit(account, request_id, content.strip())
            return web.json_response(bridge.snapshot(account), status=202)
        except BusyError:
            return web.json_response({"error": "حمودي مشغول حاليًا. انتظر انتهاء الطلب.", "code": "HERMES_BUSY"}, status=409)
        except ConflictError:
            return web.json_response({"error": "Request ID already used for different text"}, status=409)
        except (ValueError, AttributeError):
            return web.json_response({"error": "Invalid message"}, status=400)
    if request.path.endswith("/events"):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})
        await response.prepare(request)
        previous = None
        heartbeat = 0
        try:
            while True:
                payload = json.dumps(bridge.snapshot(account), ensure_ascii=False)
                if payload != previous:
                    await response.write(("event: snapshot\ndata: " + payload + "\n\n").encode())
                    previous = payload
                elif heartbeat % 15 == 0:
                    await response.write(b": heartbeat\n\n")
                heartbeat += 1
                await asyncio.sleep(1)
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass  # A subscription never owns or interrupts agent work.
        return response
    return web.json_response(bridge.snapshot(account), headers={"Cache-Control": "no-store"})
