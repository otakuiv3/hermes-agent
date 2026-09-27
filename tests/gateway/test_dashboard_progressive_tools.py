"""Dashboard disclosure preserves the native catalog, scope and memory dispatch."""
import json
from types import SimpleNamespace

import model_tools
from agent.tool_executor import _unwrap_tool_search_call
from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from tools.memory_scope import bind_memory_namespace, memory_namespace
from tools.memory_tool import MemoryStore
from tools import tool_search as ts

A = 'a' * 64


def test_scoped_tool_array_cache_and_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    enabled = ['memory', 'file', 'skills']
    before = model_tools.get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True)
    token = bind_memory_namespace(A)
    try:
        raw = model_tools.get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True, skip_tool_search_assembly=True)
        assert any(x['function']['name'] == 'memory' for x in raw)
        visible = model_tools.get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True)
        visible_names = {x['function']['name'] for x in visible}
        assert ts.BRIDGE_TOOL_NAMES <= visible_names
        assert 'memory' in visible_names and 'read_file' not in visible_names
        assert ts.load_config().listing_max_tokens == 400
        assert ts.scoped_deferrable_names(raw) | (visible_names - ts.BRIDGE_TOOL_NAMES) == {x['function']['name'] for x in raw}
        assert 'read_file' in str(ts.dispatch_tool_describe({'names': ['read_file']}, current_tool_defs=raw))
        assert visible == model_tools.get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True)
    finally:
        memory_namespace.reset(token)
    assert model_tools.get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True) == before


def test_direct_memory_and_deferred_file_preserve_scope(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    token = bind_memory_namespace(A)
    try:
        store = MemoryStore()
        store.load_from_disk()
        agent = SimpleNamespace(enabled_toolsets=['memory', 'file'], disabled_toolsets=[],
                                _memory_store=store, _memory_manager=None)
        name, args, error = _unwrap_tool_search_call(agent, 'memory',
            {'action': 'add', 'target': 'user', 'content': 'Prefers short replies'})
        assert error is None and name == 'memory'
        result = INLINE_TOOL_EXECUTORS[name](agent, args, InlineToolContext(effective_task_id=None, tool_call_id='local-test'))
        assert json.loads(result)['success']
        assert 'Prefers short replies' in (tmp_path / 'dashboard-memories' / A / 'USER.md').read_text(encoding='utf-8')
        file_name, file_args, file_error = _unwrap_tool_search_call(agent, 'tool_call', {
            'name': 'read_file', 'arguments': {'path': str(tmp_path / 'note.txt')}})
        assert file_error is None and file_name == 'read_file' and file_args['path'].endswith('note.txt')
        _, _, blocked = _unwrap_tool_search_call(agent, 'tool_call', {
            'name': 'terminal', 'arguments': {'command': 'echo test'}})
        assert blocked and 'not available' in blocked
        assert not (tmp_path / 'memories' / 'USER.md').exists()
    finally:
        memory_namespace.reset(token)
