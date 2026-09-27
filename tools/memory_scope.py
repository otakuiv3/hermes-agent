"""Validated, context-local memory namespace for authenticated dashboard turns."""
from contextvars import ContextVar
import re

memory_namespace = ContextVar("memory_namespace", default=None)


def bind_memory_namespace(account):
    if not isinstance(account, str) or not re.fullmatch(r"[a-f0-9]{64}", account):
        raise ValueError("Invalid memory namespace")
    return memory_namespace.set(account)
