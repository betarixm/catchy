from __future__ import annotations

import json
import re
from typing import Any

from django.db import migrations


APP_EVENT_FORMAT = "catchy-app-event"
AGENT_SOURCES = frozenset({"agent_stream"})

# Status values that produced "Thread <status>" text via _record_event.
_THREAD_STATUS_KINDS = (
    "started",
    "completed",
    "failed",
    "stopped",
    "waiting",
    "queued",
    "running",
)
_THREAD_STATUS_PATTERN = re.compile(
    r"^Thread (" + "|".join(_THREAD_STATUS_KINDS) + r")$"
)
_THREAD_FORKED_PATTERN = re.compile(r"^Forked from thread #(\d+)$")

# Codex JSONL `type` values that were logged with source=codex_jsonl in the
# legacy schema. Recovering them lets the frontend's trace classifier render
# them as cards instead of observation chunks.
_CODEX_JSONL_TYPES = frozenset(
    {
        "function_call",
        "function_call_output",
        "local_shell_call",
        "local_shell_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "web_search_call",
        "message",
        "reasoning",
    }
)


def forward(apps: Any, schema_editor: Any) -> None:
    stream_event_model = apps.get_model("ctf", "StreamEvent")
    for event in stream_event_model.objects.iterator():
        recovered: dict[str, Any] | None = None
        if event.format == "codex-notification":
            recovered = _recover_from_codex_notification(event.raw)
        elif event.format == "claude-code-message":
            recovered = _recover_from_claude_message(event.raw)
        if recovered is None:
            continue
        stream_event_model.objects.filter(pk=event.pk).update(
            format=APP_EVENT_FORMAT,
            raw=json.dumps(recovered, ensure_ascii=False, separators=(",", ":")),
        )


def _recover_from_codex_notification(raw_text: object) -> dict[str, Any] | None:
    payload = _json_object(raw_text)
    if not payload:
        return None
    if payload.get("method") != "item/commandExecution/outputDelta":
        return None
    body = payload.get("payload")
    if not isinstance(body, dict):
        return None
    delta = body.get("delta")
    if not isinstance(delta, str):
        return None
    return _recover_from_delta_text(delta)


def _recover_from_claude_message(raw_text: object) -> dict[str, Any] | None:
    payload = _json_object(raw_text)
    if not payload:
        return None
    if payload.get("__claude_sdk_type__") != "StreamEvent":
        return None
    event_payload = payload.get("event")
    if not isinstance(event_payload, dict):
        return None
    if event_payload.get("type") != "content_block_delta":
        return None
    delta = event_payload.get("delta")
    if not isinstance(delta, dict):
        return None
    if delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    if not isinstance(text, str):
        return None
    return _recover_from_delta_text(text)


def _recover_from_delta_text(delta: str) -> dict[str, Any] | None:
    decoded = _json_object(delta)
    if _looks_like_legacy_payload(decoded):
        return _from_legacy_payload(decoded)
    if decoded:
        codex_type = decoded.get("type")
        if isinstance(codex_type, str) and codex_type in _CODEX_JSONL_TYPES:
            return {
                "source": "codex_jsonl",
                "kind": codex_type,
                "text": delta,
                "raw": decoded,
            }
    match = _THREAD_STATUS_PATTERN.match(delta)
    if match:
        return {
            "source": "system",
            "kind": f"thread.{match.group(1)}",
            "text": delta,
            "raw": {},
        }
    match = _THREAD_FORKED_PATTERN.match(delta)
    if match:
        return {
            "source": "system",
            "kind": "thread.forked",
            "text": delta,
            "raw": {"source_thread_id": int(match.group(1))},
        }
    if delta == "Workspace updated":
        return {
            "source": "system",
            "kind": "workspace.changed",
            "text": delta,
            "raw": {},
        }
    return None


def _from_legacy_payload(decoded: dict[str, Any]) -> dict[str, Any] | None:
    source = decoded.get("source")
    if not isinstance(source, str) or source in AGENT_SOURCES:
        return None
    raw_payload = decoded.get("raw")
    return {
        "source": source,
        "kind": decoded.get("kind") if isinstance(decoded.get("kind"), str) else "",
        "text": decoded.get("text") if isinstance(decoded.get("text"), str) else "",
        "raw": raw_payload if isinstance(raw_payload, dict) else {},
    }


def _looks_like_legacy_payload(value: dict[str, Any]) -> bool:
    return (
        isinstance(value.get("source"), str)
        and isinstance(value.get("kind"), str)
        and "text" in value
    )


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class Migration(migrations.Migration):
    dependencies = [("ctf", "0018_reclassify_stream_event_formats")]
    operations = [migrations.RunPython(forward, migrations.RunPython.noop)]
