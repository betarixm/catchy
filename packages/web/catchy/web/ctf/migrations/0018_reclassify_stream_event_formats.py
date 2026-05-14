from __future__ import annotations

import json
from typing import Any

from django.db import migrations


APP_EVENT_FORMAT = "catchy-app-event"


def forward(apps: Any, schema_editor: Any) -> None:
    stream_event_model = apps.get_model("ctf", "StreamEvent")
    for event in stream_event_model.objects.select_related("thread__agent").iterator():
        if event.format not in {"codex-notification", "claude-code-message"}:
            continue
        legacy = _legacy_payload_from_current(event)
        source = legacy.get("source")
        if isinstance(source, str) and source not in ("", "agent_stream"):
            # Recover non-agent-stream legacy events back into the app event
            # format so they render via the trace path rather than being
            # forced into a codex/claude observation chunk.
            stream_event_model.objects.filter(pk=event.pk).update(
                format=APP_EVENT_FORMAT,
                raw=json.dumps(legacy, ensure_ascii=False, separators=(",", ":")),
            )
            continue
        agent_yaml = getattr(getattr(event.thread, "agent", None), "yaml", "")
        is_claude = _is_claude_agent_yaml(agent_yaml)
        new_format = "claude-code-message" if is_claude else "codex-notification"
        new_raw = (
            _claude_raw_from_legacy(event, legacy)
            if is_claude
            else _codex_raw_from_legacy(event, legacy)
        )
        stream_event_model.objects.filter(pk=event.pk).update(
            format=new_format,
            raw=json.dumps(new_raw, ensure_ascii=False, separators=(",", ":")),
        )


def _is_claude_agent_yaml(agent_yaml: object) -> bool:
    if not isinstance(agent_yaml, str):
        return False
    text = agent_yaml.lower()
    return (
        "catchy.claude_code.claudecodeagent" in text
        or "claudecodeagent" in text
    )


def _legacy_payload_from_current(event: Any) -> dict[str, Any]:
    payload = _json_object(event.raw)
    if event.format == "codex-notification":
        return _legacy_from_codex_payload(payload)
    if event.format == "claude-code-message":
        return _legacy_from_claude_payload(payload)
    return {"source": "agent_stream", "kind": "chunk", "text": "", "raw": {}}


def _legacy_from_codex_payload(payload: dict[str, Any]) -> dict[str, Any]:
    method = payload.get("method")
    body = payload.get("payload")
    if not isinstance(method, str) or not isinstance(body, dict):
        return {"source": "agent_stream", "kind": "chunk", "text": "", "raw": {}}

    def _legacy(source: str, kind: str, text: str, raw: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"source": source, "kind": kind, "text": text, "raw": raw or {}}

    if method == "item/agentMessage/delta":
        return _legacy("agent_stream", "chunk", str(body.get("delta") or ""), {"tag": "action"})
    if method == "item/reasoning/textDelta":
        return _legacy("agent_stream", "thinking", str(body.get("delta") or ""), {"tag": "thinking"})
    if method == "item/started":
        item = body.get("item")
        if isinstance(item, dict):
            return _legacy(
                "agent_stream",
                "tool_use",
                json.dumps(item, ensure_ascii=False, sort_keys=True),
                {"tag": "tool_use"},
            )
    if method == "item/completed":
        return _legacy("agent_stream", "item.terminated", "")
    if method == "turn/completed":
        return _legacy("agent_stream", "turn.completed", "")
    if method == "thread/tokenUsage/updated":
        token_usage = body.get("tokenUsage")
        usage = token_usage.get("last") if isinstance(token_usage, dict) else {}
        return _legacy("agent_stream", "token_count", "", {"usage": usage if isinstance(usage, dict) else {}})
    if method == "item/commandExecution/outputDelta":
        delta_text = str(body.get("delta") or "")
        decoded = _json_object(delta_text)
        if _looks_like_legacy_payload(decoded):
            return {
                "source": str(decoded.get("source") or "agent_stream"),
                "kind": str(decoded.get("kind") or "chunk"),
                "text": str(decoded.get("text") or ""),
                "raw": decoded.get("raw") if isinstance(decoded.get("raw"), dict) else {},
            }
        return _legacy("agent_stream", "chunk", delta_text, {"tag": "observation"})
    return _legacy("agent_stream", "chunk", "")


def _legacy_from_claude_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def _legacy(kind: str, text: str, raw: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"source": "agent_stream", "kind": kind, "text": text, "raw": raw or {}}

    if payload.get("__claude_sdk_type__") != "StreamEvent":
        return _legacy("chunk", "")
    event_payload = payload.get("event")
    if not isinstance(event_payload, dict):
        return _legacy("chunk", "")
    event_type = event_payload.get("type")
    if event_type == "content_block_delta":
        delta = event_payload.get("delta")
        if not isinstance(delta, dict):
            return _legacy("chunk", "")
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = str(delta.get("text") or "")
            decoded = _json_object(text)
            if _looks_like_legacy_payload(decoded):
                return {
                    "source": str(decoded.get("source") or "agent_stream"),
                    "kind": str(decoded.get("kind") or "chunk"),
                    "text": str(decoded.get("text") or ""),
                    "raw": decoded.get("raw") if isinstance(decoded.get("raw"), dict) else {},
                }
            return _legacy("chunk", text, {"tag": "action"})
        if delta_type == "thinking_delta":
            return _legacy("thinking", str(delta.get("thinking") or ""), {"tag": "thinking"})
        if delta_type == "input_json_delta":
            return _legacy("tool_input", str(delta.get("partial_json") or ""), {"tag": "tool_input"})
    if event_type == "content_block_start":
        content_block = event_payload.get("content_block")
        if isinstance(content_block, dict):
            return _legacy(
                "tool_use",
                json.dumps(content_block, ensure_ascii=False, sort_keys=True),
                {"tag": "tool_use"},
            )
    if event_type == "content_block_stop":
        return _legacy("item.terminated", "")
    if event_type == "message_delta":
        usage = event_payload.get("usage")
        return _legacy("token_count", "", {"usage": usage if isinstance(usage, dict) else {}})
    return _legacy("chunk", "")


def _codex_raw_from_legacy(event: Any, legacy: dict[str, Any]) -> dict[str, Any]:
    kind = legacy.get("kind") if isinstance(legacy.get("kind"), str) else "chunk"
    text = legacy.get("text") if isinstance(legacy.get("text"), str) else ""
    legacy_raw = legacy.get("raw") if isinstance(legacy.get("raw"), dict) else {}
    turn_id = f"legacy-turn-{event.thread_id}"
    thread_id = f"legacy-thread-{event.thread_id}"
    item_id = f"legacy-item-{event.pk}"
    if kind in {"chunk", "delta"}:
        if isinstance(legacy_raw.get("tag"), str) and legacy_raw["tag"] == "observation":
            method = "item/commandExecution/outputDelta"
        else:
            method = "item/agentMessage/delta"
        payload = {"threadId": thread_id, "turnId": turn_id, "itemId": item_id, "delta": text}
    elif kind == "thinking":
        method = "item/reasoning/textDelta"
        payload = {
            "threadId": thread_id,
            "turnId": turn_id,
            "itemId": item_id,
            "contentIndex": 0,
            "delta": text,
        }
    elif kind == "tool_use":
        method = "item/started"
        payload = {
            "threadId": thread_id,
            "turnId": turn_id,
            "item": _codex_tool_item(event, legacy_raw, text=text, status="inProgress"),
        }
    elif kind == "item.terminated":
        method = "item/completed"
        payload = {
            "threadId": thread_id,
            "turnId": turn_id,
            "item": _codex_tool_item(event, legacy_raw, text=text, status="completed"),
        }
    elif kind == "turn.completed":
        method = "turn/completed"
        payload = {
            "threadId": thread_id,
            "turn": {"id": turn_id, "status": "completed", "items": [], "error": None},
        }
    elif kind == "token_count":
        method = "thread/tokenUsage/updated"
        payload = {
            "threadId": thread_id,
            "turnId": turn_id,
            "tokenUsage": _codex_token_usage(legacy_raw),
        }
    else:
        method = "item/commandExecution/outputDelta"
        payload = {"threadId": thread_id, "turnId": turn_id, "itemId": item_id, "delta": text}
    return {"method": method, "payload": payload}


def _claude_raw_from_legacy(event: Any, legacy: dict[str, Any]) -> dict[str, Any]:
    kind = legacy.get("kind") if isinstance(legacy.get("kind"), str) else "chunk"
    text = legacy.get("text") if isinstance(legacy.get("text"), str) else ""
    legacy_raw = legacy.get("raw") if isinstance(legacy.get("raw"), dict) else {}
    if kind in {"chunk", "delta"}:
        event_payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    elif kind == "thinking":
        event_payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": text}}
    elif kind == "tool_input":
        event_payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": text}}
    elif kind == "tool_use":
        event_payload = {"type": "content_block_start", "index": 0, "content_block": _claude_tool_use_block(text)}
    elif kind == "item.terminated":
        event_payload = {"type": "content_block_stop", "index": 0}
    elif kind == "token_count":
        event_payload = {"type": "message_delta", "usage": _claude_usage(legacy_raw)}
    else:
        event_payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    return {
        "__claude_sdk_type__": "StreamEvent",
        "uuid": f"legacy-{event.pk}",
        "session_id": f"legacy-thread-{event.thread_id}",
        "event": event_payload,
        "parent_tool_use_id": None,
    }


def _codex_tool_item(event: Any, raw: dict[str, Any], *, text: str, status: str) -> dict[str, Any]:
    parsed = _json_object(text)
    command = parsed.get("command") or parsed.get("name") or text or "command"
    return {
        "type": "commandExecution",
        "id": str(parsed.get("id") or f"legacy-item-{event.pk}"),
        "command": str(command),
        "cwd": str(parsed.get("cwd") or ""),
        "status": status,
        "commandActions": [],
        "exitCode": _int_or_none(raw.get("exitCode") or raw.get("exit_code")),
        "aggregatedOutput": str(raw.get("aggregatedOutput") or raw.get("output") or text or ""),
    }


def _codex_token_usage(raw: dict[str, Any]) -> dict[str, Any]:
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        usage = raw
    total = {
        "inputTokens": _int_value(usage.get("inputTokens") or usage.get("input_tokens")),
        "cachedInputTokens": _int_value(usage.get("cachedInputTokens") or usage.get("cached_input_tokens")),
        "outputTokens": _int_value(usage.get("outputTokens") or usage.get("output_tokens")),
        "reasoningOutputTokens": _int_value(
            usage.get("reasoningOutputTokens") or usage.get("reasoning_output_tokens")
        ),
    }
    total["totalTokens"] = _int_value(usage.get("totalTokens")) or sum(total.values())
    return {"last": total, "total": total, "modelContextWindow": 0}


def _claude_tool_use_block(text: str) -> dict[str, Any]:
    payload = _json_object(text)
    return {
        "type": "tool_use",
        "id": str(payload.get("id") or "legacy-tool"),
        "name": str(payload.get("name") or payload.get("tool") or "tool"),
        "input": payload.get("input") if isinstance(payload.get("input"), dict) else {},
    }


def _claude_usage(raw: dict[str, Any]) -> dict[str, Any]:
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        usage = raw
    return {
        "input_tokens": _int_value(usage.get("input_tokens") or usage.get("inputTokens")),
        "cache_creation_input_tokens": _int_value(
            usage.get("cache_creation_input_tokens") or usage.get("cacheCreationInputTokens")
        ),
        "cache_read_input_tokens": _int_value(
            usage.get("cache_read_input_tokens") or usage.get("cacheReadInputTokens")
        ),
        "output_tokens": _int_value(usage.get("output_tokens") or usage.get("outputTokens")),
    }


def _looks_like_legacy_payload(value: dict[str, Any]) -> bool:
    return (
        isinstance(value.get("source"), str)
        and isinstance(value.get("kind"), str)
        and "text" in value
    )


def _json_object(text: object) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    return _int_value(value)


class Migration(migrations.Migration):
    dependencies = [("ctf", "0017_stream_event_raw_event")]
    operations = [migrations.RunPython(forward, migrations.RunPython.noop)]
