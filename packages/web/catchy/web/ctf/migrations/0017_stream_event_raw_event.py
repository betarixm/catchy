from __future__ import annotations

import json
from typing import Any

from django.db import migrations, models

APP_EVENT_FORMAT = "catchy-app-event"


def backfill_stream_event_raw_event(apps: Any, schema_editor: Any) -> None:
    stream_event_model = apps.get_model("ctf", "StreamEvent")
    for event in stream_event_model.objects.select_related("thread__agent").iterator():
        old_raw_value: Any = event.old_raw
        raw: dict[str, Any] = old_raw_value if isinstance(old_raw_value, dict) else {}
        source = getattr(event, "source", "") or ""
        if source not in ("", "agent_stream"):
            # System/user/legacy-jsonl events live outside the agent renderer's
            # vocabulary. Preserve them as app-format events so the renderer does
            # not attempt to bucket them into observation/agent chunks.
            format_name = APP_EVENT_FORMAT
            raw_payload = {
                "source": source,
                "kind": event.kind or "",
                "text": event.text or "",
                "raw": raw,
            }
        else:
            agent_yaml = getattr(getattr(event.thread, "agent", None), "yaml", "")
            is_claude = _is_claude_agent_yaml(agent_yaml)
            if is_claude:
                format_name = "claude-code-message"
                raw_payload = _claude_raw(event, raw)
            else:
                format_name = "codex-notification"
                raw_payload = _codex_raw(event, raw)
        stream_event_model.objects.filter(pk=event.pk).update(
            format=format_name,
            raw=json.dumps(raw_payload, ensure_ascii=False, separators=(",", ":")),
        )


def _is_claude_agent_yaml(agent_yaml: object) -> bool:
    if not isinstance(agent_yaml, str):
        return False
    text = agent_yaml.lower()
    # Prefer explicit class path detection; avoid over-broad "contains claude".
    return (
        "catchy.claude_code.claudecodeagent" in text
        or "claudecodeagent" in text
    )


def _legacy_payload(event: Any, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": event.source,
        "kind": event.kind,
        "text": event.text,
        "raw": raw,
    }


def _codex_raw(event: Any, raw: dict[str, Any]) -> dict[str, Any]:
    kind = event.kind
    text = event.text
    turn_id = f"legacy-turn-{event.thread_id}"
    thread_id = f"legacy-thread-{event.thread_id}"
    item_id = f"legacy-item-{event.pk}"
    if kind in {"chunk", "delta"}:
        if raw.get("tag") == "observation":
            method = "item/commandExecution/outputDelta"
        else:
            method = "item/agentMessage/delta"
        payload = {
            "threadId": thread_id,
            "turnId": turn_id,
            "itemId": item_id,
            "delta": text,
        }
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
            "item": _codex_tool_item(event, raw, status="inProgress"),
        }
    elif kind == "item.terminated":
        method = "item/completed"
        payload = {
            "threadId": thread_id,
            "turnId": turn_id,
            "item": _codex_tool_item(event, raw, status="completed"),
        }
    elif kind == "turn.completed":
        method = "turn/completed"
        payload = {
            "threadId": thread_id,
            "turn": {
                "id": turn_id,
                "status": "completed",
                "items": [],
                "error": None,
            },
        }
    elif kind == "token_count":
        method = "thread/tokenUsage/updated"
        payload = {
            "threadId": thread_id,
            "turnId": turn_id,
            "tokenUsage": _codex_token_usage(raw),
        }
    else:
        method = "item/commandExecution/outputDelta"
        payload = {
            "threadId": thread_id,
            "turnId": turn_id,
            "itemId": item_id,
            "delta": json.dumps(
                _legacy_payload(event, raw), ensure_ascii=False, separators=(",", ":")
            ),
        }
    return {"method": method, "payload": payload}


def _codex_tool_item(
    event: Any,
    raw: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    parsed = _json_object(event.text)
    command = parsed.get("command") or parsed.get("name") or event.text or event.kind
    return {
        "type": "commandExecution",
        "id": str(parsed.get("id") or f"legacy-item-{event.pk}"),
        "command": str(command),
        "cwd": str(parsed.get("cwd") or ""),
        "status": status,
        "commandActions": [],
        "exitCode": _int_or_none(raw.get("exitCode") or raw.get("exit_code")),
        "aggregatedOutput": str(
            raw.get("aggregatedOutput") or raw.get("output") or event.text or ""
        ),
    }


def _codex_token_usage(raw: dict[str, Any]) -> dict[str, Any]:
    token_usage = raw.get("tokenUsage")
    if isinstance(token_usage, dict):
        return token_usage
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        usage = raw
    total = {
        "inputTokens": _int_value(
            usage.get("inputTokens") or usage.get("input_tokens")
        ),
        "cachedInputTokens": _int_value(
            usage.get("cachedInputTokens") or usage.get("cached_input_tokens")
        ),
        "outputTokens": _int_value(
            usage.get("outputTokens") or usage.get("output_tokens")
        ),
        "reasoningOutputTokens": _int_value(
            usage.get("reasoningOutputTokens")
            or usage.get("reasoning_output_tokens")
        ),
    }
    total["totalTokens"] = _int_value(usage.get("totalTokens")) or sum(total.values())
    return {"last": total, "total": total, "modelContextWindow": 0}


def _claude_raw(event: Any, raw: dict[str, Any]) -> dict[str, Any]:
    kind = event.kind
    text = event.text
    if kind in {"chunk", "delta"}:
        event_payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }
    elif kind == "thinking":
        event_payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": text},
        }
    elif kind == "tool_input":
        event_payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": text},
        }
    elif kind == "tool_use":
        event_payload = {
            "type": "content_block_start",
            "index": 0,
            "content_block": _claude_tool_use_block(text),
        }
    elif kind == "item.terminated":
        event_payload = {"type": "content_block_stop", "index": 0}
    elif kind == "token_count":
        event_payload = {"type": "message_delta", "usage": _claude_usage(raw)}
    else:
        event_payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": json.dumps(
                    _legacy_payload(event, raw),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        }
    return {
        "__claude_sdk_type__": "StreamEvent",
        "uuid": f"legacy-{event.pk}",
        "session_id": f"legacy-thread-{event.thread_id}",
        "event": event_payload,
        "parent_tool_use_id": None,
    }


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
            usage.get("cache_creation_input_tokens")
            or usage.get("cacheCreationInputTokens")
        ),
        "cache_read_input_tokens": _int_value(
            usage.get("cache_read_input_tokens") or usage.get("cacheReadInputTokens")
        ),
        "output_tokens": _int_value(
            usage.get("output_tokens") or usage.get("outputTokens")
        ),
    }


def _json_object(text: str) -> dict[str, Any]:
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
    dependencies = [
        ("ctf", "0016_credential_allowed_users"),
    ]

    operations = [
        migrations.AddField(
            model_name="streamevent",
            name="format",
            field=models.CharField(blank=True, max_length=120),
        ),
        migrations.RenameField(
            model_name="streamevent",
            old_name="raw",
            new_name="old_raw",
        ),
        migrations.AddField(
            model_name="streamevent",
            name="raw",
            field=models.TextField(blank=True),
        ),
        migrations.RunPython(backfill_stream_event_raw_event, migrations.RunPython.noop),
        migrations.AlterModelOptions(
            name="streamevent",
            options={"ordering": ["id"]},
        ),
        migrations.AlterUniqueTogether(
            name="streamevent",
            unique_together=set(),
        ),
        migrations.RemoveField(
            model_name="streamevent",
            name="dedupe_key",
        ),
        migrations.RemoveField(
            model_name="streamevent",
            name="kind",
        ),
        migrations.RemoveField(
            model_name="streamevent",
            name="old_raw",
        ),
        migrations.RemoveField(
            model_name="streamevent",
            name="sequence",
        ),
        migrations.RemoveField(
            model_name="streamevent",
            name="source",
        ),
        migrations.RemoveField(
            model_name="streamevent",
            name="text",
        ),
    ]
