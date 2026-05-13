from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import threading
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from asgiref.sync import sync_to_async
from catchy.core.agents.models import (
    Chunk,
    Event,
    Interrupt,
    ItemCompleted,
    Log,
    Nop,
    Prompt,
    Steer,
    Stop,
    TokenUsage,
    TurnCompleted,
)
from catchy.core.agents.protocols import Agent
from catchy.core.challenge.models import Challenge as CoreChallenge
from catchy.core.webhook.models import Webhook
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.db.models import Max
from django.utils import timezone
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .models import (
    AgentConfiguration,
    Credential,
    ModelConfiguration,
    ModelPricing,
    SteeringMessage,
    StreamEvent,
    Thread,
    ThreadCostSnapshot,
)
from .source_archives import safe_extract_archive

_CODEX_RUNTIME_METADATA_DIRS = frozenset({".tmp", "tmp"})
THREAD_COST_USD_KEY = "cost_usd"
_WORKSPACE_REFRESH_DEBOUNCE_SECONDS = 0.35
_workspace_refresh_lock = threading.Lock()
_workspace_refresh_timers: dict[int, threading.Timer] = {}
_workspace_refresh_paths: dict[int, list[str]] = {}


class _WorkspaceEventHandler(FileSystemEventHandler):
    def __init__(self, thread_id: int):
        self.thread_id = thread_id

    def on_any_event(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        path = getattr(event, "src_path", None) or getattr(event, "dest_path", None)
        if not path:
            return
        _queue_workspace_changed_event(self.thread_id, str(path))


def _start_workspace_observer(thread_id: int, workspace: Path) -> Any:
    observer = Observer()
    observer.schedule(_WorkspaceEventHandler(thread_id), str(workspace), recursive=True)
    observer.start()
    return observer


def _queue_workspace_changed_event(thread_id: int, changed_path: str) -> None:
    with _workspace_refresh_lock:
        paths = _workspace_refresh_paths.setdefault(thread_id, [])
        paths.append(changed_path)
        if len(paths) > 20:
            del paths[:-20]

        timer = _workspace_refresh_timers.get(thread_id)
        if timer is not None:
            timer.cancel()

        timer = threading.Timer(
            _WORKSPACE_REFRESH_DEBOUNCE_SECONDS,
            _flush_workspace_changed_event,
            args=(thread_id,),
        )
        timer.daemon = True
        _workspace_refresh_timers[thread_id] = timer
        timer.start()


def _record_workspace_changed_stream_event(thread_id: int, changed_paths: list[str]) -> None:
    thread = Thread.objects.get(pk=thread_id)
    _record_event(
        thread,
        source="system",
        kind="workspace.changed",
        text="Workspace updated",
        raw={"changed_paths": changed_paths},
    )


def _flush_workspace_changed_event(thread_id: int) -> None:
    with _workspace_refresh_lock:
        changed_paths = _workspace_refresh_paths.pop(thread_id, [])
        timer = _workspace_refresh_timers.pop(thread_id, None)
    if timer is not None:
        timer.cancel()
    if changed_paths:
        _record_workspace_changed_stream_event(thread_id, changed_paths)


def start_thread(thread: Thread) -> Any:
    worker = threading.Thread(
        target=run_thread_sync,
        args=(thread.pk,),
        daemon=True,
        name=f"catchy-thread-{thread.pk}",
    )
    worker.start()
    thread.task_result_id = f"local-thread:{worker.name}"
    thread.save(update_fields=["task_result_id", "updated_at"])
    return worker


def fork_thread(thread: Thread, *, user: Any | None = None) -> Thread:
    fork = Thread.objects.create(
        ctf=thread.ctf,
        challenge=thread.challenge,
        agent=thread.agent,
        model=thread.model,
        credential=thread.credential,
        created_by=user or thread.created_by,
        name=_fork_thread_name(thread),
        status=Thread.Status.WAITING,
        latest_cost=thread.latest_cost,
    )

    thread_root = _thread_root(fork)
    metadata = thread_root / "metadata"
    workspace = thread_root / "workspace"
    metadata.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    if thread.metadata_path:
        source_metadata = Path(thread.metadata_path)
        if source_metadata.exists():
            shutil.copytree(
                source_metadata,
                metadata,
                dirs_exist_ok=True,
                ignore=_ignore_runtime_metadata,
            )

    fork.thread_root = str(thread_root)
    fork.workspace_path = str(workspace)
    fork.metadata_path = str(metadata)
    fork.save(
        update_fields=[
            "thread_root",
            "workspace_path",
            "metadata_path",
            "updated_at",
        ]
    )

    for event in thread.events.order_by("sequence"):
        StreamEvent.objects.create(
            thread=fork,
            sequence=event.sequence,
            dedupe_key=event.dedupe_key,
            source=event.source,
            kind=event.kind,
            text=event.text,
            raw=event.raw,
        )
    _record_event(
        fork,
        source="system",
        kind="thread.forked",
        text=f"Forked from thread #{thread.pk}",
        raw={"source_thread_id": thread.pk},
    )
    return fork


def _ignore_runtime_metadata(directory: str, names: list[str]) -> set[str]:
    if Path(directory).name != ".codex":
        return set()
    return set(_CODEX_RUNTIME_METADATA_DIRS.intersection(names))


def run_thread_sync(thread_id: int) -> None:
    thread = (
        Thread.objects.select_related(
            "challenge",
            "challenge__ctf",
            "agent",
            "model",
            "credential",
        )
        .select_related("created_by")
        .prefetch_related(
            "agent__use_groups",
            "model__use_groups",
            "credential__allowed_groups",
            "credential__allowed_users",
        )
        .get(pk=thread_id)
    )

    thread_root = _thread_root(thread)
    source_directory = thread_root / "source"
    workspace = thread_root / "workspace"
    metadata = thread_root / "metadata"
    source_directory.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    safe_extract_archive(Path(thread.challenge.source_archive.path), source_directory)

    thread.thread_root = str(thread_root)
    thread.workspace_path = str(workspace)
    thread.metadata_path = str(metadata)
    thread.status = Thread.Status.RUNNING
    thread.error = ""
    thread.save(
        update_fields=[
            "thread_root",
            "workspace_path",
            "metadata_path",
            "status",
            "error",
            "updated_at",
        ]
    )
    _record_event(thread, source="system", kind="thread.started", text="Thread started")

    observer = _start_workspace_observer(thread.pk, workspace)
    try:
        agent = load_agent(
            thread.agent,
            model_configuration=thread.model,
            credential=thread.credential,
            user=thread.created_by,
        )
        core_challenge = CoreChallenge(
            id=thread.challenge.challenge_id,
            description=thread.challenge.description,
            directory=source_directory,
        )
        webhook_data = thread.challenge.webhook_mapping()
        webhook = Webhook(**webhook_data) if webhook_data else None
        terminal_status = asyncio.run(
            _run_agent_stream(
                thread_id=thread.pk,
                agent=agent,
                challenge=core_challenge,
                workspace=workspace,
                metadata=metadata,
                webhook=webhook,
                model_name=_thread_model_name(thread),
            )
        )
    except Exception as exc:
        observer.stop()
        observer.join()
        Thread.objects.filter(pk=thread.pk).update(
            status=Thread.Status.FAILED,
            error=str(exc),
            updated_at=timezone.now(),
        )
        _record_event(thread, source="system", kind="thread.failed", text=str(exc))
        raise

    Thread.objects.filter(pk=thread.pk).update(
        status=terminal_status,
        updated_at=timezone.now(),
    )
    observer.stop()
    observer.join()
    thread.status = terminal_status
    _record_event(
        thread,
        source="system",
        kind=f"thread.{terminal_status}",
        text=f"Thread {terminal_status}",
    )


def load_agent(
    agent_configuration: AgentConfiguration,
    *,
    model_configuration: ModelConfiguration | None = None,
    credential: Credential | None = None,
    user: Any | None = None,
) -> Agent:
    data = build_agent_configuration(
        agent_configuration,
        model_configuration=model_configuration,
        credential=credential,
        user=user,
    )
    class_path = _agent_class_path(data)
    agent_class = _import_agent_class(class_path)
    configuration_class = getattr(
        importlib.import_module(agent_class.__module__),
        "Configuration",
        None,
    )
    if not hasattr(configuration_class, "model_validate"):
        raise TypeError(
            f"agent module must expose Configuration: {agent_class.__module__}"
        )

    from_configuration = getattr(agent_class, "from_configuration", None)
    if not callable(from_configuration):
        raise TypeError(f"agent class must expose from_configuration: {class_path}")

    agent = from_configuration(cast(Any, configuration_class).model_validate(data))
    if not isinstance(agent, Agent):
        raise TypeError(f"from_configuration did not return an Agent: {class_path}")
    return agent


def build_agent_configuration(
    agent_configuration: AgentConfiguration,
    *,
    model_configuration: ModelConfiguration | None = None,
    credential: Credential | None = None,
    user: Any | None = None,
) -> dict[str, Any]:
    if user is not None and not agent_configuration.can_use(user):
        raise PermissionDenied("agent configuration is not accessible")

    data = dict(agent_configuration.resolved_mapping(user=user))
    if model_configuration is None and credential is None:
        return data

    if model_configuration is None:
        raise ValueError("model configuration is required")
    if credential is None:
        raise ValueError("credential is required")
    if user is not None and not model_configuration.can_use(user):
        raise PermissionDenied("model configuration is not accessible")
    if user is not None and not credential.can_use(user):
        raise PermissionDenied("credential is not accessible")
    existing_model = data.get("model", {})
    model_data = (
        dict(cast(dict[str, Any], existing_model))
        if isinstance(existing_model, dict)
        else {}
    )
    model_data["name"] = model_configuration.name
    for stale_key in ("provider", "api_key", "base_url", "organization_id"):
        model_data.pop(stale_key, None)

    data["model"] = model_data
    data["credential"] = _credential_configuration_for_agent(data, credential)
    return data


def _credential_configuration_for_agent(
    data: dict[str, Any], credential: Credential
) -> dict[str, str]:
    class_path = _agent_class_path(data)
    if class_path == "catchy.codex.CodexAgent" and credential.kind not in {
        Credential.Kind.CODEX_AUTH_JSON,
        Credential.Kind.OPENAI,
    }:
        raise ValueError(
            f"credential kind is not compatible with Codex: {credential.kind}"
        )
    if class_path == "catchy.claude_code.ClaudeCodeAgent" and credential.kind not in {
        Credential.Kind.ANTHROPIC,
        Credential.Kind.CLAUDE_OAUTH_TOKEN,
    }:
        raise ValueError(
            f"credential kind is not compatible with Claude Code: {credential.kind}"
        )

    match credential.kind:
        case Credential.Kind.OPENAI:
            data = {"api_key": credential.api_key}
            if credential.base_url:
                data["base_url"] = credential.base_url
            if credential.organization_id:
                data["organization_id"] = credential.organization_id
            return data
        case Credential.Kind.CODEX_AUTH_JSON:
            data = {"json_string": credential.api_key}
            if credential.base_url:
                data["base_url"] = credential.base_url
            return data
        case Credential.Kind.ANTHROPIC:
            data = {"api_key": credential.api_key}
            if credential.base_url:
                data["base_url"] = credential.base_url
            return data
        case Credential.Kind.CLAUDE_OAUTH_TOKEN:
            return {"token": credential.api_key}
        case _:
            raise ValueError(f"unsupported credential kind: {credential.kind}")


async def _run_agent_stream(
    *,
    thread_id: int,
    agent: Agent,
    challenge: CoreChallenge,
    workspace: Path,
    metadata: Path,
    webhook: Webhook | None,
    model_name: str,
) -> Thread.Status:
    initial_prompt: str | None = None
    initial_command = await sync_to_async(
        _pop_next_thread_command,
        thread_sensitive=True,
    )(thread_id)
    match initial_command:
        case Prompt() as prompt:
            initial_prompt = prompt.text
        case Stop():
            return Thread.Status.STOPPED
        case Steer() as steer:
            initial_prompt = steer.text
        case Nop():
            ...

    stream = agent.stream(
        challenge=challenge,
        workspace=workspace,
        metadata_directory=metadata,
        webhook=webhook,
        prompt=initial_prompt,
    )
    interrupt: Interrupt = Nop()
    is_started = False
    stop_requested = False
    while True:
        try:
            if not is_started:
                event = await stream.__anext__()
                is_started = True
            else:
                event = await stream.asend(interrupt)
        except StopAsyncIteration:
            return Thread.Status.STOPPED if stop_requested else Thread.Status.WAITING

        await sync_to_async(_record_stream_event, thread_sensitive=True)(
            thread_id,
            event,
            model_name,
        )
        command = await sync_to_async(
            _pop_next_thread_command,
            thread_sensitive=True,
        )(thread_id)
        if isinstance(command, Stop):
            stop_requested = True
        interrupt = command


def _record_stream_event(thread_id: int, event: Event, model_name: str) -> None:
    thread = Thread.objects.select_related(
        "model", "credential", "credential__provider"
    ).get(pk=thread_id)
    match event:
        case Chunk() as chunk:
            if not chunk.text:
                return
            _record_event(
                thread,
                source="agent_stream",
                kind=_stream_chunk_kind(chunk.tag),
                text=chunk.text,
                raw={"tag": chunk.tag},
            )
        case Log() as log:
            if not log.text and not log.raw:
                return
            raw = dict(log.raw)
            _record_event(
                thread,
                source="agent_stream",
                kind=log.kind,
                text=log.text,
                raw=raw,
            )
            if log.kind == "token_count":
                _record_token_usage_snapshot(
                    thread,
                    model_name=model_name,
                    raw=raw,
                )
        case TokenUsage() as usage:
            raw = usage.event_raw()
            text = json.dumps(raw["usage"], separators=(",", ":"))
            _record_event(
                thread,
                source="agent_stream",
                kind="token_count",
                text=text,
                raw=raw,
            )
            _record_token_usage_snapshot(
                thread,
                model_name=usage.model or model_name,
                raw=raw,
            )
        case ItemCompleted():
            _record_event(
                thread,
                source="agent_stream",
                kind="item.terminated",
                text="",
            )
        case TurnCompleted():
            _record_event(
                thread,
                source="agent_stream",
                kind="turn.completed",
                text="",
            )
        case Nop():
            return


def _stream_chunk_kind(tag: str) -> str:
    if tag in {"thinking", "tool_input", "tool_use"}:
        return tag
    return "chunk"


def _pop_next_thread_command(thread_id: int) -> Interrupt:
    message = (
        SteeringMessage.objects.filter(thread_id=thread_id, delivered_at__isnull=True)
        .order_by("created_at")
        .first()
    )
    if message is None:
        return Nop()

    message.delivered_at = timezone.now()
    message.save(update_fields=["delivered_at", "updated_at"])
    if message.kind == SteeringMessage.Kind.STOP:
        event_kind = "stop"
        interrupt: Interrupt = Stop()
    elif message.kind == SteeringMessage.Kind.PROMPT:
        event_kind = "prompt"
        interrupt = Prompt(text=message.text)
    else:
        event_kind = "steer"
        interrupt = Steer(text=message.text)
    _record_event(
        message.thread,
        source="user",
        kind=event_kind,
        text=message.text,
        raw={"steering_message_id": message.pk},
    )
    return interrupt


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


def _record_event(
    thread: Thread,
    *,
    source: str,
    kind: str,
    text: str,
    raw: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
) -> StreamEvent:
    with transaction.atomic():
        sequence = (
            StreamEvent.objects.filter(thread=thread).aggregate(Max("sequence"))[
                "sequence__max"
            ]
            or 0
        ) + 1
        if dedupe_key is None:
            dedupe_key = f"{source}:{sequence}"
        event, _created = StreamEvent.objects.get_or_create(
            thread=thread,
            dedupe_key=dedupe_key,
            defaults={
                "sequence": sequence,
                "source": source,
                "kind": kind,
                "text": text,
                "raw": raw or {},
            },
        )
        return event


def _agent_class_path(data: dict[str, Any]) -> str:
    class_path = data.get("class", "catchy.codex.CodexAgent")
    if class_path == "CodexAgent":
        return "catchy.codex.CodexAgent"
    if not isinstance(class_path, str) or not class_path:
        raise ValueError("agent configuration has an invalid class")
    return class_path


def _import_agent_class(class_path: str) -> type[Any]:
    module_name, separator, attribute_name = class_path.rpartition(".")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            f"agent class must be a fully qualified import path: {class_path!r}"
        )
    module = importlib.import_module(module_name)
    agent_class = getattr(module, attribute_name, None)
    if not isinstance(agent_class, type):
        raise TypeError(f"agent class is not a class: {class_path!r}")
    return agent_class


def _thread_model_name(thread: Thread) -> str:
    if thread.model is not None:
        return thread.model.name
    model = thread.agent.resolved_mapping(user=thread.created_by).get("model", {})
    if isinstance(model, dict) and isinstance(model.get("name"), str):
        return str(model["name"])
    return "unknown"


def _record_token_usage_snapshot(
    thread: Thread,
    *,
    model_name: str,
    raw: dict[str, Any],
) -> None:
    snapshot = token_usage_snapshot_for_thread(
        thread,
        raw=raw,
        model_name=model_name,
    )
    if snapshot is None:
        return
    snapshot.pop("pricing", None)
    snapshot.pop("usd", None)
    cost_usd = token_usage_cost_usd(
        thread,
        raw=snapshot,
        model_name=str(snapshot.get("model") or model_name),
    )
    if cost_usd is not None:
        snapshot[THREAD_COST_USD_KEY] = str(cost_usd)
    thread.latest_cost = snapshot
    thread.save(update_fields=["latest_cost", "updated_at"])
    ThreadCostSnapshot.objects.create(
        thread=thread,
        usage=snapshot,
    )


def token_usage_snapshot_for_thread(
    thread: Thread,
    *,
    raw: dict[str, Any],
    model_name: str | None = None,
) -> dict[str, Any] | None:
    return _token_usage_snapshot(
        raw,
        model_name=model_name or _thread_model_name(thread),
        provider_slug=_provider_slug_for_thread(thread),
    )


def token_usage_cost_usd(
    thread: Thread,
    *,
    raw: dict[str, Any],
    model_name: str | None = None,
) -> Decimal | None:
    snapshot = token_usage_snapshot_for_thread(
        thread,
        raw=raw,
        model_name=model_name,
    )
    if snapshot is None:
        return None
    pricing = _model_pricing_for_snapshot(thread, snapshot)
    if pricing is None:
        return None
    return pricing.estimate_usd(snapshot)


def cached_token_usage_cost_usd(
    thread: Thread,
    *,
    model_name: str | None = None,
    persist: bool = True,
) -> Decimal | None:
    latest_cost = thread.latest_cost if isinstance(thread.latest_cost, dict) else {}
    if not latest_cost:
        return None

    cached_cost = _decimal_cost(latest_cost.get(THREAD_COST_USD_KEY))
    if cached_cost is not None:
        return cached_cost

    cost_usd = token_usage_cost_usd(
        thread,
        raw=latest_cost,
        model_name=model_name,
    )
    if cost_usd is None:
        return None
    if persist:
        cached_latest_cost = {**latest_cost, THREAD_COST_USD_KEY: str(cost_usd)}
        Thread.objects.filter(pk=thread.pk).update(latest_cost=cached_latest_cost)
        thread.latest_cost = cached_latest_cost
    return cost_usd


def _token_usage_snapshot(
    raw: dict[str, Any],
    *,
    model_name: str,
    provider_slug: str | None = None,
) -> dict[str, Any] | None:
    usage = _token_usage_from_raw(raw)
    if usage is None:
        return None
    input_tokens = _int_value(usage.get("input_tokens") or usage.get("inputTokens"))
    cached_input_tokens = _int_value(
        usage.get("cached_input_tokens") or usage.get("cachedInputTokens")
    )
    cache_creation_input_tokens = _int_value(
        usage.get("cache_creation_input_tokens")
        or usage.get("cacheCreationInputTokens")
    )
    cache_read_input_tokens = _int_value(
        usage.get("cache_read_input_tokens") or usage.get("cacheReadInputTokens")
    )
    output_tokens = _int_value(usage.get("output_tokens") or usage.get("outputTokens"))
    reasoning_output_tokens = _int_value(
        usage.get("reasoning_output_tokens") or usage.get("reasoningOutputTokens")
    )
    total_tokens = _int_value(usage.get("total_tokens") or usage.get("totalTokens"))
    if not total_tokens:
        total_tokens = (
            input_tokens
            + cache_creation_input_tokens
            + cache_read_input_tokens
            + output_tokens
        )
    provider = raw.get("provider")
    model = raw.get("model")
    return {
        "provider": provider
        if isinstance(provider, str) and provider
        else provider_slug or "openai",
        "model": model if isinstance(model, str) and model else model_name,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "total_tokens": total_tokens,
    }


def _decimal_cost(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _model_pricing_for_snapshot(
    thread: Thread,
    snapshot: dict[str, Any],
) -> ModelPricing | None:
    provider_slug = snapshot.get("provider")
    model_name = snapshot.get("model")
    if not isinstance(provider_slug, str) or not provider_slug:
        provider_slug = _provider_slug_for_thread(thread)
    if not provider_slug:
        return None

    pricing = (
        ModelPricing.objects.select_related("model", "provider")
        .filter(provider__slug=provider_slug)
        .filter(model=thread.model)
        .first()
        if thread.model_id
        else None
    )
    if pricing is not None:
        return pricing
    if not isinstance(model_name, str) or not model_name:
        return None
    return (
        ModelPricing.objects.select_related("model", "provider")
        .filter(provider__slug=provider_slug, model__name__iexact=model_name)
        .first()
    )


def _provider_slug_for_thread(thread: Thread) -> str:
    credential = thread.credential
    if credential is None:
        return ""
    provider = credential.provider
    if provider is not None:
        return provider.slug
    if credential.kind in {Credential.Kind.OPENAI, Credential.Kind.CODEX_AUTH_JSON}:
        return "openai"
    if credential.kind in {
        Credential.Kind.ANTHROPIC,
        Credential.Kind.CLAUDE_OAUTH_TOKEN,
    }:
        return "anthropic"
    return ""


def _token_usage_from_raw(raw: dict[str, Any]) -> dict[str, Any] | None:
    payload = _first_dict(raw.get("payload"), raw.get("message"), raw)
    info = _first_dict(
        payload.get("info"),
        payload.get("usage"),
        payload.get("tokenUsage"),
        raw.get("info"),
        raw.get("usage"),
        raw.get("tokenUsage"),
        payload,
    )
    usage = _first_dict(
        info.get("total_token_usage"),
        info.get("last_token_usage"),
        info.get("total"),
        info.get("last"),
        info,
    )
    if not usage:
        return None
    return {str(key): value for key, value in usage.items()}


def _first_dict(*values: object) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _fork_thread_name(thread: Thread) -> str:
    base_name = thread.name or f"thread-{thread.pk}"
    suffix = "-fork"
    max_base_length = 80 - len(suffix)
    return f"{base_name[:max_base_length]}{suffix}"


def _thread_root(thread: Thread) -> Path:
    if thread.thread_root:
        return Path(thread.thread_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return Path(settings.MEDIA_ROOT) / "threads" / f"thread-{thread.pk}-{timestamp}"
