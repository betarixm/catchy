from __future__ import annotations

import json
import time
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict, cast
from uuid import UUID

from catchy.core.agent.models import (
    Component,
    Delta,
    Event,
    EventRenderer,
    ItemCompleted,
    Nop,
    JsonLog,
    TextLog,
    ThreadStarted,
    TokenUsage,
    TurnCompleted,
    TurnStarted,
)
from catchy.core.agent.protocols import HandoffExporter
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import PermissionDenied
from django.db.models import Q, QuerySet
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import (
    AgentConfigurationForm,
    ChallengeForm,
    CredentialForm,
    CtfForm,
    ModelConfigurationForm,
    ModelPricingForm,
    ProviderForm,
    ThreadCreateForm,
)
from .models import (
    AgentConfiguration,
    Challenge,
    Credential,
    Ctf,
    ModelConfiguration,
    ModelPricing,
    Provider,
    SteeringMessage,
    StreamEvent,
    Thread,
)
from .services import (
    APP_EVENT_FORMAT,
    build_agent_configuration,
    cached_token_usage_cost_usd,
    fork_thread,
    start_thread,
    token_usage_cost_usd,
)


class _ChallengeGroup(TypedDict):
    challenge: Challenge
    threads: list[Thread]


class _ThreadGroup(TypedDict, total=False):
    ctf: Ctf
    challenges: list[_ChallengeGroup]
    thread_count: int
    stats: dict[str, Any]


def _bucket_init() -> dict[str, Any]:
    return {"count": 0, "cost": 0.0, "has_cost": False, "by_status": {}}


def _normalize_status(status: str) -> str:
    """Fold 'stopped' into 'completed' — they are functionally equivalent."""
    if str(status) == "stopped":
        return "completed"
    return status


def _bucket_record(bucket: dict[str, Any], status: str, cost: float | None) -> None:
    bucket["count"] += 1
    folded = _normalize_status(status)
    bucket["by_status"][folded] = bucket["by_status"].get(folded, 0) + 1
    if cost is not None:
        bucket["cost"] += cost
        bucket["has_cost"] = True


def _sorted_breakdown(bucket: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, entry in bucket.items():
        rows.append(
            {
                "label": label,
                "count": entry["count"],
                "cost": entry["cost"] if entry["has_cost"] else None,
                "completed": entry["by_status"].get(Thread.Status.COMPLETED, 0),
                "running": entry["by_status"].get(Thread.Status.RUNNING, 0),
                "failed": entry["by_status"].get(Thread.Status.FAILED, 0),
            }
        )
    rows.sort(key=lambda r: (-r["count"], r["label"]))
    return rows


def _thread_stats(threads: list[Thread]) -> dict[str, Any]:
    """Aggregate stats over threads with `latest_cost_usd` already attached.

    Returns totals plus per-dimension breakdowns (status, agent, model, provider,
    credential, ctf, challenge).
    """
    by_status: dict[str, int] = {}
    by_agent: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    by_provider: dict[str, dict[str, Any]] = {}
    by_credential: dict[str, dict[str, Any]] = {}
    by_ctf: dict[str, dict[str, Any]] = {}
    by_challenge: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    has_cost = False
    earliest: Any = None
    latest: Any = None

    for thread in threads:
        status = thread.status
        by_status[status] = by_status.get(status, 0) + 1
        cost_raw = getattr(thread, "latest_cost_usd", None)
        cost = float(cost_raw) if cost_raw is not None else None
        if cost is not None:
            total_cost += cost
            has_cost = True

        agent_name = thread.agent.name if thread.agent_id else "—"
        model_name = thread.model.name if thread.model_id else "—"
        if thread.credential_id and thread.credential is not None:
            provider_name = (
                thread.credential.provider.name
                if thread.credential.provider_id
                else "—"
            )
            credential_name = thread.credential.name
        else:
            provider_name = "—"
            credential_name = "—"
        ctf_title = thread.ctf.title
        challenge_id = thread.challenge.challenge_id

        for bucket, key in (
            (by_agent, agent_name),
            (by_model, model_name),
            (by_provider, provider_name),
            (by_credential, credential_name),
            (by_ctf, ctf_title),
            (by_challenge, challenge_id),
        ):
            entry = bucket.setdefault(key, _bucket_init())
            _bucket_record(entry, status, cost)

        if earliest is None or thread.created_at < earliest:
            earliest = thread.created_at
        if latest is None or thread.updated_at > latest:
            latest = thread.updated_at

    completed_count = by_status.get("completed", 0) + by_status.get("stopped", 0)
    return {
        "total": len(threads),
        "by_status": by_status,
        "completed": completed_count,
        "running": by_status.get("running", 0),
        "queued": by_status.get("queued", 0),
        "waiting": by_status.get("waiting", 0),
        "failed": by_status.get("failed", 0),
        "total_cost": total_cost if has_cost else None,
        "earliest": earliest,
        "latest": latest,
        "by_agent": _sorted_breakdown(by_agent),
        "by_model": _sorted_breakdown(by_model),
        "by_provider": _sorted_breakdown(by_provider),
        "by_credential": _sorted_breakdown(by_credential),
        "by_ctf": _sorted_breakdown(by_ctf),
        "by_challenge": _sorted_breakdown(by_challenge),
    }


def _thread_event_stats(
    events: list[StreamEvent], thread: Thread
) -> dict[str, Any]:
    """Aggregate per-event stats for a single thread."""
    by_format: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    earliest: Any = None
    latest: Any = None

    for event in events:
        by_format[event.format] = by_format.get(event.format, 0) + 1
        if earliest is None or event.created_at < earliest:
            earliest = event.created_at
        if latest is None or event.created_at > latest:
            latest = event.created_at

    renderers: dict[str, EventRenderer[Any]] = {}
    for event in events:
        payload = _event_payload(event, thread=thread, renderers=renderers)
        src = str(payload.get("source") or "—")
        kind = str(payload.get("kind") or "—")
        by_source[src] = by_source.get(src, 0) + 1
        by_kind[kind] = by_kind.get(kind, 0) + 1

    usage = _latest_token_usage_from_thread(thread)
    duration_seconds = None
    if earliest is not None and latest is not None:
        delta = latest - earliest
        duration_seconds = delta.total_seconds()

    return {
        "events": len(events),
        "duration_seconds": duration_seconds,
        "started_at": earliest,
        "last_event_at": latest,
        "by_format": _sorted_count_map(by_format),
        "by_source": _sorted_count_map(by_source),
        "by_kind": _sorted_count_map(by_kind),
        "token_usage": usage,
    }


def _thread_counts_by(
    queryset: QuerySet[Any], field_name: str
) -> dict[int, int]:
    """Group Thread rows by a foreign-key id and return a map of id → count."""
    counts: dict[int, int] = {}
    for row in queryset:
        pk = row.get(field_name) if isinstance(row, dict) else None
        if pk is None:
            continue
        counts[pk] = counts.get(pk, 0) + 1
    return counts


def _sorted_count_map(data: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"label": label, "count": count, "cost": None}
        for label, count in data.items()
    ]
    rows.sort(key=lambda r: (-r["count"], r["label"]))
    return rows


def _latest_token_usage_from_thread(thread: Thread) -> dict[str, Any] | None:
    raw = thread.latest_cost if isinstance(thread.latest_cost, dict) else None
    if not raw:
        return None
    return {
        "provider": raw.get("provider"),
        "model": raw.get("model"),
        "input": raw.get("input_tokens"),
        "cached_input": raw.get("cached_input_tokens"),
        "cache_creation": raw.get("cache_creation_input_tokens"),
        "cache_read": raw.get("cache_read_input_tokens"),
        "output": raw.get("output_tokens"),
        "reasoning": raw.get("reasoning_output_tokens"),
        "total": raw.get("total_tokens"),
    }


def index(request: HttpRequest) -> HttpResponse:
    ctfs = [
        ctf
        for ctf in Ctf.objects.prefetch_related("view_groups")
        if ctf.can_view(request.user)
    ]
    ctf_ids = [ctf.pk for ctf in ctfs]
    thread_filter = Q(is_public=True)
    if request.user.is_authenticated:
        thread_filter |= Q(ctf_id__in=ctf_ids)
    visible_threads_qs = (
        Thread.objects.select_related(
            "ctf",
            "challenge",
            "agent",
            "model",
            "credential",
            "credential__provider",
        )
        .prefetch_related("credential__allowed_groups", "credential__allowed_users")
        .filter(thread_filter)
        .distinct()
    )
    visible_threads = _attach_thread_costs(
        _attach_credential_visibility(list(visible_threads_qs), request.user)
    )
    threads = visible_threads[:20]
    public_thread_groups = _group_threads_by_ctf_and_challenge(
        _attach_thread_costs(
            _attach_credential_visibility(
                Thread.objects.select_related(
                    "ctf",
                    "challenge",
                    "agent",
                    "model",
                    "credential",
                    "credential__provider",
                )
                .prefetch_related(
                    "credential__allowed_groups", "credential__allowed_users"
                )
                .filter(is_public=True)[:40],
                request.user,
            )
        )
    )
    public_thread_count = sum(group["thread_count"] for group in public_thread_groups)
    stats = _thread_stats(visible_threads)
    # Per-CTF stats over the public threads of each group.
    for group in public_thread_groups:
        group_threads: list[Thread] = []
        for ch in group["challenges"]:
            group_threads.extend(ch["threads"])
        group["stats"] = _thread_stats(group_threads)
    return render(
        request,
        "ctf/index.html",
        {
            "ctfs": ctfs,
            "threads": threads,
            "public_thread_groups": public_thread_groups,
            "public_thread_count": public_thread_count,
            "thread_stats": stats,
        },
    )


@login_required
def credential_list(request: HttpRequest) -> HttpResponse:
    credentials = [
        credential
        for credential in Credential.objects.select_related(
            "provider"
        ).prefetch_related("allowed_groups", "allowed_users")
        if credential.can_view(request.user)
    ]
    usage_map = _thread_counts_by(Thread.objects.values("credential_id"), "credential_id")
    for credential in credentials:
        credential.thread_count = usage_map.get(credential.pk, 0)
    total_threads = sum(c.thread_count for c in credentials)
    return render(
        request,
        "ctf/credential_list.html",
        {
            "credentials": credentials,
            "list_stats": {"count": len(credentials), "thread_total": total_threads},
        },
    )


@login_required
def credential_create(request: HttpRequest) -> HttpResponse:
    form = CredentialForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        credential = form.save(commit=False)
        credential.created_by = request.user
        credential.save()
        form.save_m2m()
        messages.success(request, "Credential saved.")
        return redirect("ctf:credential_list")
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": "New credential"},
    )


@login_required
def credential_update(request: HttpRequest, slug: str) -> HttpResponse:
    credential = get_object_or_404(
        Credential.objects.prefetch_related("allowed_groups", "allowed_users"),
        slug=slug,
    )
    if not credential.can_view(request.user):
        raise PermissionDenied

    form = CredentialForm(request.POST or None, instance=credential)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Credential updated.")
        return redirect("ctf:credential_list")
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit credential: {credential.name}"},
    )


@login_required
def provider_list(request: HttpRequest) -> HttpResponse:
    providers = list(Provider.objects.all())
    credential_counts: dict[int, int] = {}
    for row in Credential.objects.values("provider_id"):
        pk = row.get("provider_id")
        if pk is not None:
            credential_counts[pk] = credential_counts.get(pk, 0) + 1
    pricing_counts: dict[int, int] = {}
    for row in ModelPricing.objects.values("provider_id"):
        pk = row.get("provider_id")
        if pk is not None:
            pricing_counts[pk] = pricing_counts.get(pk, 0) + 1
    for provider in providers:
        provider.credential_count = credential_counts.get(provider.pk, 0)
        provider.pricing_count = pricing_counts.get(provider.pk, 0)
    list_stats = {
        "count": len(providers),
        "credentials_total": sum(credential_counts.values()),
        "pricing_total": sum(pricing_counts.values()),
    }
    return render(
        request,
        "ctf/provider_list.html",
        {"providers": providers, "list_stats": list_stats},
    )


@login_required
def provider_create(request: HttpRequest) -> HttpResponse:
    form = ProviderForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Provider saved.")
        return redirect("ctf:provider_list")
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": "New provider"},
    )


@login_required
def provider_update(request: HttpRequest, slug: str) -> HttpResponse:
    provider = get_object_or_404(Provider, slug=slug)
    form = ProviderForm(request.POST or None, instance=provider)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Provider updated.")
        return redirect("ctf:provider_list")
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit provider: {provider.name}"},
    )


@login_required
def model_list(request: HttpRequest) -> HttpResponse:
    models = [
        model
        for model in ModelConfiguration.objects.prefetch_related(
            "view_groups", "use_groups"
        )
        if model.can_view(request.user)
    ]
    usage_map = _thread_counts_by(Thread.objects.values("model_id"), "model_id")
    pricing_map: dict[int, int] = {}
    for row in ModelPricing.objects.values("model_id"):
        pk = row.get("model_id")
        if pk is not None:
            pricing_map[pk] = pricing_map.get(pk, 0) + 1
    for model in models:
        model.thread_count = usage_map.get(model.pk, 0)
        model.pricing_count = pricing_map.get(model.pk, 0)
    list_stats = {
        "count": len(models),
        "thread_total": sum(m.thread_count for m in models),
        "priced": sum(1 for m in models if m.pricing_count),
    }
    return render(
        request, "ctf/model_list.html", {"models": models, "list_stats": list_stats}
    )


@login_required
def model_create(request: HttpRequest) -> HttpResponse:
    form = ModelConfigurationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        model = form.save(commit=False)
        model.created_by = request.user
        model.save()
        form.save_m2m()
        messages.success(request, "Model saved.")
        return redirect("ctf:model_list")
    return render(request, "ctf/form.html", {"form": form, "title": "New model"})


@login_required
def model_update(request: HttpRequest, slug: str) -> HttpResponse:
    model = get_object_or_404(
        ModelConfiguration.objects.prefetch_related("view_groups", "use_groups"),
        slug=slug,
    )
    if not model.can_view(request.user):
        raise PermissionDenied

    form = ModelConfigurationForm(request.POST or None, instance=model)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Model updated.")
        return redirect("ctf:model_list")
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit model: {model.name}"},
    )


@login_required
def pricing_list(request: HttpRequest) -> HttpResponse:
    pricing = [
        item
        for item in ModelPricing.objects.select_related(
            "model", "provider"
        ).prefetch_related("model__view_groups")
        if item.model.can_view(request.user)
    ]
    providers_seen = {item.provider_id for item in pricing if item.provider_id}
    models_seen = {item.model_id for item in pricing if item.model_id}
    list_stats = {
        "count": len(pricing),
        "providers": len(providers_seen),
        "models": len(models_seen),
    }
    return render(
        request,
        "ctf/pricing_list.html",
        {"pricing": pricing, "list_stats": list_stats},
    )


@login_required
def pricing_create(request: HttpRequest) -> HttpResponse:
    form = ModelPricingForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Model pricing saved.")
        return redirect("ctf:pricing_list")
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": "New model pricing"},
    )


@login_required
def pricing_update(request: HttpRequest, pk: int) -> HttpResponse:
    pricing = get_object_or_404(
        ModelPricing.objects.select_related("model", "provider").prefetch_related(
            "model__view_groups"
        ),
        pk=pk,
    )
    if not pricing.model.can_view(request.user):
        raise PermissionDenied

    form = ModelPricingForm(request.POST or None, instance=pricing)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Model pricing updated.")
        return redirect("ctf:pricing_list")
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit pricing: {pricing}"},
    )


@login_required
def agent_list(request: HttpRequest) -> HttpResponse:
    agents = [
        agent
        for agent in AgentConfiguration.objects.prefetch_related(
            "view_groups", "use_groups"
        )
        if agent.can_view(request.user)
    ]
    usage_map = _thread_counts_by(Thread.objects.values("agent_id"), "agent_id")
    for agent in agents:
        agent.thread_count = usage_map.get(agent.pk, 0)
    list_stats = {
        "count": len(agents),
        "thread_total": sum(a.thread_count for a in agents),
        "active": sum(1 for a in agents if a.thread_count),
    }
    return render(
        request, "ctf/agent_list.html", {"agents": agents, "list_stats": list_stats}
    )


@login_required
def agent_create(request: HttpRequest) -> HttpResponse:
    form = AgentConfigurationForm(request.POST or None, user=request.user)
    if request.method == "POST" and form.is_valid():
        agent = form.save(commit=False)
        agent.created_by = request.user
        agent.save()
        form.save_m2m()
        messages.success(request, "Agent configuration saved.")
        return redirect(agent)
    return render(request, "ctf/form.html", {"form": form, "title": "New agent"})


@login_required
def agent_update(request: HttpRequest, slug: str) -> HttpResponse:
    agent = get_object_or_404(AgentConfiguration, slug=slug)
    if not agent.can_view(request.user):
        raise PermissionDenied

    form = AgentConfigurationForm(
        request.POST or None, instance=agent, user=request.user
    )
    if request.method == "POST" and form.is_valid():
        agent = form.save()
        messages.success(request, "Agent configuration updated.")
        return redirect(agent)
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit agent: {agent.name}"},
    )


@login_required
def agent_detail(request: HttpRequest, slug: str) -> HttpResponse:
    agent = get_object_or_404(AgentConfiguration, slug=slug)
    if not agent.can_view(request.user):
        raise PermissionDenied
    resolves = False
    try:
        agent.resolved_mapping(user=request.user)
        resolves = True
    except Exception as exc:
        messages.error(request, f"Could not resolve YAML: {exc}")
    agent_threads = _attach_thread_costs(
        Thread.objects.filter(agent=agent).select_related("model")
    )
    stats = _thread_stats(agent_threads)
    return render(
        request,
        "ctf/agent_detail.html",
        {"agent": agent, "resolves": resolves, "thread_stats": stats},
    )


@login_required
def ctf_create(request: HttpRequest) -> HttpResponse:
    form = CtfForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        ctf = form.save(commit=False)
        ctf.created_by = request.user
        ctf.save()
        form.save_m2m()
        messages.success(request, "CTF saved.")
        return redirect(ctf)
    return render(request, "ctf/form.html", {"form": form, "title": "New CTF"})


@login_required
def ctf_update(request: HttpRequest, slug: str) -> HttpResponse:
    ctf = get_object_or_404(
        Ctf.objects.prefetch_related("view_groups", "init_groups"), slug=slug
    )
    if not ctf.can_init_thread(request.user):
        raise PermissionDenied
    form = CtfForm(request.POST or None, instance=ctf)
    if request.method == "POST" and form.is_valid():
        ctf = form.save()
        messages.success(request, "CTF updated.")
        return redirect(ctf)
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit CTF: {ctf.title}"},
    )


@login_required
def ctf_detail(request: HttpRequest, slug: str) -> HttpResponse:
    ctf = get_object_or_404(
        Ctf.objects.prefetch_related("view_groups", "init_groups"), slug=slug
    )
    if not ctf.can_view(request.user):
        raise PermissionDenied

    challenges = list(ctf.challenges.all())
    ctf_threads = _attach_thread_costs(
        Thread.objects.filter(ctf=ctf).select_related("model")
    )
    stats = _thread_stats(ctf_threads)
    stats["challenge_count"] = len(challenges)
    return render(
        request,
        "ctf/ctf_detail.html",
        {
            "ctf": ctf,
            "challenges": challenges,
            "can_init": ctf.can_init_thread(request.user),
            "thread_stats": stats,
        },
    )


@login_required
def challenge_create(request: HttpRequest, ctf_slug: str) -> HttpResponse:
    ctf = get_object_or_404(Ctf, slug=ctf_slug)
    if not ctf.can_init_thread(request.user):
        raise PermissionDenied
    form = ChallengeForm(request.POST or None, request.FILES or None, ctf=ctf)
    if request.method == "POST" and form.is_valid():
        challenge = form.save(commit=False)
        challenge.ctf = ctf
        challenge.created_by = request.user
        challenge.save()
        messages.success(request, "Challenge saved.")
        return redirect(ctf)
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"New challenge for {ctf.title}"},
    )


@login_required
def challenge_update(
    request: HttpRequest, ctf_slug: str, challenge_id: str
) -> HttpResponse:
    challenge = get_object_or_404(
        Challenge.objects.select_related("ctf").prefetch_related("ctf__init_groups"),
        ctf__slug=ctf_slug,
        challenge_id=challenge_id,
    )
    if not challenge.ctf.can_init_thread(request.user):
        raise PermissionDenied
    form = ChallengeForm(
        request.POST or None,
        request.FILES or None,
        instance=challenge,
    )
    if request.method == "POST" and form.is_valid():
        challenge = form.save()
        messages.success(request, "Challenge updated.")
        return redirect(challenge)
    return render(
        request,
        "ctf/form.html",
        {"form": form, "title": f"Edit challenge: {challenge.challenge_id}"},
    )


@login_required
def challenge_detail(
    request: HttpRequest, ctf_slug: str, challenge_id: str
) -> HttpResponse:
    challenge = get_object_or_404(
        Challenge.objects.select_related("ctf").prefetch_related(
            "ctf__view_groups", "ctf__init_groups"
        ),
        ctf__slug=ctf_slug,
        challenge_id=challenge_id,
    )
    ctf = challenge.ctf
    if not ctf.can_view(request.user):
        raise PermissionDenied

    thread_form = ThreadCreateForm(user=request.user)
    threads = _attach_thread_costs(
        _attach_credential_visibility(
            challenge.threads.select_related(
                "agent", "model", "credential", "credential__provider"
            ).prefetch_related(
                "credential__allowed_groups", "credential__allowed_users"
            ),
            request.user,
        )
    )
    stats = _thread_stats(threads)
    return render(
        request,
        "ctf/challenge_detail.html",
        {
            "ctf": ctf,
            "challenge": challenge,
            "threads": threads,
            "thread_form": thread_form,
            "can_init": ctf.can_init_thread(request.user),
            "thread_stats": stats,
        },
    )


@login_required
@require_POST
def thread_create(
    request: HttpRequest, ctf_slug: str, challenge_id: str
) -> HttpResponse:
    ctf = get_object_or_404(Ctf.objects.prefetch_related("init_groups"), slug=ctf_slug)
    if not ctf.can_init_thread(request.user):
        raise PermissionDenied
    challenge = get_object_or_404(Challenge, ctf=ctf, challenge_id=challenge_id)

    form = ThreadCreateForm(request.POST, user=request.user)
    if not form.is_valid():
        messages.error(request, "Could not start thread.")
        return redirect(challenge)

    agent = form.cleaned_data["agent"]
    if not agent.can_use(request.user):
        raise PermissionDenied
    model = form.cleaned_data["model"]
    if not model.can_use(request.user):
        raise PermissionDenied
    credential = form.cleaned_data["credential"]
    if not credential.can_use(request.user):
        raise PermissionDenied

    try:
        build_agent_configuration(
            agent,
            model_configuration=model,
            credential=credential,
            user=request.user,
        )
    except PermissionDenied:
        raise
    except Exception as exc:
        messages.error(request, f"Could not resolve agent configuration: {exc}")
        return redirect(challenge)

    thread = Thread.objects.create(
        ctf=ctf,
        challenge=challenge,
        agent=agent,
        model=model,
        credential=credential,
        created_by=request.user,
        name=form.cleaned_data["name"],
    )
    start_thread(thread)
    messages.success(request, "Thread queued.")
    return redirect(thread)


def thread_detail(request: HttpRequest, thread_uuid: UUID) -> HttpResponse:
    thread = get_object_or_404(
        Thread.objects.select_related(
            "ctf", "challenge", "agent", "model", "credential", "credential__provider"
        ).prefetch_related("credential__allowed_groups", "credential__allowed_users"),
        uuid=thread_uuid,
    )
    can_manage_thread = thread.can_interact(request.user)
    _attach_credential_visibility([thread], request.user)
    promptable_statuses = {
        Thread.Status.QUEUED,
        Thread.Status.RUNNING,
        Thread.Status.WAITING,
        Thread.Status.COMPLETED,
    }
    stoppable_statuses = {
        Thread.Status.QUEUED,
        Thread.Status.RUNNING,
        Thread.Status.WAITING,
        Thread.Status.COMPLETED,
        Thread.Status.FAILED,
    }
    events = list(thread.events.all()[:2000])
    model_name = thread.model.name if thread.model is not None else None
    latest_cost_usd = cached_token_usage_cost_usd(thread, model_name=model_name)
    thread_event_stats = _thread_event_stats(events, thread)
    return render(
        request,
        "ctf/thread_detail.html",
        {
            "thread": thread,
            "latest_cost_usd": latest_cost_usd,
            "events": events,
            "events_json": _event_payloads(events, thread=thread),
            "can_manage_thread": can_manage_thread,
            "can_prompt_thread": can_manage_thread
            and thread.status in promptable_statuses,
            "can_stop_thread": can_manage_thread
            and thread.status in stoppable_statuses,
            "event_stats": thread_event_stats,
        },
    )


@login_required
@require_POST
def thread_publish(request: HttpRequest, thread_uuid: UUID) -> HttpResponse:
    thread = get_object_or_404(Thread.objects.select_related("ctf"), uuid=thread_uuid)
    if not thread.can_publish(request.user):
        raise PermissionDenied

    thread.is_public = request.POST.get("is_public") == "1"
    thread.save(update_fields=["is_public", "updated_at"])
    messages.success(
        request,
        "Thread published." if thread.is_public else "Thread unpublished.",
    )
    return redirect(thread)


@login_required
@require_POST
def thread_steer(request: HttpRequest, thread_uuid: UUID) -> HttpResponse:
    thread = get_object_or_404(Thread.objects.select_related("ctf"), uuid=thread_uuid)
    if not thread.can_interact(request.user):
        raise PermissionDenied

    text = request.POST.get("text", "").strip()
    if not text:
        messages.error(request, "Message cannot be empty.")
        return redirect(thread)
    active_statuses = {Thread.Status.RUNNING}
    prompt_statuses = {
        Thread.Status.QUEUED,
        Thread.Status.WAITING,
        Thread.Status.COMPLETED,
    }
    if thread.status in active_statuses:
        kind = SteeringMessage.Kind.STEER
        should_resume = False
    elif thread.status == Thread.Status.QUEUED:
        kind = SteeringMessage.Kind.PROMPT
        should_resume = False
    elif thread.status in prompt_statuses:
        kind = SteeringMessage.Kind.PROMPT
        should_resume = True
    else:
        messages.error(request, "This thread cannot receive messages.")
        return redirect(thread)

    SteeringMessage.objects.create(
        thread=thread,
        created_by=request.user,
        kind=kind,
        text=text,
    )
    if should_resume:
        Thread.objects.filter(pk=thread.pk, status__in=prompt_statuses).update(
            status=Thread.Status.QUEUED,
            error="",
            updated_at=timezone.now(),
        )
        thread.status = Thread.Status.QUEUED
        thread.error = ""
        start_thread(thread)
        messages.success(request, "Prompt queued; thread is resuming.")
    else:
        messages.success(request, "Steer message queued.")
    return redirect(thread)


@login_required
@require_POST
def thread_stop(request: HttpRequest, thread_uuid: UUID) -> HttpResponse:
    thread = get_object_or_404(Thread.objects.select_related("ctf"), uuid=thread_uuid)
    if not thread.can_interact(request.user):
        raise PermissionDenied

    active_statuses = {Thread.Status.QUEUED, Thread.Status.RUNNING}
    if thread.status in active_statuses:
        SteeringMessage.objects.create(
            thread=thread,
            created_by=request.user,
            kind=SteeringMessage.Kind.STOP,
        )
        messages.success(request, "Stop queued.")
        return redirect(thread)

    if thread.status in {
        Thread.Status.WAITING,
        Thread.Status.COMPLETED,
        Thread.Status.FAILED,
    }:
        thread.status = Thread.Status.STOPPED
        thread.save(update_fields=["status", "updated_at"])
        StreamEvent.objects.create(
            thread=thread,
            format=APP_EVENT_FORMAT,
            raw=json.dumps(
                {
                    "source": "user",
                    "kind": "stop",
                    "text": "",
                    "raw": {"user_id": request.user.pk},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        messages.success(request, "Thread stopped.")
        return redirect(thread)

    messages.error(request, "This thread cannot be stopped.")
    return redirect(thread)


@login_required
@require_POST
def thread_fork(request: HttpRequest, thread_uuid: UUID) -> HttpResponse:
    thread = get_object_or_404(
        Thread.objects.select_related(
            "ctf", "challenge", "agent", "model", "credential", "created_by"
        ),
        uuid=thread_uuid,
    )
    if not thread.can_interact(request.user):
        raise PermissionDenied
    if thread.credential is not None and not thread.credential.can_use(request.user):
        raise PermissionDenied

    fork = fork_thread(thread, user=request.user)
    messages.success(request, "Thread forked.")
    return redirect(fork)


def thread_stream(request: HttpRequest, thread_uuid: UUID) -> HttpResponse:
    thread = get_object_or_404(Thread.objects.select_related("ctf"), uuid=thread_uuid)
    last_sequence = max(
        _nonnegative_int(request.GET.get("after")),
        _nonnegative_int(request.headers.get("Last-Event-ID")),
    )
    response = StreamingHttpResponse(
        _event_stream(thread.pk, last_sequence),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def thread_handoff_markdown(request: HttpRequest, thread_uuid: UUID) -> HttpResponse:
    thread = get_object_or_404(
        Thread.objects.select_related(
            "ctf", "challenge", "agent", "model", "credential", "credential__provider"
        ),
        uuid=thread_uuid,
    )
    if not thread.can_view(request.user):
        if not request.user.is_authenticated:
            return redirect_to_login(request.get_full_path())
        raise PermissionDenied

    markdown = _thread_handoff_markdown(thread)
    return HttpResponse(markdown, content_type="text/plain; charset=utf-8")


def thread_filetree(request: HttpRequest, thread_uuid: UUID) -> HttpResponse:
    thread = get_object_or_404(Thread.objects.select_related("ctf"), uuid=thread_uuid)
    if not thread.can_view(request.user):
        if not request.user.is_authenticated:
            return redirect_to_login(request.get_full_path())
        raise PermissionDenied

    # The thread root also contains runtime internals; the viewer should expose
    # only the editable workspace subtree that the agent actually works in.
    root_path = thread.workspace_path
    if not root_path and thread.thread_root:
        root_path = str(Path(thread.thread_root) / "workspace")
    if not root_path:
        return JsonResponse({"error": "no workspace for thread"}, status=404)

    root = Path(root_path)
    if not root.exists():
        return JsonResponse({"error": "workspace not found"}, status=404)

    def build_node(
        p: Path,
        base: Path,
        depth: int = 6,
        entries_limit: int = 2000,
        counter: dict[str, int] | None = None,
    ):
        if counter is None:
            counter = {"n": 0}
        name = p.name
        rel = str(p.relative_to(base))
        node: dict[str, Any] = {
            "name": name,
            "path": rel,
            "type": "dir" if p.is_dir() else "file",
        }
        if p.is_dir() and depth > 0 and counter["n"] < entries_limit:
            children = []
            try:
                for child in sorted(
                    p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
                ):
                    counter["n"] += 1
                    if counter["n"] > entries_limit:
                        break
                    children.append(
                        build_node(child, base, depth - 1, entries_limit, counter)
                    )
            except PermissionError:
                pass
            node["children"] = children
        return node

    tree = build_node(root, root)
    return JsonResponse(tree)


def _event_stream(thread_id: int, last_sequence: int = 0) -> Iterator[str]:
    renderers: dict[str, EventRenderer[Any]] = {}
    if last_sequence:
        thread = Thread.objects.select_related("model").get(pk=thread_id)
        for event in StreamEvent.objects.filter(
            thread_id=thread_id, id__lte=last_sequence
        ).order_by("id"):
            _components_from_event(event, thread=thread, renderers=renderers)

    while True:
        thread = Thread.objects.select_related(
            "agent", "model", "credential", "credential__provider", "created_by"
        ).get(pk=thread_id)
        for event in _events_after(thread_id, last_sequence):
            last_sequence = event.pk or last_sequence
            yield f"id: {event.pk}\n"
            yield "event: stream\n"
            payload = _event_payload(event, thread=thread, renderers=renderers)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        if thread.status in {
            Thread.Status.WAITING,
            Thread.Status.STOPPED,
            Thread.Status.COMPLETED,
            Thread.Status.FAILED,
        }:
            yield "retry: 5000\n"
            yield "event: status\n"
            yield f"data: {json.dumps({'status': thread.status, 'error': thread.error}, ensure_ascii=False)}\n\n"
            return
        time.sleep(1)


def _group_threads_by_ctf_and_challenge(
    threads: QuerySet[Thread] | list[Thread],
) -> list[_ThreadGroup]:
    groups: list[_ThreadGroup] = []
    group_by_ctf_id: dict[int, _ThreadGroup] = {}
    challenge_index: dict[tuple[int, int], _ChallengeGroup] = {}
    for thread in threads:
        ctf_group = group_by_ctf_id.get(thread.ctf_id)
        if ctf_group is None:
            ctf_group = {"ctf": thread.ctf, "challenges": [], "thread_count": 0}
            group_by_ctf_id[thread.ctf_id] = ctf_group
            groups.append(ctf_group)
        ch_key = (thread.ctf_id, thread.challenge_id)
        challenge_group = challenge_index.get(ch_key)
        if challenge_group is None:
            challenge_group = {"challenge": thread.challenge, "threads": []}
            challenge_index[ch_key] = challenge_group
            ctf_group["challenges"].append(challenge_group)
        challenge_group["threads"].append(thread)
        ctf_group["thread_count"] += 1
    return groups


def _attach_credential_visibility(
    threads: QuerySet[Thread] | list[Thread],
    user: Any,
) -> list[Thread]:
    marked_threads = list(threads)
    for thread in marked_threads:
        credential = thread.credential
        thread.can_show_credential = credential is not None and credential.can_view(
            user
        )
    return marked_threads


def _attach_thread_costs(threads: QuerySet[Thread] | list[Thread]) -> list[Thread]:
    marked_threads = list(threads)
    for thread in marked_threads:
        model_name = thread.model.name if thread.model is not None else None
        thread.latest_cost_usd = cached_token_usage_cost_usd(
            thread,
            model_name=model_name,
        )
    return marked_threads


def _events_after(thread_id: int, sequence: int) -> QuerySet[StreamEvent]:
    return StreamEvent.objects.filter(thread_id=thread_id, id__gt=sequence).order_by(
        "id"
    )


def _event_payloads(
    events: list[StreamEvent],
    *,
    thread: Thread,
) -> list[dict[str, object]]:
    renderers: dict[str, EventRenderer[Any]] = {}
    return [
        _event_payload(event, thread=thread, renderers=renderers) for event in events
    ]


def _event_payload(
    event: StreamEvent,
    *,
    thread: Thread | None = None,
    renderers: dict[str, EventRenderer[Any]] | None = None,
) -> dict[str, object]:
    if event.format == APP_EVENT_FORMAT:
        return _app_event_payload(event, thread=thread)

    components = _components_from_event(event, thread=thread, renderers=renderers)
    if not components and event.format == "codex-notification":
        if payload := _codex_payload_from_raw(event, thread=thread):
            return payload
    component_payloads = [
        payload
        for component in components
        if (payload := _component_payload(event, component, thread=thread)) is not None
    ]
    if component_payloads:
        payload = dict(component_payloads[0])
    else:
        payload = {
            "sequence": event.pk,
            "source": "agent_stream",
            "kind": event.format,
            "text": "",
            "raw": {},
            "format": event.format,
            "created_at": event.created_at.isoformat(),
        }
    if len(component_payloads) > 1:
        payload["components"] = component_payloads
    return payload


def _app_event_payload(
    event: StreamEvent,
    *,
    thread: Thread | None = None,
) -> dict[str, object]:
    try:
        app_raw = json.loads(event.raw) if event.raw else {}
    except json.JSONDecodeError:
        app_raw = {}
    if not isinstance(app_raw, dict):
        app_raw = {}
    raw = app_raw.get("raw")
    raw_dict = raw if isinstance(raw, dict) else {}
    kind = app_raw.get("kind")
    payload: dict[str, object] = {
        "sequence": event.pk,
        "source": app_raw.get("source")
        if isinstance(app_raw.get("source"), str)
        else "",
        "kind": kind if isinstance(kind, str) else event.format,
        "text": app_raw.get("text") if isinstance(app_raw.get("text"), str) else "",
        "raw": raw_dict,
        "format": event.format,
        "created_at": event.created_at.isoformat(),
    }
    if kind == "token_count" and thread is not None:
        model_name = thread.model.name if thread.model is not None else None
        cost_usd = token_usage_cost_usd(thread, raw=raw_dict, model_name=model_name)
        if cost_usd is not None:
            payload["cost_usd"] = str(cost_usd)
    return payload


def _components_from_event(
    event: StreamEvent,
    *,
    thread: Thread | None = None,
    renderers: dict[str, EventRenderer[Any]] | None = None,
) -> list[Component]:
    renderer = _event_renderer(event, thread=thread, renderers=renderers)
    if renderer is None:
        return []
    raw_event = _raw_event(event)
    if raw_event is None:
        return []
    try:
        return list(renderer.render(raw_event))
    except Exception:
        return []


def _event_renderer(
    event: StreamEvent,
    *,
    thread: Thread | None,
    renderers: dict[str, EventRenderer[Any]] | None,
) -> EventRenderer[Any] | None:
    if renderers is not None and event.format in renderers:
        return renderers[event.format]
    model_name = thread.model.name if thread is not None and thread.model else None
    try:
        if event.format == "codex-notification":
            from catchy.codex import CodexEventRenderer

            renderer: EventRenderer[Any] = CodexEventRenderer(model_name=model_name)
        elif event.format == "claude-code-message":
            from catchy.claude_code import ClaudeCodeEventRenderer

            renderer = ClaudeCodeEventRenderer(model_name=model_name)
        else:
            return None
    except Exception:
        return None
    if renderers is not None:
        renderers[event.format] = renderer
    return renderer


def _handoff_exporter(
    event: StreamEvent,
    *,
    thread: Thread | None,
    exporters: dict[str, HandoffExporter[Any]] | None,
) -> HandoffExporter[Any] | None:
    if exporters is not None and event.format in exporters:
        return exporters[event.format]
    model_name = thread.model.name if thread is not None and thread.model else None
    try:
        if event.format == "codex-notification":
            from catchy.codex import CodexHandoffExporter

            exporter: HandoffExporter[Any] = CodexHandoffExporter(model_name=model_name)
        elif event.format == "claude-code-message":
            from catchy.claude_code import ClaudeCodeHandoffExporter

            exporter = ClaudeCodeHandoffExporter(model_name=model_name)
        else:
            return None
    except Exception:
        return None
    if exporters is not None:
        exporters[event.format] = exporter
    return exporter


def _raw_event(event: StreamEvent) -> Event[Any] | None:
    try:
        if event.format == "codex-notification":
            from catchy.codex import CodexEvent

            return cast(Event[Any], CodexEvent(raw=event.raw))
        if event.format == "claude-code-message":
            from catchy.claude_code import ClaudeCodeEvent

            return cast(Event[Any], ClaudeCodeEvent(raw=event.raw))
        return None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _thread_handoff_markdown(thread: Thread) -> str:
    lines = [
        "# Thread Handoff",
        "",
        f"- Thread: {thread}",
        f"- UUID: {thread.uuid}",
        f"- CTF: {thread.ctf.slug}",
        f"- Challenge: {thread.challenge.challenge_id}",
        f"- Agent: {thread.agent.slug}",
        f"- Model: {thread.model.name if thread.model is not None else 'unknown'}",
        f"- Status: {thread.status}",
        "",
        "## Transcript",
        "",
    ]

    exporters: dict[str, HandoffExporter[Any]] = {}
    for event in thread.events.order_by("id"):
        event_markdown = _event_handoff_markdown(
            event,
            thread=thread,
            exporters=exporters,
        )
        if not event_markdown:
            continue
        lines.append(event_markdown.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def _event_handoff_markdown(
    event: StreamEvent,
    *,
    thread: Thread,
    exporters: dict[str, HandoffExporter[Any]] | None = None,
) -> str:
    if event.format == APP_EVENT_FORMAT:
        return _app_event_handoff_markdown(event)
    exporter = _handoff_exporter(event, thread=thread, exporters=exporters)
    raw_event = _raw_event(event)
    if exporter is None or raw_event is None:
        return ""
    return exporter.export(raw_event)


def _app_event_handoff_markdown(event: StreamEvent) -> str:
    payload = _app_event_payload(event)
    source = str(payload.get("source") or "")
    kind = str(payload.get("kind") or "")
    text = str(payload.get("text") or "")
    if source == "user" and kind in {"prompt", "steer"} and text:
        return f"- **User:** {text.strip()}\n"
    if kind in {"thread.failed", "thread.completed", "thread.stopped"} and text:
        return f"- **System:** {text.strip()}\n"
    return ""


def _component_payload(
    event: StreamEvent,
    component: Component,
    *,
    thread: Thread | None = None,
) -> dict[str, object] | None:
    source = "agent_stream"
    kind = ""
    text = ""
    raw: dict[str, object] = {}
    match component:
        case TextLog() as log:
            kind = log.tag
            text = log.text
        case JsonLog() as log:
            kind = log.tag
            text = json.dumps(
                log.data,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            parsed = json.loads(text)
            raw = parsed if isinstance(parsed, dict) else {"value": parsed}
        case TokenUsage() as usage:
            kind = "token_count"
            source = "agent_stream"
            usage_source = (
                "thread_token_usage_updated"
                if event.format == "codex-notification"
                else "stream_token_usage_updated"
            )
            provider = (
                thread.credential.provider.slug
                if (
                    thread is not None
                    and thread.credential is not None
                    and thread.credential.provider is not None
                )
                else ("anthropic" if event.format == "claude-code-message" else "openai")
            )
            model_name = thread.model.name if thread is not None and thread.model else ""
            event_raw: dict[str, object]
            try:
                decoded = json.loads(event.raw) if event.raw else {}
            except json.JSONDecodeError:
                decoded = {}
            event_raw = decoded if isinstance(decoded, dict) else {}
            raw = {
                "provider": provider,
                "model": model_name,
                "source": usage_source,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": usage.cached_input_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_output_tokens": usage.reasoning_output_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "raw": event_raw,
            }
            text = json.dumps(raw["usage"], separators=(",", ":"))
        case Delta() as delta:
            if not delta.text:
                return None
            if delta.tag == "tool_input":
                kind = "tool_input"
                raw = {"tag": "tool_input"}
            elif delta.tag == "thinking":
                kind = "chunk"
                raw = {"tag": "thinking"}
            elif delta.tag == "observation":
                kind = "chunk"
                raw = {"tag": "observation"}
            elif delta.tag == "user":
                kind = "chunk"
                raw = {"tag": "user"}
            else:
                kind = "chunk"
                raw = {"tag": "action"}
            text = delta.text
        case ItemCompleted():
            kind = "item.terminated"
        case TurnCompleted():
            kind = "turn.completed"
        case TurnStarted():
            kind = "turn.started"
        case ThreadStarted():
            kind = "thread.started"
        case Nop():
            return None

    payload: dict[str, object] = {
        "sequence": event.pk,
        "source": source,
        "kind": kind,
        "text": text,
        "raw": raw,
        "format": event.format,
        "created_at": event.created_at.isoformat(),
    }
    if kind == "token_count" and thread is not None:
        model_name = thread.model.name if thread.model is not None else None
        cost_usd = token_usage_cost_usd(thread, raw=raw, model_name=model_name)
        if cost_usd is not None:
            payload["cost_usd"] = str(cost_usd)
    return payload


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _nonnegative_int(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return max(int(value), 0)
    except ValueError:
        return 0


def _codex_payload_from_raw(
    event: StreamEvent,
    *,
    thread: Thread | None = None,
) -> dict[str, object] | None:
    try:
        raw = json.loads(event.raw) if event.raw else {}
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    method = raw.get("method")
    payload = raw.get("payload")
    if not isinstance(method, str) or not isinstance(payload, dict):
        return None

    source = "agent_stream"
    kind = method
    text = ""
    item_raw: dict[str, object] = {}

    if method == "item/agentMessage/delta":
        kind = "chunk"
        text = str(payload.get("delta") or "")
        item_raw = {"tag": "action"}
    elif method == "item/reasoning/textDelta":
        kind = "thinking"
        text = str(payload.get("delta") or "")
        item_raw = {"tag": "thinking"}
    elif method == "item/commandExecution/outputDelta":
        kind = "chunk"
        text = str(payload.get("delta") or "")
        item_raw = {"tag": "observation"}
    elif method == "item/completed":
        kind = "item.terminated"
    elif method == "turn/completed":
        kind = "turn.completed"
    elif method == "item/started":
        kind = "tool_use"
        item = payload.get("item")
        if isinstance(item, dict):
            text = json.dumps(item, ensure_ascii=False, sort_keys=True)
        item_raw = {"tag": "tool_use"}
    elif method == "thread/tokenUsage/updated":
        kind = "token_count"
        token_usage = payload.get("tokenUsage")
        usage = token_usage.get("last") if isinstance(token_usage, dict) else {}
        usage_dict = usage if isinstance(usage, dict) else {}
        parsed_usage = {
            "input_tokens": _nonnegative_int(str(usage_dict.get("inputTokens", 0))),
            "cached_input_tokens": _nonnegative_int(
                str(usage_dict.get("cachedInputTokens", 0))
            ),
            "output_tokens": _nonnegative_int(str(usage_dict.get("outputTokens", 0))),
            "reasoning_output_tokens": _nonnegative_int(
                str(usage_dict.get("reasoningOutputTokens", 0))
            ),
            "total_tokens": _nonnegative_int(str(usage_dict.get("totalTokens", 0))),
        }
        model_name = thread.model.name if thread is not None and thread.model else ""
        provider = (
            thread.credential.provider.slug
            if (
                thread is not None
                and thread.credential is not None
                and thread.credential.provider is not None
            )
            else "openai"
        )
        item_raw = {
            "provider": provider,
            "model": model_name,
            "source": "thread_token_usage_updated",
            "usage": parsed_usage,
            "raw": raw,
        }
        text = json.dumps(parsed_usage, separators=(",", ":"))

    result: dict[str, object] = {
        "sequence": event.pk,
        "source": source,
        "kind": kind,
        "text": text,
        "raw": item_raw,
        "format": event.format,
        "created_at": event.created_at.isoformat(),
    }
    if kind == "token_count" and thread is not None:
        model_name = thread.model.name if thread.model is not None else None
        cost_usd = token_usage_cost_usd(thread, raw=item_raw, model_name=model_name)
        if cost_usd is not None:
            result["cost_usd"] = str(cost_usd)
    return result
