from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID

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
    build_agent_configuration,
    cached_token_usage_cost_usd,
    fork_thread,
    start_thread,
    token_usage_cost_usd,
)


class _ChallengeGroup(TypedDict):
    challenge: Challenge
    threads: list[Thread]


class _ThreadGroup(TypedDict):
    ctf: Ctf
    challenges: list[_ChallengeGroup]
    thread_count: int


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
    threads = _attach_thread_costs(
        _attach_credential_visibility(
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
            .distinct()[:20],
            request.user,
        )
    )
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
    return render(
        request,
        "ctf/index.html",
        {
            "ctfs": ctfs,
            "threads": threads,
            "public_thread_groups": public_thread_groups,
            "public_thread_count": public_thread_count,
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
    return render(
        request,
        "ctf/credential_list.html",
        {"credentials": credentials},
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
    return render(request, "ctf/provider_list.html", {"providers": providers})


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
    return render(request, "ctf/model_list.html", {"models": models})


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
    return render(request, "ctf/pricing_list.html", {"pricing": pricing})


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
    return render(request, "ctf/agent_list.html", {"agents": agents})


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
    return render(
        request,
        "ctf/agent_detail.html",
        {"agent": agent, "resolves": resolves},
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

    return render(
        request,
        "ctf/ctf_detail.html",
        {
            "ctf": ctf,
            "challenges": ctf.challenges.all(),
            "can_init": ctf.can_init_thread(request.user),
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
    return render(
        request,
        "ctf/challenge_detail.html",
        {
            "ctf": ctf,
            "challenge": challenge,
            "threads": threads,
            "thread_form": thread_form,
            "can_init": ctf.can_init_thread(request.user),
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
    return render(
        request,
        "ctf/thread_detail.html",
        {
            "thread": thread,
            "latest_cost_usd": latest_cost_usd,
            "events": events,
            "events_json": [_event_payload(event, thread=thread) for event in events],
            "can_manage_thread": can_manage_thread,
            "can_prompt_thread": can_manage_thread
            and thread.status in promptable_statuses,
            "can_stop_thread": can_manage_thread
            and thread.status in stoppable_statuses,
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
            sequence=(
                StreamEvent.objects.filter(thread=thread)
                .order_by("-sequence")
                .values_list("sequence", flat=True)
                .first()
                or 0
            )
            + 1,
            dedupe_key=f"user:stop:{timezone.now().timestamp()}",
            source="user",
            kind="stop",
            text="",
            raw={"user_id": request.user.pk},
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
    while True:
        thread = Thread.objects.select_related(
            "agent", "model", "credential", "credential__provider", "created_by"
        ).get(pk=thread_id)
        for event in _events_after(thread_id, last_sequence):
            last_sequence = event.sequence
            yield f"id: {event.sequence}\n"
            yield "event: stream\n"
            payload = _event_payload(event, thread=thread)
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
    return StreamEvent.objects.filter(
        thread_id=thread_id, sequence__gt=sequence
    ).order_by("sequence")


def _event_payload(
    event: StreamEvent,
    *,
    thread: Thread | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "sequence": event.sequence,
        "source": event.source,
        "kind": event.kind,
        "text": event.text,
        "raw": event.raw,
        "created_at": event.created_at.isoformat(),
    }
    if event.kind == "token_count" and thread is not None:
        model_name = thread.model.name if thread.model is not None else None
        cost_usd = token_usage_cost_usd(thread, raw=event.raw, model_name=model_name)
        if cost_usd is not None:
            payload["cost_usd"] = str(cost_usd)
    return payload


def _nonnegative_int(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return max(int(value), 0)
    except ValueError:
        return 0
