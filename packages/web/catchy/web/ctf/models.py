from __future__ import annotations

import secrets
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from django.conf import settings as django_settings
from django.contrib.auth.models import Group
from django.core.exceptions import PermissionDenied
from django.db import models
from django.urls import reverse
from django.utils.text import slugify
from django.utils import timezone
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser

_credential_resolver_user: ContextVar[Any | None] = ContextVar(
    "catchy_credential_resolver_user",
    default=None,
)

THREAD_NAME_ADJECTIVES = [
    "agile",
    "bright",
    "calm",
    "clever",
    "curious",
    "eager",
    "electric",
    "fearless",
    "gentle",
    "golden",
    "hidden",
    "lucid",
    "nimble",
    "quiet",
    "rapid",
    "sharp",
    "steady",
    "vivid",
]
THREAD_NAME_NOUNS = [
    "beacon",
    "cipher",
    "comet",
    "delta",
    "ember",
    "engine",
    "harbor",
    "key",
    "lantern",
    "matrix",
    "orbit",
    "packet",
    "puzzle",
    "signal",
    "vector",
    "waypoint",
]


def generate_thread_name() -> str:
    adjective = secrets.choice(THREAD_NAME_ADJECTIVES)
    noun = secrets.choice(THREAD_NAME_NOUNS)
    suffix = secrets.randbelow(10_000)
    return f"{adjective}-{noun}-{suffix:04d}"


class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Provider(TimeStampedModel):
    name = models.CharField(max_length=120)
    slug = models.SlugField(max_length=120, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Credential(TimeStampedModel):
    class Kind(models.TextChoices):
        ANTHROPIC = "anthropic", "Anthropic API key"
        CLAUDE_OAUTH_TOKEN = "claude_oauth_token", "Claude Code OAuth token"
        CODEX_AUTH_JSON = "codex_auth_json", "Codex auth.json"
        OPENAI = "openai", "OpenAI API key"

    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    kind = models.CharField(max_length=30, choices=Kind.choices, default=Kind.OPENAI)
    provider = models.ForeignKey(
        Provider,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="credentials",
    )
    api_key = models.TextField()
    base_url = models.URLField(blank=True, default="")
    organization_id = models.CharField(max_length=200, blank=True)
    allowed_groups = models.ManyToManyField(
        Group, blank=True, related_name="credentials"
    )
    allowed_users = models.ManyToManyField(
        django_settings.AUTH_USER_MODEL,
        blank=True,
        related_name="credentials",
    )
    created_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_credentials",
    )

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    def can_view(self, user: AbstractUser) -> bool:
        if not user.is_authenticated:
            return False
        if user.is_superuser:
            return True

        allowed_user_ids = set(self.allowed_users.values_list("id", flat=True))
        allowed_group_ids = set(self.allowed_groups.values_list("id", flat=True))
        if not allowed_user_ids and not allowed_group_ids:
            return True
        if user.pk in allowed_user_ids:
            return True
        return user.groups.filter(id__in=allowed_group_ids).exists()

    def can_use(self, user: AbstractUser) -> bool:
        return self.can_view(user)


class ModelConfiguration(TimeStampedModel):
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    view_groups = models.ManyToManyField(
        Group, blank=True, related_name="viewable_model_configurations"
    )
    use_groups = models.ManyToManyField(
        Group, blank=True, related_name="usable_model_configurations"
    )
    created_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_model_configurations",
    )

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    def can_view(self, user: AbstractUser) -> bool:
        return _can_access_grouped_object(user, self.view_groups)

    def can_use(self, user: AbstractUser) -> bool:
        return _can_access_grouped_object(user, self.use_groups)


class ModelPricing(TimeStampedModel):
    model = models.ForeignKey(
        ModelConfiguration,
        on_delete=models.CASCADE,
        related_name="pricing",
    )
    provider = models.ForeignKey(
        Provider,
        on_delete=models.CASCADE,
        related_name="model_pricing",
    )
    input_per_million = models.DecimalField(max_digits=12, decimal_places=6)
    cached_input_per_million = models.DecimalField(max_digits=12, decimal_places=6)
    output_per_million = models.DecimalField(max_digits=12, decimal_places=6)

    class Meta:
        ordering = ["provider__name", "model__name"]
        constraints = [
            models.UniqueConstraint(
                fields=["model", "provider"],
                name="unique_model_pricing_by_model_provider",
            )
        ]

    def __str__(self) -> str:
        return f"{self.provider.slug}:{self.model.name}"

    def estimate_usd(self, usage: dict[str, Any]) -> Decimal:
        input_tokens = _decimal_token_count(usage.get("input_tokens"))
        cached_input_tokens = _decimal_token_count(usage.get("cached_input_tokens"))
        cache_creation_input_tokens = _decimal_token_count(
            usage.get("cache_creation_input_tokens")
        )
        cache_read_input_tokens = _decimal_token_count(
            usage.get("cache_read_input_tokens")
        )
        output_tokens = _decimal_token_count(usage.get("output_tokens"))
        billable_input_tokens = max(
            input_tokens + cache_creation_input_tokens - cached_input_tokens,
            Decimal("0"),
        )
        billable_cached_tokens = cached_input_tokens + cache_read_input_tokens
        usd = (
            billable_input_tokens * self.input_per_million
            + billable_cached_tokens * self.cached_input_per_million
            + output_tokens * self.output_per_million
        ) / Decimal("1000000")
        return usd.quantize(Decimal("0.000001"))

    def snapshot(self) -> dict[str, str]:
        return {
            "provider": self.provider.slug,
            "model": self.model.name,
            "input_per_million": str(self.input_per_million),
            "cached_input_per_million": str(self.cached_input_per_million),
            "output_per_million": str(self.output_per_million),
        }


class AgentConfiguration(TimeStampedModel):
    name = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    yaml = models.TextField()
    view_groups = models.ManyToManyField(
        Group, blank=True, related_name="viewable_agent_configurations"
    )
    use_groups = models.ManyToManyField(
        Group, blank=True, related_name="usable_agent_configurations"
    )
    created_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_agent_configurations",
    )

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("ctf:agent_detail", kwargs={"slug": self.slug})

    def can_view(self, user: AbstractUser) -> bool:
        return _can_access_grouped_object(user, self.view_groups)

    def can_use(self, user: AbstractUser) -> bool:
        return _can_access_grouped_object(user, self.use_groups)

    def resolved_yaml(self, *, user: Any | None = None) -> str:
        register_credential_resolver()
        config = OmegaConf.create(self.yaml)
        with resolve_credentials_as(user):
            try:
                return OmegaConf.to_yaml(config, resolve=True)
            except InterpolationResolutionError as exc:
                _raise_permission_denied_from_interpolation(exc)
                raise

    def resolved_mapping(self, *, user: Any | None = None) -> dict[str, Any]:
        register_credential_resolver()
        with resolve_credentials_as(user):
            try:
                data = OmegaConf.to_container(OmegaConf.create(self.yaml), resolve=True)
            except InterpolationResolutionError as exc:
                _raise_permission_denied_from_interpolation(exc)
                raise
        if not isinstance(data, dict):
            raise ValueError("agent YAML must resolve to a mapping")
        return {str(key): value for key, value in data.items()}


class FlowConfiguration(TimeStampedModel):
    name = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    yaml = models.TextField()
    view_groups = models.ManyToManyField(
        Group, blank=True, related_name="viewable_flow_configurations"
    )
    use_groups = models.ManyToManyField(
        Group, blank=True, related_name="usable_flow_configurations"
    )
    created_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_flow_configurations",
    )

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("ctf:flow_detail", kwargs={"slug": self.slug})

    def can_view(self, user: AbstractUser) -> bool:
        return _can_access_grouped_object(user, self.view_groups)

    def can_use(self, user: AbstractUser) -> bool:
        return _can_access_grouped_object(user, self.use_groups)

    def resolved_yaml(self, *, user: Any | None = None) -> str:
        register_credential_resolver()
        config = OmegaConf.create(self.yaml)
        with resolve_credentials_as(user):
            try:
                return OmegaConf.to_yaml(config, resolve=True)
            except InterpolationResolutionError as exc:
                _raise_permission_denied_from_interpolation(exc)
                raise

    def resolved_mapping(self, *, user: Any | None = None) -> dict[str, Any]:
        register_credential_resolver()
        with resolve_credentials_as(user):
            try:
                data = OmegaConf.to_container(OmegaConf.create(self.yaml), resolve=True)
            except InterpolationResolutionError as exc:
                _raise_permission_denied_from_interpolation(exc)
                raise
        if not isinstance(data, dict):
            raise ValueError("flow YAML must resolve to a mapping")
        return {str(key): value for key, value in data.items()}


class Ctf(TimeStampedModel):
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    prompt = models.TextField(blank=True, default="")
    settings = models.TextField(blank=True, default="")
    view_groups = models.ManyToManyField(
        Group, blank=True, related_name="viewable_ctfs"
    )
    init_groups = models.ManyToManyField(
        Group, blank=True, related_name="initializable_ctfs"
    )
    created_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_ctfs",
    )

    class Meta:
        verbose_name = "CTF"
        verbose_name_plural = "CTFs"
        ordering = ["title"]

    def __str__(self) -> str:
        return self.title

    def get_absolute_url(self) -> str:
        return reverse("ctf:ctf_detail", kwargs={"slug": self.slug})

    def can_view(self, user: AbstractUser) -> bool:
        return _can_access_grouped_object(user, self.view_groups)

    def can_init_thread(self, user: AbstractUser) -> bool:
        return _can_access_grouped_object(user, self.init_groups)

    def settings_mapping(self) -> dict[str, Any]:
        return _yaml_mapping(self.settings)


def challenge_source_upload_path(instance: Challenge, filename: str) -> str:
    return f"ctfs/{instance.ctf.slug}/challenges/{instance.challenge_id}/{filename}"


class Challenge(TimeStampedModel):
    ctf = models.ForeignKey(Ctf, on_delete=models.CASCADE, related_name="challenges")
    challenge_id = models.SlugField()
    description = models.TextField(blank=True)
    webhook = models.TextField(blank=True, default="")
    config = models.TextField(blank=True, default="")
    source_archive = models.FileField(
        upload_to=challenge_source_upload_path,
        blank=True,
    )
    created_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_challenges",
    )

    class Meta:
        unique_together = [("ctf", "challenge_id")]
        ordering = ["ctf__title", "challenge_id"]

    def __str__(self) -> str:
        return f"{self.ctf}: {self.challenge_id}"

    def get_absolute_url(self) -> str:
        return reverse(
            "ctf:challenge_detail",
            kwargs={"ctf_slug": self.ctf.slug, "challenge_id": self.challenge_id},
        )

    def webhook_mapping(self) -> dict[str, Any]:
        return _yaml_mapping(self.webhook)

    def config_mapping(self) -> dict[str, Any]:
        return _yaml_mapping(self.config)

    @property
    def webhook_summary(self) -> dict[str, str] | None:
        try:
            mapping = self.webhook_mapping()
        except Exception:
            return None
        if not mapping:
            return None
        url = str(mapping.get("url") or "")
        if "discord.com" in url:
            provider = "Discord"
        elif "hooks.slack.com" in url:
            provider = "Slack"
        elif url:
            provider = "Webhook"
        else:
            return None
        language = mapping.get("preferred_language")
        return {"provider": provider, "language": str(language) if language else ""}


class Thread(TimeStampedModel):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        WAITING = "waiting", "Waiting"
        STOPPED = "stopped", "Stopped"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    uuid = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    ctf = models.ForeignKey(Ctf, on_delete=models.CASCADE, related_name="threads")
    challenge = models.ForeignKey(
        Challenge, on_delete=models.PROTECT, related_name="threads"
    )
    agent = models.ForeignKey(
        AgentConfiguration,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="threads",
    )
    flow = models.ForeignKey(
        FlowConfiguration,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="threads",
    )
    model = models.ForeignKey(
        ModelConfiguration,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="threads",
    )
    credential = models.ForeignKey(
        Credential,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="threads",
    )
    created_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_threads",
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.QUEUED
    )
    name = models.SlugField(max_length=80, blank=True, default=generate_thread_name)
    task_result_id = models.CharField(max_length=64, blank=True)
    thread_root = models.CharField(max_length=500, blank=True)
    workspace_path = models.CharField(max_length=500, blank=True)
    metadata_path = models.CharField(max_length=500, blank=True)
    error = models.TextField(blank=True)
    latest_cost = models.JSONField(default=dict, blank=True)
    is_public = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name or f"{self.challenge.challenge_id} #{self.pk}"

    def save(self, *args: Any, **kwargs: Any) -> None:
        original_name = self.name
        self.name = slugify(self.name)[:80] if self.name else generate_thread_name()
        if not self.name:
            self.name = generate_thread_name()
        if self.name != original_name and kwargs.get("update_fields") is not None:
            kwargs["update_fields"] = {*kwargs["update_fields"], "name"}
        super().save(*args, **kwargs)

    def get_absolute_url(self) -> str:
        return reverse("ctf:thread_detail", kwargs={"thread_uuid": self.uuid})

    @property
    def metadata_directory(self) -> Path | None:
        return Path(self.metadata_path) if self.metadata_path else None

    def can_view(self, user: AbstractUser) -> bool:
        return True

    def can_interact(self, user: AbstractUser) -> bool:
        return self.ctf.can_view(user)

    def can_publish(self, user: AbstractUser) -> bool:
        return self.can_interact(user)


class StreamEvent(TimeStampedModel):
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="events")
    format = models.CharField(max_length=120, blank=True)
    raw = models.TextField(blank=True)

    class Meta:
        ordering = ["id"]

    def __str__(self) -> str:
        return f"{self.thread_id}:{self.pk}:{self.format}"


class SteeringMessage(TimeStampedModel):
    class Kind(models.TextChoices):
        STEER = "steer", "Steer"
        PROMPT = "prompt", "Prompt"
        STOP = "stop", "Stop"

    thread = models.ForeignKey(
        Thread, on_delete=models.CASCADE, related_name="steering_messages"
    )
    created_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_steering_messages",
    )
    kind = models.CharField(max_length=20, choices=Kind.choices, default=Kind.STEER)
    text = models.TextField(blank=True)
    delivered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.thread_id}:{self.created_at.isoformat()}"


class ThreadCostSnapshot(TimeStampedModel):
    thread = models.ForeignKey(
        Thread, on_delete=models.CASCADE, related_name="cost_snapshots"
    )
    usage = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["created_at"]


def register_credential_resolver() -> None:
    OmegaConf.register_new_resolver("credential", _resolve_credential, replace=True)
    OmegaConf.register_new_resolver("secret", _resolve_credential, replace=True)


@contextmanager
def resolve_credentials_as(user: Any | None) -> Iterator[None]:
    token = _credential_resolver_user.set(user)
    try:
        yield
    finally:
        _credential_resolver_user.reset(token)


def _resolve_credential(name: str) -> str:
    user = _credential_resolver_user.get()
    if user is None:
        raise PermissionDenied("credential resolver requires an authenticated user")

    credential = Credential.objects.prefetch_related(
        "allowed_groups", "allowed_users"
    ).get(slug=name)
    if not credential.can_view(user):
        raise PermissionDenied(f"credential is not accessible: {name}")
    return credential.api_key


def _raise_permission_denied_from_interpolation(
    exc: InterpolationResolutionError,
) -> None:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, PermissionDenied):
            raise current
        current = current.__cause__ or current.__context__
    if str(exc).startswith("PermissionDenied raised while resolving interpolation"):
        raise PermissionDenied(str(exc)) from exc


def _yaml_mapping(value: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    data = OmegaConf.to_container(OmegaConf.create(value), resolve=True)
    if not isinstance(data, dict):
        raise ValueError("YAML value must resolve to a mapping")
    return {str(key): item for key, item in data.items()}


def _decimal_token_count(value: object) -> Decimal:
    if isinstance(value, bool):
        return Decimal("0")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(int(value)))
    if isinstance(value, str) and value.isdecimal():
        return Decimal(value)
    return Decimal("0")


def _can_access_grouped_object(
    user: AbstractUser,
    groups: models.Manager[Group],
) -> bool:
    if not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    allowed_group_ids = groups.values_list("id", flat=True)
    if not allowed_group_ids:
        return True
    return user.groups.filter(id__in=allowed_group_ids).exists()
