from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from django import forms
from django.core.files import File
from django.core.files.uploadedfile import UploadedFile
from django.utils.text import slugify
from omegaconf import OmegaConf

from .models import (
    AgentConfiguration,
    Challenge,
    Credential,
    Ctf,
    ModelConfiguration,
    ModelPricing,
    Provider,
)
from .pricing import PRICING_PRESET_BY_KEY, PRICING_PRESETS
from .source_archives import (
    SOURCE_ARCHIVE_FORMAT_HINT,
    DownloadedSourceArchive,
    download_source_archive,
    validate_source_archive_upload,
)


class CredentialForm(forms.ModelForm):
    class Meta:
        model = Credential
        fields = [
            "name",
            "slug",
            "kind",
            "provider",
            "api_key",
            "base_url",
            "organization_id",
            "allowed_groups",
            "allowed_users",
        ]
        labels = {
            "api_key": "Secret",
            "organization_id": "Organization ID",
        }
        help_texts = {
            "api_key": "API key, OAuth token, or raw Codex auth.json depending on the selected kind.",
            "provider": "Used for matching token usage to pricing. If blank, Catchy infers it from the credential kind.",
            "base_url": "Optional override for API-compatible endpoints.",
            "organization_id": "Only used by OpenAI API key credentials.",
        }
        widgets = {"api_key": forms.PasswordInput(render_value=False)}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields["api_key"].required = False
            self.fields["api_key"].initial = ""
            self.fields["api_key"].help_text = (
                "Leave blank to keep the current secret. Enter a new API key, "
                "OAuth token, or raw Codex auth.json to replace it."
            )

    def clean_api_key(self) -> str:
        value = self.cleaned_data.get("api_key", "")
        if not value and self.instance.pk:
            value = self.instance.api_key
        kind = self.cleaned_data.get("kind")
        if kind != Credential.Kind.CODEX_AUTH_JSON:
            return value

        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise forms.ValidationError("Enter valid Codex auth.json.") from exc
        if not isinstance(payload, dict):
            raise forms.ValidationError("Codex auth.json must contain a JSON object.")
        return value

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        if cleaned.get("provider") is None:
            provider_slug = _provider_slug_for_credential_kind(cleaned.get("kind"))
            if provider_slug:
                cleaned["provider"] = Provider.objects.filter(
                    slug=provider_slug
                ).first()
        return cleaned


class ProviderForm(forms.ModelForm):
    class Meta:
        model = Provider
        fields = ["name", "slug"]
        help_texts = {
            "slug": "Stable provider identifier used by token usage and pricing events.",
        }


class ModelConfigurationForm(forms.ModelForm):
    class Meta:
        model = ModelConfiguration
        fields = ["name", "slug", "view_groups", "use_groups"]


class ModelPricingForm(forms.ModelForm):
    pricing_preset = forms.ChoiceField(
        required=False,
        choices=[("", "Custom pricing")]
        + [(preset.key, preset.label) for preset in PRICING_PRESETS],
        help_text="Optional preset. Selecting one fills provider and token rates on save.",
    )

    class Meta:
        model = ModelPricing
        fields = [
            "model",
            "provider",
            "input_per_million",
            "cached_input_per_million",
            "output_per_million",
        ]
        labels = {
            "input_per_million": "Input / 1M tokens",
            "cached_input_per_million": "Cached input / 1M tokens",
            "output_per_million": "Output / 1M tokens",
        }
        help_texts = {
            "provider": "Provider for this model pricing entry.",
            "cached_input_per_million": "Use 0 when the provider has no separate cached-input price.",
        }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fields["provider"].required = False
        for field_name in (
            "input_per_million",
            "cached_input_per_million",
            "output_per_million",
        ):
            self.fields[field_name].required = False
        self.order_fields(
            [
                "model",
                "pricing_preset",
                "provider",
                "input_per_million",
                "cached_input_per_million",
                "output_per_million",
            ]
        )

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        preset_key = str(cleaned.get("pricing_preset") or "")
        if preset_key:
            preset = PRICING_PRESET_BY_KEY[preset_key]
            provider = Provider.objects.filter(slug=preset.provider_slug).first()
            if provider is None:
                self.add_error(
                    "pricing_preset",
                    f"Provider does not exist for preset: {preset.provider_slug}",
                )
                return cleaned
            cleaned["provider"] = provider
            for field_name, value in preset.as_pricing().items():
                cleaned[field_name] = value
            return cleaned

        if cleaned.get("provider") is None:
            self.add_error("provider", "Select a provider or pricing preset.")
        for field_name in (
            "input_per_million",
            "cached_input_per_million",
            "output_per_million",
        ):
            if cleaned.get(field_name) is None:
                self.add_error(field_name, "Enter a price or select a pricing preset.")
        return cleaned


def _provider_slug_for_credential_kind(kind: object) -> str:
    if kind in {Credential.Kind.OPENAI, Credential.Kind.CODEX_AUTH_JSON}:
        return "openai"
    if kind in {Credential.Kind.ANTHROPIC, Credential.Kind.CLAUDE_OAUTH_TOKEN}:
        return "anthropic"
    return ""


class AgentConfigurationForm(forms.ModelForm):
    class Meta:
        model = AgentConfiguration
        fields = ["name", "slug", "yaml", "view_groups", "use_groups"]
        widgets = {"yaml": forms.Textarea(attrs={"rows": 22, "cols": 100})}

    def __init__(self, *args: Any, user=None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.user = user

    def clean_yaml(self) -> str:
        yaml = self.cleaned_data["yaml"]
        try:
            AgentConfiguration(yaml=yaml).resolved_mapping(user=self.user)
        except Exception as exc:
            raise forms.ValidationError(f"invalid agent YAML: {exc}") from exc
        return yaml


class CtfForm(forms.ModelForm):
    settings_yaml = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 8, "cols": 80}),
        help_text="Optional YAML mapping for future CTF-level settings.",
    )

    class Meta:
        model = Ctf
        fields = [
            "title",
            "slug",
            "description",
            "prompt",
            "view_groups",
            "init_groups",
        ]
        help_texts = {
            "prompt": "Optional prompt appended to the initial agent prompt for this CTF.",
        }
        widgets = {
            "prompt": forms.Textarea(attrs={"rows": 8, "cols": 80}),
        }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields["settings_yaml"].initial = self.instance.settings

    def clean_settings_yaml(self) -> str:
        value = str(self.cleaned_data.get("settings_yaml", ""))
        _clean_yaml_mapping(value)
        return value

    def save(self, commit: bool = True) -> Ctf:
        instance = super().save(commit=False)
        instance.settings = self.cleaned_data["settings_yaml"]
        if commit:
            instance.save()
            self.save_m2m()
        return instance


class ChallengeForm(forms.ModelForm):
    source_url = forms.URLField(
        required=False,
        label="Download URL",
        help_text=(
            "Use instead of uploading a file. Catchy downloads it now and stores a "
            f"reusable archive. Supported formats: {SOURCE_ARCHIVE_FORMAT_HINT}."
        ),
        widget=forms.URLInput(attrs={"placeholder": "https://example.com/source.zip"}),
    )
    webhook_url = forms.URLField(
        required=False,
        label="Webhook URL",
        help_text="Endpoint the agent can POST to during the run. Leave blank to disable.",
        widget=forms.URLInput(attrs={"placeholder": "https://example.com/hook"}),
    )
    webhook_preferred_language = forms.CharField(
        required=False,
        label="Preferred language",
        help_text="Spoken language the agent should respond in (e.g. English, 한국어). Optional.",
        widget=forms.TextInput(attrs={"placeholder": "English"}),
    )
    clear_webhook = forms.BooleanField(
        required=False,
        label="Remove webhook",
        help_text="Check to remove the existing webhook from this challenge.",
    )
    config_yaml = forms.CharField(
        required=False,
        label="Config (YAML)",
        help_text="Free-form YAML mapping forwarded to the challenge runner.",
        widget=forms.Textarea(attrs={"rows": 8, "cols": 80}),
    )

    fieldsets = [
        ("Basics", ["challenge_id", "description", "source_archive", "source_url"]),
        ("Webhook", ["webhook_url", "webhook_preferred_language", "clear_webhook"]),
        ("Advanced", ["config_yaml"]),
    ]

    class Meta:
        model = Challenge
        fields = ["challenge_id", "description", "source_archive"]
        help_texts = {
            "description": "Markdown is supported.",
            "source_archive": (
                "Upload a challenge source archive. Supported formats: "
                f"{SOURCE_ARCHIVE_FORMAT_HINT}."
            ),
        }

    def __init__(
        self,
        *args: Any,
        ctf: Ctf | None = None,
        **kwargs: Any,
    ) -> None:
        self.ctf = ctf
        self._downloaded_source_archive: DownloadedSourceArchive | None = None
        super().__init__(*args, **kwargs)
        if self.ctf is not None and not self.instance.pk:
            self.instance.ctf = self.ctf
        self.fields["source_archive"].required = False
        if self.instance.pk:
            webhook_data = _safe_yaml_mapping(self.instance.webhook)
            existing_webhook_url = str(webhook_data.get("url") or "")
            self.fields["webhook_preferred_language"].initial = webhook_data.get(
                "preferred_language", ""
            )
            if existing_webhook_url:
                self.fields[
                    "webhook_url"
                ].help_text = "A webhook URL is set. Leave blank to keep it, or enter a new URL to replace."
                self.fields["webhook_url"].widget.attrs["placeholder"] = (
                    "(URL set — leave blank to keep)"
                )
            else:
                self.fields["clear_webhook"].widget = forms.HiddenInput()
            self.fields["config_yaml"].initial = self.instance.config
            self.fields["source_archive"].required = False
            if self.instance.source_archive:
                self.fields["source_archive"].help_text = (
                    "Leave blank to keep the existing archive, upload a new archive "
                    "to replace it, or enter a download URL. Supported formats: "
                    f"{SOURCE_ARCHIVE_FORMAT_HINT}."
                )
            else:
                self.fields["source_archive"].help_text = (
                    "Upload an archive, or leave blank when using a download URL. "
                    f"Supported formats: {SOURCE_ARCHIVE_FORMAT_HINT}."
                )
        else:
            self.fields["clear_webhook"].widget = forms.HiddenInput()

    def clean_source_archive(self):
        archive = self.cleaned_data.get("source_archive")
        if not archive:
            if self.instance.pk and self.instance.source_archive:
                return self.instance.source_archive
            return archive
        if not isinstance(archive, UploadedFile):
            return archive
        try:
            archive.file.seek(0)
            validate_source_archive_upload(archive.file, archive.name)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc
        finally:
            archive.file.seek(0)
        return archive

    def clean_source_url(self) -> str:
        url = str(self.cleaned_data.get("source_url") or "").strip()
        if not url:
            return ""
        scheme = urlparse(url).scheme.lower()
        if scheme not in {"http", "https"}:
            raise forms.ValidationError("Download URL must use http or https.")
        if self.files.get("source_archive"):
            return url
        try:
            self._downloaded_source_archive = download_source_archive(url)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc
        return url

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        source_url = str(cleaned.get("source_url") or "").strip()
        has_new_upload = bool(self.files.get("source_archive"))
        has_existing_archive = bool(self.instance.pk and self.instance.source_archive)
        if has_new_upload and source_url:
            self.add_error(
                "source_url",
                "Upload a file or enter a download URL, not both.",
            )
        elif not has_new_upload and not source_url and not has_existing_archive:
            self.add_error(
                "source_archive",
                "Upload a source archive or enter a download URL.",
            )

        url = (cleaned.get("webhook_url") or "").strip()
        lang = (cleaned.get("webhook_preferred_language") or "").strip()
        clear = bool(cleaned.get("clear_webhook"))
        existing_url = ""
        if self.instance.pk:
            existing_url = str(
                _safe_yaml_mapping(self.instance.webhook).get("url") or ""
            )
        effective_url = "" if clear else (url or existing_url)
        if lang and not effective_url:
            self.add_error(
                "webhook_url",
                "Webhook URL is required when a preferred language is set.",
            )
        return cleaned

    def clean_config_yaml(self) -> str:
        value = str(self.cleaned_data.get("config_yaml", ""))
        _clean_yaml_mapping(value)
        return value

    def save(self, commit: bool = True) -> Challenge:
        instance = super().save(commit=False)
        has_new_upload = bool(self.files.get("source_archive"))
        source_url = self.cleaned_data["source_url"]
        if not has_new_upload and source_url:
            if self._downloaded_source_archive is None:
                raise ValueError("source URL was not downloaded")
            if self.ctf is not None and not instance.ctf_id:
                instance.ctf = self.ctf
            self._downloaded_source_archive.file.seek(0)
            instance.source_archive.save(
                self._downloaded_source_archive.name,
                File(self._downloaded_source_archive.file),
                save=False,
            )
        instance.webhook = self._serialize_webhook()
        instance.config = self.cleaned_data["config_yaml"]
        if commit:
            instance.save()
        return instance

    def _serialize_webhook(self) -> str:
        if self.cleaned_data.get("clear_webhook"):
            return ""
        url = (self.cleaned_data.get("webhook_url") or "").strip()
        lang = (self.cleaned_data.get("webhook_preferred_language") or "").strip()
        if not url and self.instance.pk:
            url = str(_safe_yaml_mapping(self.instance.webhook).get("url") or "")
        if not url:
            return ""
        payload: dict[str, Any] = {"url": url}
        if lang:
            payload["preferred_language"] = lang
        return OmegaConf.to_yaml(OmegaConf.create(payload))


class ThreadCreateForm(forms.Form):
    name = forms.CharField(
        max_length=80,
        required=False,
        widget=forms.TextInput(attrs={"placeholder": "bright-cipher-0427"}),
    )
    agent = forms.ModelChoiceField(queryset=AgentConfiguration.objects.none())
    model = forms.ModelChoiceField(queryset=ModelConfiguration.objects.none())
    credential = forms.ModelChoiceField(queryset=Credential.objects.none())

    def __init__(
        self,
        *args: Any,
        user,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        agent_ids = [
            agent.pk
            for agent in AgentConfiguration.objects.prefetch_related("use_groups")
            if agent.can_use(user)
        ]
        self.fields["agent"].queryset = AgentConfiguration.objects.filter(
            pk__in=agent_ids
        )
        model_ids = [
            model.pk
            for model in ModelConfiguration.objects.prefetch_related("use_groups")
            if model.can_use(user)
        ]
        self.fields["model"].queryset = ModelConfiguration.objects.filter(
            pk__in=model_ids
        )
        credential_ids = [
            credential.pk
            for credential in Credential.objects.prefetch_related(
                "allowed_groups", "allowed_users"
            )
            if credential.can_use(user)
        ]
        self.fields["credential"].queryset = Credential.objects.filter(
            pk__in=credential_ids
        )

    def clean_name(self) -> str:
        value = self.cleaned_data.get("name", "")
        if not value:
            return ""
        name = slugify(value)[:80]
        if not name:
            raise forms.ValidationError("Enter a name with letters or numbers.")
        return name


def _clean_yaml_mapping(value: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    try:
        data = OmegaConf.to_container(OmegaConf.create(value), resolve=True)
    except Exception as exc:
        raise forms.ValidationError(f"invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise forms.ValidationError("YAML value must be a mapping")
    return {str(key): item for key, item in data.items()}


def _safe_yaml_mapping(value: str) -> dict[str, Any]:
    if not value or not value.strip():
        return {}
    try:
        data = OmegaConf.to_container(OmegaConf.create(value), resolve=True)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): item for key, item in data.items()}
