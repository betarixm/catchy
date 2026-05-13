from __future__ import annotations

import io
import importlib
import tarfile
import tempfile
import zipfile
from collections.abc import AsyncGenerator
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

from asgiref.sync import async_to_sync
from catchy.core.agents.models import (
    Chunk,
    Event,
    Interrupt,
    ItemCompleted,
    Log,
    Nop,
    Steer,
    Stop,
    TokenUsage,
)
from catchy.core.agents.protocols import Agent
from catchy.core.challenge.models import Challenge as CoreChallenge
from django.contrib.auth.models import Group, User
from django.core.exceptions import PermissionDenied
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from catchy.web.ctf import services, views
from catchy.web.ctf.forms import (
    ChallengeForm,
    CredentialForm,
    ModelConfigurationForm,
    ModelPricingForm,
    ProviderForm,
    ThreadCreateForm,
)
from catchy.web.ctf.models import (
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
    ThreadCostSnapshot,
)
from catchy.web.ctf.source_archives import (
    DownloadedSourceArchive,
    safe_extract_archive,
)


class CredentialAgentPermissionTests(TestCase):
    def setUp(self) -> None:
        self.allowed_group = Group.objects.create(name="credential-users")
        self.allowed_user = User.objects.create_user(
            username="allowed",
            password="password",
        )
        self.allowed_user.groups.add(self.allowed_group)
        self.direct_user = User.objects.create_user(
            username="direct",
            password="password",
        )
        self.denied_user = User.objects.create_user(
            username="denied",
            password="password",
        )
        self.credential = Credential.objects.create(
            name="OpenAI Token",
            slug="api-token",
            api_key="top-secret",
        )
        self.credential.allowed_groups.add(self.allowed_group)

    def test_credential_access_allows_group_or_user(self) -> None:
        user_credential = Credential.objects.create(
            name="Direct Token",
            slug="direct-token",
            api_key="direct-secret",
        )
        user_credential.allowed_users.add(self.direct_user)

        self.assertTrue(self.credential.can_view(self.allowed_user))
        self.assertFalse(self.credential.can_view(self.direct_user))
        self.assertTrue(user_credential.can_view(self.direct_user))
        self.assertFalse(user_credential.can_view(self.denied_user))

    def test_credential_name_accepts_non_slug_text(self) -> None:
        form = CredentialForm(
            {
                "name": "OpenAI Production Key",
                "slug": "openai-prod",
                "kind": Credential.Kind.OPENAI,
                "api_key": "test-key",
                "base_url": "",
                "organization_id": "",
            }
        )

        self.assertTrue(form.is_valid(), form.errors)

    def test_anthropic_credential_accepts_blank_base_url(self) -> None:
        form = CredentialForm(
            {
                "name": "Anthropic Production Key",
                "slug": "anthropic-prod",
                "kind": Credential.Kind.ANTHROPIC,
                "api_key": "test-key",
                "base_url": "",
                "organization_id": "",
            }
        )

        self.assertTrue(form.is_valid(), form.errors)

    def test_credential_form_infers_provider_from_kind(self) -> None:
        form = CredentialForm(
            {
                "name": "Claude Login",
                "slug": "claude-login",
                "kind": Credential.Kind.CLAUDE_OAUTH_TOKEN,
                "api_key": "oauth-secret",
                "base_url": "",
                "organization_id": "",
            }
        )

        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["provider"].slug, "anthropic")

    def test_codex_auth_json_credential_requires_json_object(self) -> None:
        form = CredentialForm(
            {
                "name": "Codex Login",
                "slug": "codex-login",
                "kind": Credential.Kind.CODEX_AUTH_JSON,
                "api_key": "not-json",
                "base_url": "",
                "organization_id": "",
            }
        )

        self.assertFalse(form.is_valid())
        self.assertIn("api_key", form.errors)

    def test_credential_update_form_does_not_expose_secret(self) -> None:
        self.client.force_login(self.allowed_user)

        response = self.client.get(
            reverse("ctf:credential_update", kwargs={"slug": self.credential.slug})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Edit credential")
        self.assertNotContains(response, "top-secret")

    def test_credential_update_blank_secret_keeps_existing_value(self) -> None:
        self.client.force_login(self.allowed_user)

        response = self.client.post(
            reverse("ctf:credential_update", kwargs={"slug": self.credential.slug}),
            {
                "name": "OpenAI Updated",
                "slug": "api-token",
                "kind": Credential.Kind.OPENAI,
                "api_key": "",
                "base_url": "",
                "organization_id": "org_updated",
                "allowed_groups": [str(self.allowed_group.pk)],
            },
        )

        self.assertRedirects(response, reverse("ctf:credential_list"))
        self.credential.refresh_from_db()
        self.assertEqual(self.credential.name, "OpenAI Updated")
        self.assertEqual(self.credential.api_key, "top-secret")
        self.assertEqual(self.credential.organization_id, "org_updated")

    def test_credential_update_replaces_secret_when_provided(self) -> None:
        self.client.force_login(self.allowed_user)

        response = self.client.post(
            reverse("ctf:credential_update", kwargs={"slug": self.credential.slug}),
            {
                "name": "OpenAI Token",
                "slug": "api-token",
                "kind": Credential.Kind.OPENAI,
                "api_key": "new-secret",
                "base_url": "",
                "organization_id": "",
                "allowed_groups": [str(self.allowed_group.pk)],
            },
        )

        self.assertRedirects(response, reverse("ctf:credential_list"))
        self.credential.refresh_from_db()
        self.assertEqual(self.credential.api_key, "new-secret")

    def test_credential_update_rejects_disallowed_user(self) -> None:
        self.client.force_login(self.denied_user)

        response = self.client.get(
            reverse("ctf:credential_update", kwargs={"slug": self.credential.slug})
        )

        self.assertEqual(response.status_code, 403)

    def test_credential_update_allows_direct_user(self) -> None:
        self.credential.allowed_users.add(self.direct_user)
        self.client.force_login(self.direct_user)

        response = self.client.get(
            reverse("ctf:credential_update", kwargs={"slug": self.credential.slug})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Edit credential")

    def test_model_name_accepts_non_slug_text(self) -> None:
        form = ModelConfigurationForm(
            {
                "name": "custom model/name",
                "slug": "custom-model",
            }
        )

        self.assertTrue(form.is_valid(), form.errors)

    def test_provider_form_accepts_provider_identifier(self) -> None:
        form = ProviderForm({"name": "OpenRouter", "slug": "openrouter"})

        self.assertTrue(form.is_valid(), form.errors)

    def test_provider_create_and_update_from_web(self) -> None:
        self.client.force_login(self.allowed_user)

        response = self.client.post(
            reverse("ctf:provider_create"),
            {"name": "OpenRouter", "slug": "openrouter"},
        )

        self.assertRedirects(response, reverse("ctf:provider_list"))
        provider = Provider.objects.get(slug="openrouter")
        self.assertEqual(provider.name, "OpenRouter")

        response = self.client.post(
            reverse("ctf:provider_update", kwargs={"slug": provider.slug}),
            {"name": "OpenRouter AI", "slug": "openrouter"},
        )

        self.assertRedirects(response, reverse("ctf:provider_list"))
        provider.refresh_from_db()
        self.assertEqual(provider.name, "OpenRouter AI")

    def test_model_pricing_form_applies_preset_to_provider_and_prices(self) -> None:
        model = ModelConfiguration.objects.create(name="gpt-5.5", slug="gpt-55")
        form = ModelPricingForm(
            {
                "model": str(model.pk),
                "pricing_preset": "openai:gpt-5.5",
                "provider": "",
                "input_per_million": "",
                "cached_input_per_million": "",
                "output_per_million": "",
            }
        )

        self.assertTrue(form.is_valid(), form.errors)
        pricing = form.save()
        self.assertEqual(pricing.provider.slug, "openai")
        self.assertEqual(pricing.input_per_million, Decimal("5.00"))

    def test_model_pricing_migration_considers_existing_gpt5_models(self) -> None:
        migration = importlib.import_module(
            "catchy.web.ctf.migrations.0013_provider_credential_provider_modelpricing"
        )

        expected = {
            "gpt-5.3-codex": "gpt-5.3-codex",
            "gpt-5.4": "gpt-5.4",
            "gpt-5.4-nano": "gpt-5.4-nano",
            "gpt-5.5": "gpt-5.5",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-5-nano": "gpt-5-nano",
        }

        for model_name, preset_model in expected.items():
            with self.subTest(model_name=model_name):
                preset = migration._preset_for_model_name(model_name)
                self.assertIsNotNone(preset)
                self.assertEqual(preset[0], "openai")
                self.assertEqual(preset[1], preset_model)

    def test_agent_resolves_credential_for_allowed_user(self) -> None:
        agent = AgentConfiguration(
            name="Codex",
            slug="codex",
            yaml="model:\n  api_key: ${credential:api-token}\n",
        )

        self.assertEqual(
            agent.resolved_mapping(user=self.allowed_user),
            {"model": {"api_key": "top-secret"}},
        )

    def test_agent_resolution_rejects_disallowed_credential_user(self) -> None:
        agent = AgentConfiguration(
            name="Codex",
            slug="codex",
            yaml="model:\n  api_key: ${credential:api-token}\n",
        )

        with self.assertRaises(PermissionDenied):
            agent.resolved_mapping(user=self.denied_user)

    def test_agent_create_rejects_credential_user_cannot_view(self) -> None:
        self.client.force_login(self.denied_user)

        response = self.client.post(
            reverse("ctf:agent_create"),
            {
                "name": "Codex",
                "slug": "codex",
                "yaml": "model:\n  api_key: ${credential:api-token}\n",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentConfiguration.objects.filter(slug="codex").exists())

    def test_build_agent_configuration_overlays_model_and_credential(self) -> None:
        agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="id: codex\nclass: catchy.codex.CodexAgent\nmodel:\n  name: old\n",
        )
        model = ModelConfiguration.objects.create(name="gpt-5.5", slug="gpt-55")
        self.credential.base_url = "https://example.test/v1"
        self.credential.organization_id = "org_123"
        self.credential.save(
            update_fields=["base_url", "organization_id", "updated_at"]
        )

        data = services.build_agent_configuration(
            agent,
            model_configuration=model,
            credential=self.credential,
            user=self.allowed_user,
        )

        self.assertEqual(data["model"]["name"], "gpt-5.5")
        self.assertNotIn("provider", data["model"])
        self.assertNotIn("api_key", data["model"])
        self.assertEqual(data["credential"]["api_key"], "top-secret")
        self.assertEqual(data["credential"]["base_url"], "https://example.test/v1")
        self.assertEqual(data["credential"]["organization_id"], "org_123")

    def test_build_agent_configuration_overlays_anthropic_credential(self) -> None:
        agent = AgentConfiguration.objects.create(
            name="Claude Code",
            slug="claude-code",
            yaml=(
                "id: claude-code\n"
                "class: catchy.claude_code.ClaudeCodeAgent\n"
                "model:\n"
                "  provider: anthropic\n"
                "  name: old\n"
            ),
        )
        model = ModelConfiguration.objects.create(
            name="claude-sonnet-4-5", slug="claude-sonnet-45"
        )
        credential = Credential.objects.create(
            name="Anthropic Token",
            slug="anthropic-token",
            kind=Credential.Kind.ANTHROPIC,
            api_key="anthropic-secret",
            base_url="",
        )
        credential.allowed_groups.add(self.allowed_group)

        data = services.build_agent_configuration(
            agent,
            model_configuration=model,
            credential=credential,
            user=self.allowed_user,
        )

        self.assertEqual(data["model"]["name"], "claude-sonnet-4-5")
        self.assertNotIn("provider", data["model"])
        self.assertNotIn("api_key", data["model"])
        self.assertEqual(data["credential"]["api_key"], "anthropic-secret")
        self.assertNotIn("base_url", data["credential"])

    def test_build_agent_configuration_overlays_codex_auth_json(self) -> None:
        agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="id: codex\nclass: catchy.codex.CodexAgent\nmodel:\n  name: old\n",
        )
        model = ModelConfiguration.objects.create(name="gpt-5.5", slug="gpt-55")
        credential = Credential.objects.create(
            name="Codex Login",
            slug="codex-login",
            kind=Credential.Kind.CODEX_AUTH_JSON,
            api_key='{"auth_mode": "chatgpt"}',
            base_url="https://example.test/v1",
        )
        credential.allowed_groups.add(self.allowed_group)

        data = services.build_agent_configuration(
            agent,
            model_configuration=model,
            credential=credential,
            user=self.allowed_user,
        )

        self.assertEqual(data["model"]["name"], "gpt-5.5")
        self.assertEqual(data["credential"]["json_string"], '{"auth_mode": "chatgpt"}')
        self.assertEqual(data["credential"]["base_url"], "https://example.test/v1")

    def test_build_agent_configuration_overlays_claude_oauth_token(self) -> None:
        agent = AgentConfiguration.objects.create(
            name="Claude Code",
            slug="claude-code",
            yaml=(
                "id: claude-code\n"
                "class: catchy.claude_code.ClaudeCodeAgent\n"
                "model:\n"
                "  name: old\n"
            ),
        )
        model = ModelConfiguration.objects.create(
            name="claude-sonnet-4-5", slug="claude-sonnet-45"
        )
        credential = Credential.objects.create(
            name="Claude Login",
            slug="claude-login",
            kind=Credential.Kind.CLAUDE_OAUTH_TOKEN,
            api_key="oauth-secret",
        )
        credential.allowed_groups.add(self.allowed_group)

        data = services.build_agent_configuration(
            agent,
            model_configuration=model,
            credential=credential,
            user=self.allowed_user,
        )

        self.assertEqual(data["model"]["name"], "claude-sonnet-4-5")
        self.assertEqual(data["credential"], {"token": "oauth-secret"})

    def test_thread_create_rejects_credential_user_cannot_view(self) -> None:
        ctf = Ctf.objects.create(title="Study", slug="study")
        challenge = Challenge.objects.create(
            ctf=ctf,
            challenge_id="canary",
            source_archive="ctfs/study/challenges/canary/source.tgz",
        )
        agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="{}",
        )
        model = ModelConfiguration.objects.create(name="gpt-5.5", slug="gpt-55")
        self.client.force_login(self.denied_user)

        response = self.client.post(
            reverse(
                "ctf:thread_create",
                kwargs={"ctf_slug": ctf.slug, "challenge_id": challenge.challenge_id},
            ),
            {
                "agent": str(agent.pk),
                "model": str(model.pk),
                "credential": str(self.credential.pk),
            },
        )

        self.assertRedirects(response, challenge.get_absolute_url())
        self.assertFalse(Thread.objects.exists())

    def test_thread_create_form_hides_credentials_user_cannot_use(self) -> None:
        form = ThreadCreateForm(user=self.denied_user)

        self.assertNotIn(self.credential, form.fields["credential"].queryset)

    def test_thread_create_form_includes_direct_user_credential(self) -> None:
        credential = Credential.objects.create(
            name="Direct Token",
            slug="direct-token",
            api_key="direct-secret",
        )
        credential.allowed_users.add(self.direct_user)

        form = ThreadCreateForm(user=self.direct_user)

        self.assertIn(credential, form.fields["credential"].queryset)


class ChallengeSourceFormTests(TestCase):
    def setUp(self) -> None:
        self.ctf = Ctf.objects.create(title="Study", slug="study")

    def test_challenge_form_accepts_zip_upload(self) -> None:
        form = ChallengeForm(
            data={
                "challenge_id": "zip-source",
                "description": "",
                "source_url": "",
                "webhook_url": "",
                "webhook_preferred_language": "",
                "config_yaml": "",
            },
            files={
                "source_archive": SimpleUploadedFile(
                    "source.zip",
                    _zip_archive_bytes(),
                ),
            },
            ctf=self.ctf,
        )

        self.assertTrue(form.is_valid(), form.errors)

    def test_challenge_form_accepts_tar_xz_upload(self) -> None:
        form = ChallengeForm(
            data={
                "challenge_id": "tar-xz-source",
                "description": "",
                "source_url": "",
                "webhook_url": "",
                "webhook_preferred_language": "",
                "config_yaml": "",
            },
            files={
                "source_archive": SimpleUploadedFile(
                    "source.tar.xz",
                    _tar_xz_archive_bytes(),
                ),
            },
            ctf=self.ctf,
        )

        self.assertTrue(form.is_valid(), form.errors)

    def test_challenge_form_downloads_url_source_once_and_saves_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, override_settings(MEDIA_ROOT=tmp):
            downloaded = DownloadedSourceArchive(
                name="source.zip",
                file=io.BytesIO(_zip_archive_bytes()),
            )
            with patch(
                "catchy.web.ctf.forms.download_source_archive",
                return_value=downloaded,
            ) as download:
                form = ChallengeForm(
                    data={
                        "challenge_id": "remote-source",
                        "description": "",
                        "source_url": "https://example.test/source.zip",
                        "webhook_url": "",
                        "webhook_preferred_language": "",
                        "config_yaml": "",
                    },
                    ctf=self.ctf,
                )

                self.assertTrue(form.is_valid(), form.errors)
                challenge = form.save()

            download.assert_called_once_with("https://example.test/source.zip")
            self.assertEqual(
                challenge.source_archive.name,
                "ctfs/study/challenges/remote-source/source.zip",
            )
            self.assertTrue((Path(tmp) / challenge.source_archive.name).exists())

    def test_challenge_form_rejects_missing_source(self) -> None:
        form = ChallengeForm(
            data={
                "challenge_id": "missing-source",
                "description": "",
                "source_url": "",
                "webhook_url": "",
                "webhook_preferred_language": "",
                "config_yaml": "",
            },
            ctf=self.ctf,
        )

        self.assertFalse(form.is_valid())
        self.assertIn("source_archive", form.errors)


class ChallengeSourceArchiveTests(TestCase):
    def test_safe_extract_archive_supports_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "source.zip"
            destination = Path(tmp) / "source"
            archive_path.write_bytes(_zip_archive_bytes())

            safe_extract_archive(archive_path, destination)

            self.assertEqual((destination / "README.md").read_text(), "hello\n")


class ModelConfigurationViewTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username="member", password="password")
        self.group = Group.objects.create(name="model-users")
        self.user.groups.add(self.group)
        self.model = ModelConfiguration.objects.create(name="gpt-5.5", slug="gpt-55")
        self.client.force_login(self.user)

    def test_model_update_edits_model_configuration(self) -> None:
        response = self.client.post(
            reverse("ctf:model_update", kwargs={"slug": self.model.slug}),
            {
                "name": "gpt-5.5 latest",
                "slug": "gpt-55-latest",
                "view_groups": [str(self.group.pk)],
                "use_groups": [str(self.group.pk)],
            },
        )

        self.assertRedirects(response, reverse("ctf:model_list"))
        self.model.refresh_from_db()
        self.assertEqual(self.model.name, "gpt-5.5 latest")
        self.assertEqual(self.model.slug, "gpt-55-latest")
        self.assertEqual(list(self.model.view_groups.all()), [self.group])
        self.assertEqual(list(self.model.use_groups.all()), [self.group])


class ThreadCreateNameTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username="member", password="password")
        self.ctf = Ctf.objects.create(title="Study", slug="study")
        self.challenge = Challenge.objects.create(
            ctf=self.ctf,
            challenge_id="canary",
            source_archive="ctfs/study/challenges/canary/source.tgz",
        )
        self.agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="{}",
        )
        self.model = ModelConfiguration.objects.create(name="gpt-5.5", slug="gpt-55")
        self.credential = Credential.objects.create(
            name="OpenAI",
            slug="openai",
            api_key="test-key",
        )
        self.client.force_login(self.user)

    def test_thread_create_accepts_optional_name(self) -> None:
        with patch("catchy.web.ctf.views.start_thread") as start_thread:
            response = self.client.post(
                self._thread_create_url(),
                {
                    "agent": str(self.agent.pk),
                    "model": str(self.model.pk),
                    "credential": str(self.credential.pk),
                    "name": "My First Run",
                },
            )

        thread = Thread.objects.get()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.name, "my-first-run")
        self.assertEqual(thread.model, self.model)
        self.assertEqual(thread.credential, self.credential)
        self.assertEqual(str(thread), "my-first-run")
        self.assertEqual(start_thread.call_args.args[0].pk, thread.pk)

    def test_thread_create_generates_kebab_name_when_blank(self) -> None:
        with patch("catchy.web.ctf.views.start_thread"):
            response = self.client.post(
                self._thread_create_url(),
                {
                    "agent": str(self.agent.pk),
                    "model": str(self.model.pk),
                    "credential": str(self.credential.pk),
                    "name": "",
                },
            )

        thread = Thread.objects.get()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertRegex(thread.name, r"^[a-z]+-[a-z]+-\d{4}$")
        self.assertEqual(str(thread), thread.name)

    def _thread_create_url(self) -> str:
        return reverse(
            "ctf:thread_create",
            kwargs={
                "ctf_slug": self.ctf.slug,
                "challenge_id": self.challenge.challenge_id,
            },
        )


class StreamEventRecordingTests(TestCase):
    def setUp(self) -> None:
        self.ctf = Ctf.objects.create(title="Study", slug="study")
        self.challenge = Challenge.objects.create(
            ctf=self.ctf,
            challenge_id="canary",
            source_archive="ctfs/study/challenges/canary/source.tgz",
        )
        self.agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="{}",
        )
        self.thread = Thread.objects.create(
            ctf=self.ctf,
            challenge=self.challenge,
            agent=self.agent,
        )

    def test_record_stream_event_persists_chunk_tag(self) -> None:
        services._record_stream_event(
            self.thread.pk,
            Chunk(tag="action", text="hello"),
            "gpt-5.5",
        )

        event = StreamEvent.objects.get(thread=self.thread)
        self.assertEqual(event.source, "agent_stream")
        self.assertEqual(event.kind, "chunk")
        self.assertEqual(event.text, "hello")
        self.assertEqual(event.raw, {"tag": "action"})

    def test_record_stream_event_persists_log_event(self) -> None:
        services._record_stream_event(
            self.thread.pk,
            Log(
                kind="token_count",
                text='{"total":{"inputTokens":1,"outputTokens":2}}',
                raw={
                    "tokenUsage": {
                        "total": {
                            "inputTokens": 1,
                            "cachedInputTokens": 0,
                            "outputTokens": 2,
                        }
                    }
                },
            ),
            "gpt-5.5",
        )

        event = StreamEvent.objects.get(thread=self.thread)
        self.assertEqual(event.source, "agent_stream")
        self.assertEqual(event.kind, "token_count")
        self.assertEqual(event.text, '{"total":{"inputTokens":1,"outputTokens":2}}')
        self.assertEqual(
            event.raw,
            {
                "tokenUsage": {
                    "total": {
                        "inputTokens": 1,
                        "cachedInputTokens": 0,
                        "outputTokens": 2,
                    }
                }
            },
        )
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.latest_cost["provider"], "openai")
        self.assertEqual(self.thread.latest_cost["model"], "gpt-5.5")
        self.assertEqual(self.thread.latest_cost["input_tokens"], 1)
        self.assertEqual(self.thread.latest_cost["cached_input_tokens"], 0)
        self.assertEqual(self.thread.latest_cost["output_tokens"], 2)
        self.assertNotIn("usd", self.thread.latest_cost)
        self.assertEqual(ThreadCostSnapshot.objects.count(), 1)

    def test_record_stream_event_persists_standard_token_usage_event(self) -> None:
        services._record_stream_event(
            self.thread.pk,
            TokenUsage(
                provider="openai",
                model="gpt-5.5",
                source="thread_token_usage_updated",
                input_tokens=1,
                output_tokens=2,
                raw={"turnId": "turn-1"},
            ),
            "fallback-model",
        )

        event = StreamEvent.objects.get(thread=self.thread)
        self.assertEqual(event.source, "agent_stream")
        self.assertEqual(event.kind, "token_count")
        self.assertEqual(
            event.raw,
            {
                "provider": "openai",
                "model": "gpt-5.5",
                "source": "thread_token_usage_updated",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
                "raw": {"turnId": "turn-1"},
            },
        )
        self.assertNotIn("pricing", event.raw)
        self.assertNotIn("usd", event.raw)
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.latest_cost["model"], "gpt-5.5")
        self.assertEqual(self.thread.latest_cost["input_tokens"], 1)
        self.assertEqual(self.thread.latest_cost["output_tokens"], 2)

    def test_record_stream_event_calculates_cost_lazily_when_model_pricing_exists(
        self,
    ) -> None:
        provider = Provider.objects.get(slug="openai")
        model = ModelConfiguration.objects.create(name="gpt-5.5", slug="gpt-55")
        ModelPricing.objects.create(
            model=model,
            provider=provider,
            input_per_million=Decimal("2.00"),
            cached_input_per_million=Decimal("1.00"),
            output_per_million=Decimal("10.00"),
        )
        self.thread.model = model
        self.thread.save(update_fields=["model", "updated_at"])

        services._record_stream_event(
            self.thread.pk,
            TokenUsage(
                provider="openai",
                model="gpt-5.5",
                source="thread_token_usage_updated",
                input_tokens=1_000_000,
                cached_input_tokens=100_000,
                cache_read_input_tokens=200_000,
                output_tokens=500_000,
            ),
            "fallback-model",
        )

        event = StreamEvent.objects.get(thread=self.thread)
        self.thread.refresh_from_db()
        self.assertNotIn("usd", self.thread.latest_cost)
        self.assertNotIn("pricing", self.thread.latest_cost)
        self.assertNotIn("usd", ThreadCostSnapshot.objects.get().usage)
        payload = views._event_payload(event, thread=self.thread)
        self.assertEqual(payload["cost_usd"], "7.100000")

    def test_record_stream_event_persists_item_termination(self) -> None:
        services._record_stream_event(
            self.thread.pk,
            ItemCompleted(),
            "gpt-5.5",
        )

        event = StreamEvent.objects.get(thread=self.thread)
        self.assertEqual(event.source, "agent_stream")
        self.assertEqual(event.kind, "item.terminated")
        self.assertEqual(event.text, "")

    def test_record_stream_event_persists_claude_usage_log_event(self) -> None:
        services._record_stream_event(
            self.thread.pk,
            Log(
                kind="token_count",
                text='{"input_tokens":10,"output_tokens":5}',
                raw={
                    "provider": "anthropic",
                    "source": "result_message",
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 3,
                        "cache_read_input_tokens": 2,
                        "output_tokens": 5,
                    },
                },
            ),
            "claude-sonnet-4-5",
        )

        event = StreamEvent.objects.get(thread=self.thread)
        self.assertEqual(event.source, "agent_stream")
        self.assertEqual(event.kind, "token_count")
        self.assertEqual(event.raw["provider"], "anthropic")
        self.thread.refresh_from_db()
        self.assertEqual(
            self.thread.latest_cost,
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2,
                "output_tokens": 5,
                "reasoning_output_tokens": 0,
                "total_tokens": 20,
            },
        )
        self.assertEqual(ThreadCostSnapshot.objects.count(), 1)

    def test_record_stream_event_persists_standard_claude_token_usage_event(
        self,
    ) -> None:
        services._record_stream_event(
            self.thread.pk,
            TokenUsage(
                provider="anthropic",
                model="claude-sonnet-4-5",
                source="result_message",
                input_tokens=10,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=2,
                output_tokens=5,
            ),
            "fallback-model",
        )

        event = StreamEvent.objects.get(thread=self.thread)
        self.assertEqual(event.source, "agent_stream")
        self.assertEqual(event.kind, "token_count")
        self.assertEqual(event.raw["provider"], "anthropic")
        self.assertNotIn("usd", event.raw)
        self.thread.refresh_from_db()
        self.assertEqual(
            self.thread.latest_cost,
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2,
                "output_tokens": 5,
                "reasoning_output_tokens": 0,
                "total_tokens": 20,
            },
        )

    def test_record_stream_event_uses_structured_chunk_tags_as_kind(self) -> None:
        services._record_stream_event(
            self.thread.pk,
            Chunk(tag="tool_use", text='{"name":"Read"}'),
            "claude-sonnet-4-5",
        )

        event = StreamEvent.objects.get(thread=self.thread)
        self.assertEqual(event.source, "agent_stream")
        self.assertEqual(event.kind, "tool_use")
        self.assertEqual(event.text, '{"name":"Read"}')
        self.assertEqual(event.raw, {"tag": "tool_use"})

    def test_record_stream_event_ignores_nop(self) -> None:
        services._record_stream_event(self.thread.pk, Nop(), "gpt-5.5")

        self.assertFalse(StreamEvent.objects.filter(thread=self.thread).exists())

    def test_pop_next_thread_command_returns_steer_interrupt(self) -> None:
        SteeringMessage.objects.create(
            thread=self.thread,
            kind=SteeringMessage.Kind.STEER,
            text="try another path",
        )

        command = services._pop_next_thread_command(self.thread.pk)

        self.assertIsInstance(command, Steer)
        self.assertEqual(command.text, "try another path")
        message = SteeringMessage.objects.get(thread=self.thread)
        self.assertIsNotNone(message.delivered_at)
        self.assertEqual(
            list(
                StreamEvent.objects.filter(thread=self.thread).values_list(
                    "source", "kind", "text"
                )
            ),
            [("user", "steer", "try another path")],
        )

    def test_pop_next_thread_command_returns_stop_interrupt(self) -> None:
        SteeringMessage.objects.create(
            thread=self.thread,
            kind=SteeringMessage.Kind.STOP,
        )

        command = services._pop_next_thread_command(self.thread.pk)

        self.assertIsInstance(command, Stop)
        self.assertEqual(
            list(
                StreamEvent.objects.filter(thread=self.thread).values_list(
                    "source", "kind", "text"
                )
            ),
            [("user", "stop", "")],
        )

    def test_run_agent_stream_uses_initial_prompt_and_waits_after_turn(self) -> None:
        agent = _PromptRecordingAgent()
        SteeringMessage.objects.create(
            thread=self.thread,
            kind=SteeringMessage.Kind.PROMPT,
            text="try another path",
        )

        status = async_to_sync(services._run_agent_stream)(
            thread_id=self.thread.pk,
            agent=agent,
            challenge=CoreChallenge(
                id=self.challenge.challenge_id,
                description="",
                directory=Path("/tmp"),
            ),
            workspace=Path("/tmp"),
            metadata=Path("/tmp"),
            webhook=None,
            model_name="gpt-5.5",
        )

        self.assertEqual(status, Thread.Status.WAITING)
        self.assertEqual(agent.prompts, ["try another path"])
        self.assertEqual(
            list(
                StreamEvent.objects.filter(thread=self.thread).values_list(
                    "source", "kind", "text"
                )
            ),
            [
                ("user", "prompt", "try another path"),
                ("agent_stream", "chunk", "ready"),
                ("agent_stream", "item.terminated", ""),
            ],
        )

    def test_thread_root_reuses_existing_root_when_resuming(self) -> None:
        self.thread.thread_root = "/tmp/catchy-existing-thread"

        self.assertEqual(
            services._thread_root(self.thread),
            Path("/tmp/catchy-existing-thread"),
        )


class _SteerRecordingAgent(Agent):
    def __init__(self) -> None:
        self.interrupts: list[Interrupt] = []

    async def stream(
        self,
        *,
        challenge: CoreChallenge,
        workspace: Path,
        metadata_directory: Path,
        webhook: Any | None = None,
        prompt: str | None = None,
    ) -> AsyncGenerator[Event, Interrupt]:
        interrupt = yield Chunk(tag="action", text="ready")
        self.interrupts.append(interrupt)
        yield ItemCompleted()


class _PromptRecordingAgent(Agent):
    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    async def stream(
        self,
        *,
        challenge: CoreChallenge,
        workspace: Path,
        metadata_directory: Path,
        webhook: Any | None = None,
        prompt: str | None = None,
    ) -> AsyncGenerator[Event, Interrupt]:
        self.prompts.append(prompt)
        yield Chunk(tag="action", text="ready")
        yield ItemCompleted()


class PublicThreadAccessTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username="member", password="password")
        self.ctf = Ctf.objects.create(title="Study", slug="study")
        self.agent = AgentConfiguration.objects.create(
            name="Codex",
            slug="codex",
            yaml="{}",
        )

    def test_anonymous_dashboard_groups_only_public_threads_by_ctf(self) -> None:
        public_thread = self._create_thread("public", is_public=True)
        second_public_thread = self._create_thread("second-public", is_public=True)
        private_thread = self._create_thread("private", is_public=False)
        other_ctf = Ctf.objects.create(title="Other", slug="other")
        other_public_thread = self._create_thread(
            "other-public",
            is_public=True,
            ctf=other_ctf,
        )

        response = self.client.get(reverse("ctf:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(public_thread))
        self.assertContains(response, str(second_public_thread))
        self.assertContains(response, str(other_public_thread))
        self.assertContains(response, "Public threads")
        self.assertNotContains(response, str(private_thread))
        self.assertEqual(response.context["public_thread_count"], 3)

        groups = response.context["public_thread_groups"]
        grouped_threads = {
            group["ctf"].slug: {
                thread.pk
                for challenge_group in group["challenges"]
                for thread in challenge_group["threads"]
            }
            for group in groups
        }
        self.assertEqual(
            grouped_threads,
            {
                "study": {public_thread.pk, second_public_thread.pk},
                "other": {other_public_thread.pk},
            },
        )

    def test_dashboard_thread_list_shows_and_caches_latest_cost(self) -> None:
        provider, _created = Provider.objects.get_or_create(
            slug="openai",
            defaults={"name": "OpenAI"},
        )
        model = ModelConfiguration.objects.create(name="gpt-5.5", slug="gpt-55")
        ModelPricing.objects.create(
            model=model,
            provider=provider,
            input_per_million=Decimal("2.00"),
            cached_input_per_million=Decimal("1.00"),
            output_per_million=Decimal("10.00"),
        )
        thread = self._create_thread("priced-public", is_public=True)
        thread.model = model
        thread.latest_cost = {
            "provider": "openai",
            "model": "gpt-5.5",
            "input_tokens": 1_000_000,
            "cached_input_tokens": 100_000,
            "cache_read_input_tokens": 200_000,
            "output_tokens": 500_000,
            "total_tokens": 1_700_000,
        }
        thread.save(update_fields=["model", "latest_cost", "updated_at"])

        response = self.client.get(reverse("ctf:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "$7.100000")
        thread.refresh_from_db()
        self.assertEqual(thread.latest_cost["cost_usd"], "7.100000")

    def test_anonymous_can_view_public_thread(self) -> None:
        thread = self._create_thread("public-detail", is_public=True)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(thread))
        self.assertContains(response, "public")
        self.assertNotContains(response, "Publish")
        self.assertNotContains(response, "Steer</button>")

    def test_anonymous_can_view_private_thread_readonly(self) -> None:
        thread = self._create_thread("private-detail", is_public=False)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(thread))
        self.assertNotContains(response, "Publish")
        self.assertNotContains(response, 'id="steer-form"')

    def test_user_without_ctf_access_views_thread_readonly(self) -> None:
        managers = Group.objects.create(name="study-managers")
        self.ctf.view_groups.add(managers)
        thread = self._create_thread("restricted-detail", is_public=False)
        self.client.force_login(self.user)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(thread))
        self.assertNotContains(response, "Publish")
        self.assertNotContains(response, "Fork")
        self.assertNotContains(response, 'id="steer-form"')

    def test_thread_detail_hides_credential_name_without_credential_access(
        self,
    ) -> None:
        credential_group = Group.objects.create(name="credential-managers")
        credential = Credential.objects.create(
            name="Restricted Credential",
            slug="restricted-credential",
            api_key="secret",
        )
        credential.allowed_groups.add(credential_group)
        thread = self._create_thread(
            "credential-detail",
            is_public=False,
            credential=credential,
        )
        self.client.force_login(self.user)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(thread))
        self.assertContains(response, 'id="steer-form"')
        self.assertNotContains(response, "Restricted Credential")

    def test_challenge_detail_hides_credential_name_without_credential_access(
        self,
    ) -> None:
        credential_group = Group.objects.create(name="credential-managers")
        credential = Credential.objects.create(
            name="Restricted Credential",
            slug="restricted-credential",
            api_key="secret",
        )
        credential.allowed_groups.add(credential_group)
        thread = self._create_thread(
            "credential-list",
            is_public=False,
            credential=credential,
        )
        self.client.force_login(self.user)

        response = self.client.get(thread.challenge.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(thread))
        self.assertNotContains(response, "Restricted Credential")

    def test_user_without_ctf_access_cannot_publish_thread(self) -> None:
        managers = Group.objects.create(name="study-managers")
        self.ctf.view_groups.add(managers)
        thread = self._create_thread("restricted-publish", is_public=False)
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("ctf:thread_publish", kwargs={"thread_uuid": thread.uuid}),
            {"is_public": "1"},
        )

        thread.refresh_from_db()
        self.assertEqual(response.status_code, 403)
        self.assertFalse(thread.is_public)

    def test_published_thread_detail_shows_unpublish_button(self) -> None:
        thread = self._create_thread("published", is_public=True)
        self.client.force_login(self.user)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Unpublish")
        self.assertNotContains(response, ">Publish</button>")

    def test_completed_thread_detail_shows_steering_form_to_manager(self) -> None:
        thread = self._create_thread("steerable", is_public=False)
        self.client.force_login(self.user)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="steer-form"')

    def test_failed_thread_detail_hides_steering_form_to_manager(self) -> None:
        thread = self._create_thread("failed-detail", is_public=False)
        thread.status = Thread.Status.FAILED
        thread.error = "Codex turn failed"
        thread.save(update_fields=["status", "error", "updated_at"])
        self.client.force_login(self.user)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'id="steer-form"')

    def test_stopped_thread_detail_hides_interaction_controls(self) -> None:
        thread = self._create_thread("stopped-detail", is_public=False)
        thread.status = Thread.Status.STOPPED
        thread.save(update_fields=["status", "updated_at"])
        self.client.force_login(self.user)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'id="steer-form"')
        self.assertNotContains(response, ">Stop</button>")

    def test_thread_filetree_shows_workspace_only(self) -> None:
        thread = self._create_thread("workspace-tree", is_public=True)
        self.client.force_login(self.user)

        with tempfile.TemporaryDirectory() as temp_dir:
            thread_root = Path(temp_dir) / f"thread-{thread.pk}"
            workspace = thread_root / "workspace"
            metadata = thread_root / "metadata"
            source = thread_root / "source"
            (workspace / "src").mkdir(parents=True)
            metadata.mkdir(parents=True)
            source.mkdir(parents=True)
            (workspace / "README.md").write_text("hello\n", encoding="utf-8")
            (workspace / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
            (metadata / "secret.txt").write_text("nope\n", encoding="utf-8")
            (source / "archive.txt").write_text("nope\n", encoding="utf-8")

            thread.thread_root = str(thread_root)
            thread.workspace_path = str(workspace)
            thread.metadata_path = str(metadata)
            thread.save(
                update_fields=[
                    "thread_root",
                    "workspace_path",
                    "metadata_path",
                    "updated_at",
                ]
            )

            response = self.client.get(
                reverse("ctf:thread_filetree", kwargs={"thread_uuid": thread.uuid})
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "workspace")
        self.assertEqual(payload["type"], "dir")
        child_names = {child["name"] for child in payload["children"]}
        self.assertIn("README.md", child_names)
        self.assertIn("src", child_names)
        self.assertNotIn("metadata", child_names)
        self.assertNotIn("source", child_names)

    def test_thread_detail_includes_live_filetree_refresh_hook(self) -> None:
        thread = self._create_thread("workspace-refresh", is_public=True)

        response = self.client.get(thread.get_absolute_url())

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "event.source === \"system\" && event.kind === \"workspace.changed\"")
        self.assertContains(response, "refreshFileTree()")
        self.assertContains(response, "EventSource")
        self.assertContains(response, "thread_filetree")

    def test_thread_stream_replays_workspace_changed_stream_event(self) -> None:
        thread = self._create_thread("workspace-stream", is_public=True)
        StreamEvent.objects.create(
            thread=thread,
            sequence=1,
            dedupe_key="workspace.changed:1",
            source="system",
            kind="workspace.changed",
            text="Workspace updated",
            raw={"changed_paths": ["workspace/src/main.py"]},
        )

        response = self.client.get(
            reverse("ctf:thread_stream", kwargs={"thread_uuid": thread.uuid})
        )
        body = b"".join(response.streaming_content).decode()

        self.assertIn('"kind": "workspace.changed"', body)
        self.assertIn('"changed_paths": ["workspace/src/main.py"]', body)

    def test_thread_filetree_shows_workspace_only(self) -> None:
        thread = self._create_thread("workspace-only", is_public=True)
        self.client.force_login(self.user)

        with tempfile.TemporaryDirectory() as temp_dir:
            thread_root = Path(temp_dir) / f"thread-{thread.pk}"
            workspace = thread_root / "workspace"
            metadata = thread_root / "metadata"
            source = thread_root / "source"
            (workspace / "src").mkdir(parents=True)
            metadata.mkdir(parents=True)
            source.mkdir(parents=True)
            (workspace / "README.md").write_text("hello\n", encoding="utf-8")
            (workspace / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
            (metadata / "secret.txt").write_text("nope\n", encoding="utf-8")
            (source / "archive.txt").write_text("nope\n", encoding="utf-8")

            thread.thread_root = str(thread_root)
            thread.workspace_path = str(workspace)
            thread.metadata_path = str(metadata)
            thread.save(
                update_fields=[
                    "thread_root",
                    "workspace_path",
                    "metadata_path",
                    "updated_at",
                ]
            )

            response = self.client.get(
                reverse("ctf:thread_filetree", kwargs={"thread_uuid": thread.uuid})
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "workspace")
        self.assertEqual(payload["type"], "dir")
        child_names = {child["name"] for child in payload["children"]}
        self.assertIn("README.md", child_names)
        self.assertIn("src", child_names)
        self.assertNotIn("metadata", child_names)
        self.assertNotIn("source", child_names)

    def test_authenticated_user_can_publish_and_unpublish_thread(self) -> None:
        thread = self._create_thread("publishable", is_public=False)
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("ctf:thread_publish", kwargs={"thread_uuid": thread.uuid}),
            {"is_public": "1"},
        )
        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertTrue(thread.is_public)

        response = self.client.post(
            reverse("ctf:thread_publish", kwargs={"thread_uuid": thread.uuid}),
            {"is_public": "0"},
        )
        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertFalse(thread.is_public)

    def test_message_to_completed_thread_queues_prompt_and_restarts_worker(
        self,
    ) -> None:
        thread = self._create_thread("resume-completed", is_public=False)
        thread.error = "old stop"
        thread.thread_root = "/tmp/catchy-existing-thread"
        thread.save(update_fields=["error", "thread_root", "updated_at"])
        self.client.force_login(self.user)

        with patch("catchy.web.ctf.views.start_thread") as start_thread:
            response = self.client.post(
                reverse("ctf:thread_steer", kwargs={"thread_uuid": thread.uuid}),
                {"text": "try the other path"},
            )

        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.status, Thread.Status.QUEUED)
        self.assertEqual(thread.error, "")
        self.assertEqual(thread.steering_messages.get().text, "try the other path")
        self.assertEqual(
            thread.steering_messages.get().kind,
            SteeringMessage.Kind.PROMPT,
        )
        self.assertEqual(start_thread.call_args.args[0].pk, thread.pk)

    def test_message_to_thread_allows_ctf_access_without_credential_access(
        self,
    ) -> None:
        credential_group = Group.objects.create(name="credential-managers")
        credential = Credential.objects.create(
            name="Restricted Credential",
            slug="restricted-credential",
            api_key="secret",
        )
        credential.allowed_groups.add(credential_group)
        thread = self._create_thread(
            "resume-restricted-credential",
            is_public=False,
            credential=credential,
        )
        thread.thread_root = "/tmp/catchy-existing-thread"
        thread.save(update_fields=["thread_root", "updated_at"])
        self.client.force_login(self.user)

        with patch("catchy.web.ctf.views.start_thread") as start_thread:
            response = self.client.post(
                reverse("ctf:thread_steer", kwargs={"thread_uuid": thread.uuid}),
                {"text": "continue"},
            )

        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(SteeringMessage.objects.get(thread=thread).text, "continue")
        self.assertEqual(start_thread.call_args.args[0].pk, thread.pk)

    def test_message_to_failed_thread_is_rejected(self) -> None:
        thread = self._create_thread("failed-prompt", is_public=False)
        thread.status = Thread.Status.FAILED
        thread.error = "Codex turn failed"
        thread.thread_root = "/tmp/catchy-existing-thread"
        thread.save(update_fields=["status", "error", "thread_root", "updated_at"])
        self.client.force_login(self.user)

        with patch("catchy.web.ctf.views.start_thread") as start_thread:
            response = self.client.post(
                reverse("ctf:thread_steer", kwargs={"thread_uuid": thread.uuid}),
                {"text": "try again"},
            )

        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.status, Thread.Status.FAILED)
        self.assertEqual(thread.error, "Codex turn failed")
        self.assertFalse(thread.steering_messages.exists())
        start_thread.assert_not_called()

    def test_message_to_stopped_thread_is_rejected(self) -> None:
        thread = self._create_thread("stopped-prompt", is_public=False)
        thread.status = Thread.Status.STOPPED
        thread.thread_root = "/tmp/catchy-existing-thread"
        thread.save(update_fields=["status", "thread_root", "updated_at"])
        self.client.force_login(self.user)

        with patch("catchy.web.ctf.views.start_thread") as start_thread:
            response = self.client.post(
                reverse("ctf:thread_steer", kwargs={"thread_uuid": thread.uuid}),
                {"text": "try again"},
            )

        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.status, Thread.Status.STOPPED)
        self.assertFalse(thread.steering_messages.exists())
        start_thread.assert_not_called()

    def test_steering_running_thread_does_not_restart_worker(self) -> None:
        thread = self._create_thread("steer-running", is_public=False)
        thread.status = Thread.Status.RUNNING
        thread.save(update_fields=["status", "updated_at"])
        self.client.force_login(self.user)

        with patch("catchy.web.ctf.views.start_thread") as start_thread:
            response = self.client.post(
                reverse("ctf:thread_steer", kwargs={"thread_uuid": thread.uuid}),
                {"text": "keep going"},
            )

        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.status, Thread.Status.RUNNING)
        self.assertEqual(SteeringMessage.objects.get(thread=thread).text, "keep going")
        self.assertEqual(
            SteeringMessage.objects.get(thread=thread).kind,
            SteeringMessage.Kind.STEER,
        )
        start_thread.assert_not_called()

    def test_message_to_queued_thread_queues_prompt_without_restarting_worker(
        self,
    ) -> None:
        thread = self._create_thread("message-queued", is_public=False)
        thread.status = Thread.Status.QUEUED
        thread.save(update_fields=["status", "updated_at"])
        self.client.force_login(self.user)

        with patch("catchy.web.ctf.views.start_thread") as start_thread:
            response = self.client.post(
                reverse("ctf:thread_steer", kwargs={"thread_uuid": thread.uuid}),
                {"text": "use this as the turn prompt"},
            )

        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.status, Thread.Status.QUEUED)
        self.assertEqual(
            SteeringMessage.objects.get(thread=thread).kind,
            SteeringMessage.Kind.PROMPT,
        )
        start_thread.assert_not_called()

    def test_stop_running_thread_queues_stop_message(self) -> None:
        thread = self._create_thread("stop-running", is_public=False)
        thread.status = Thread.Status.RUNNING
        thread.save(update_fields=["status", "updated_at"])
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("ctf:thread_stop", kwargs={"thread_uuid": thread.uuid})
        )

        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.status, Thread.Status.RUNNING)
        self.assertEqual(
            SteeringMessage.objects.get(thread=thread).kind,
            SteeringMessage.Kind.STOP,
        )

    def test_stop_waiting_thread_marks_stopped(self) -> None:
        thread = self._create_thread("stop-waiting", is_public=False)
        thread.status = Thread.Status.WAITING
        thread.save(update_fields=["status", "updated_at"])
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("ctf:thread_stop", kwargs={"thread_uuid": thread.uuid})
        )

        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.status, Thread.Status.STOPPED)
        self.assertEqual(
            list(thread.events.values_list("source", "kind", "text")),
            [("user", "stop", "")],
        )

    def test_stop_stopped_thread_is_rejected(self) -> None:
        thread = self._create_thread("stop-stopped", is_public=False)
        thread.status = Thread.Status.STOPPED
        thread.save(update_fields=["status", "updated_at"])
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("ctf:thread_stop", kwargs={"thread_uuid": thread.uuid})
        )

        thread.refresh_from_db()
        self.assertRedirects(response, thread.get_absolute_url())
        self.assertEqual(thread.status, Thread.Status.STOPPED)
        self.assertFalse(thread.events.exists())
        self.assertFalse(thread.steering_messages.exists())

    def test_fork_thread_copies_metadata_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "source"
            source_metadata = source_root / "metadata"
            source_metadata.mkdir(parents=True)
            (source_metadata / "session.jsonl").write_text("history\n")
            target_root = Path(tmp) / "target"

            thread = self._create_thread("fork-source", is_public=False)
            thread.thread_root = str(source_root)
            thread.metadata_path = str(source_metadata)
            thread.save(update_fields=["thread_root", "metadata_path", "updated_at"])
            StreamEvent.objects.create(
                thread=thread,
                sequence=1,
                dedupe_key="history-one",
                source="agent_stream",
                kind="chunk",
                text="hello",
                raw={"tag": "action"},
            )
            self.client.force_login(self.user)

            with patch("catchy.web.ctf.services._thread_root") as thread_root:
                thread_root.return_value = target_root
                response = self.client.post(
                    reverse("ctf:thread_fork", kwargs={"thread_uuid": thread.uuid})
                )

            fork = Thread.objects.exclude(pk=thread.pk).get()
            self.assertTrue((target_root / "metadata" / "session.jsonl").exists())

        self.assertRedirects(response, fork.get_absolute_url())
        self.assertEqual(fork.status, Thread.Status.WAITING)
        self.assertEqual(fork.agent, thread.agent)
        self.assertEqual(fork.metadata_path, str(target_root / "metadata"))
        self.assertEqual(
            list(fork.events.values_list("source", "kind", "text")),
            [
                ("agent_stream", "chunk", "hello"),
                ("system", "thread.forked", f"Forked from thread #{thread.pk}"),
            ],
        )

    def test_fork_thread_rejects_user_without_credential_access(self) -> None:
        credential_group = Group.objects.create(name="credential-managers")
        credential = Credential.objects.create(
            name="Restricted Credential",
            slug="restricted-credential",
            api_key="secret",
        )
        credential.allowed_groups.add(credential_group)
        thread = self._create_thread(
            "fork-restricted-credential",
            is_public=False,
            credential=credential,
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("ctf:thread_fork", kwargs={"thread_uuid": thread.uuid})
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(Thread.objects.count(), 1)

    def test_fork_thread_skips_codex_runtime_tmp_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "source"
            source_metadata = source_root / "metadata"
            source_session = source_metadata / ".codex" / "sessions" / "session.jsonl"
            source_session.parent.mkdir(parents=True)
            source_session.write_text("history\n")
            inaccessible = source_metadata / ".codex" / "tmp" / "arg0"
            inaccessible.mkdir(parents=True)
            inaccessible.chmod(0)
            target_root = Path(tmp) / "target"

            thread = self._create_thread("fork-source", is_public=False)
            thread.thread_root = str(source_root)
            thread.metadata_path = str(source_metadata)
            thread.save(update_fields=["thread_root", "metadata_path", "updated_at"])
            self.client.force_login(self.user)

            try:
                with patch("catchy.web.ctf.services._thread_root") as thread_root:
                    thread_root.return_value = target_root
                    response = self.client.post(
                        reverse("ctf:thread_fork", kwargs={"thread_uuid": thread.uuid})
                    )
            finally:
                inaccessible.chmod(0o700)

            fork = Thread.objects.exclude(pk=thread.pk).get()
            self.assertRedirects(response, fork.get_absolute_url())
            self.assertTrue(
                (
                    target_root / "metadata" / ".codex" / "sessions" / "session.jsonl"
                ).exists()
            )
            self.assertFalse((target_root / "metadata" / ".codex" / "tmp").exists())

    def test_thread_stream_starts_after_requested_sequence(self) -> None:
        thread = self._create_thread("stream-after", is_public=True)
        StreamEvent.objects.create(
            thread=thread,
            sequence=1,
            dedupe_key="one",
            source="system",
            kind="thread.started",
            text="one",
        )
        StreamEvent.objects.create(
            thread=thread,
            sequence=2,
            dedupe_key="two",
            source="system",
            kind="thread.completed",
            text="two",
        )

        response = self.client.get(
            reverse("ctf:thread_stream", kwargs={"thread_uuid": thread.uuid}),
            {"after": "1"},
        )
        body = b"".join(response.streaming_content).decode()

        self.assertNotIn('"sequence": 1', body)
        self.assertIn("id: 2", body)
        self.assertIn('"sequence": 2', body)
        self.assertEqual(response.headers["X-Accel-Buffering"], "no")

    def test_thread_stream_resumes_after_last_event_id(self) -> None:
        thread = self._create_thread("stream-last-event-id", is_public=True)
        for sequence in range(1, 4):
            StreamEvent.objects.create(
                thread=thread,
                sequence=sequence,
                dedupe_key=str(sequence),
                source="system",
                kind="thread.event",
                text=str(sequence),
            )

        response = self.client.get(
            reverse("ctf:thread_stream", kwargs={"thread_uuid": thread.uuid}),
            {"after": "1"},
            headers={"Last-Event-ID": "2"},
        )
        body = b"".join(response.streaming_content).decode()

        self.assertNotIn('"sequence": 1', body)
        self.assertNotIn('"sequence": 2', body)
        self.assertIn("id: 3", body)
        self.assertIn('"sequence": 3', body)

    def test_thread_stream_status_advertises_reconnect_retry(self) -> None:
        thread = self._create_thread("stream-waiting-retry", is_public=True)
        thread.status = Thread.Status.WAITING
        thread.save(update_fields=["status", "updated_at"])

        response = self.client.get(
            reverse("ctf:thread_stream", kwargs={"thread_uuid": thread.uuid})
        )
        body = b"".join(response.streaming_content).decode()

        self.assertIn("retry: 5000", body)
        self.assertIn("event: status", body)
        self.assertIn('"status": "waiting"', body)

    def _create_thread(
        self,
        challenge_id: str,
        *,
        is_public: bool,
        ctf: Ctf | None = None,
        credential: Credential | None = None,
    ) -> Thread:
        ctf = ctf or self.ctf
        challenge = Challenge.objects.create(
            ctf=ctf,
            challenge_id=challenge_id,
            source_archive=f"ctfs/{ctf.slug}/challenges/{challenge_id}/source.tgz",
        )
        return Thread.objects.create(
            ctf=ctf,
            challenge=challenge,
            agent=self.agent,
            credential=credential,
            status=Thread.Status.COMPLETED,
            is_public=is_public,
        )


def _zip_archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("README.md", "hello\n")
    return buffer.getvalue()


def _tar_xz_archive_bytes() -> bytes:
    buffer = io.BytesIO()
    content = b"hello\n"
    info = tarfile.TarInfo("README.md")
    info.size = len(content)
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()
