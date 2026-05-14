from __future__ import annotations

import io
import json
import logging
import os
import shlex
import tarfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Iterator, Literal, cast

from catchy.core.agent.models import (
    Component,
    Delta,
    Event,
    EventRenderer,
    Interrupt,
    ItemCompleted,
    JsonLog,
    Nop,
    Prompt,
    Steer,
    Stop,
    TextLog,
    ThreadStarted,
    TokenUsage,
    TurnCompleted,
    TurnStarted,
)
from catchy.core.agent.protocols import Agent, HandoffExporter
from catchy.core.challenge.models import Challenge
from catchy.core.webhook.models import Webhook
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
)
from claude_agent_sdk.types import (
    AssistantMessage,
    ContentBlock,
    DeferredToolUse,
    HookEventMessage,
    Message,
    MirrorErrorMessage,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from docker import DockerClient
from docker.errors import DockerException
from docker.models.containers import Container
from docker.models.images import Image
from jinja2 import Template
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_serializer,
    field_validator,
)

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)

_TYPE_KEY = "__claude_sdk_type__"

_SERIALIZABLE_TYPES = {
    # ContentBlock
    "TextBlock": TextBlock,
    "ThinkingBlock": ThinkingBlock,
    "ToolUseBlock": ToolUseBlock,
    "ToolResultBlock": ToolResultBlock,
    "ServerToolUseBlock": ServerToolUseBlock,
    "ServerToolResultBlock": ServerToolResultBlock,
    # Nested dataclasses
    "DeferredToolUse": DeferredToolUse,
    "RateLimitInfo": RateLimitInfo,
    # Message
    "UserMessage": UserMessage,
    "AssistantMessage": AssistantMessage,
    "SystemMessage": SystemMessage,
    "TaskStartedMessage": TaskStartedMessage,
    "TaskProgressMessage": TaskProgressMessage,
    "TaskNotificationMessage": TaskNotificationMessage,
    "MirrorErrorMessage": MirrorErrorMessage,
    "ResultMessage": ResultMessage,
    "StreamEvent": StreamEvent,
    "RateLimitEvent": RateLimitEvent,
    "HookEventMessage": HookEventMessage,
}

_LOGGER = logging.getLogger(__name__)


def message_json_default(value: Any) -> JsonValue:
    if is_dataclass(value) and not isinstance(value, type):
        class_name = type(value).__name__

        if class_name not in _SERIALIZABLE_TYPES:
            raise TypeError(f"Unsupported dataclass type: {class_name}")

        return {
            _TYPE_KEY: class_name,
            **{field.name: getattr(value, field.name) for field in fields(value)},
        }

    if isinstance(value, Path):
        return str(value)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def message_json_object_hook(value: dict[str, Any]) -> Any:
    class_name = value.get(_TYPE_KEY)

    if class_name is None:
        return value

    class_type = _SERIALIZABLE_TYPES.get(class_name)

    if class_type is None:
        raise ValueError(f"Unknown serialized SDK type: {class_name}")

    arguments = {key: item for key, item in value.items() if key != _TYPE_KEY}

    return class_type(**arguments)


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(
                cast(dict[str, Any], existing),
                cast(dict[str, Any], value),
            )
        else:
            merged[key] = value
    return merged


class _AnthropicCompatibleApiKeyCredential(BaseModel):
    api_key: str
    base_url: str | None = None


class _ClaudeCodeOauthTokenCredential(BaseModel):
    token: str


class _Model(BaseModel):
    name: str


class _Directory(BaseModel):
    challenge: str = "/challenge"
    workspace: str = "/workspace"
    metadata: str = "/metadata"


class _Container(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Literal["docker"] = "docker"
    socket: str = "unix:///var/run/docker.sock"
    image: Image

    @field_validator("image", mode="before")
    @classmethod
    def _deserialize_image(cls, value: Image | str, info: ValidationInfo) -> Image:
        if isinstance(value, Image):
            return value

        socket = info.data.get("socket", "unix:///var/run/docker.sock")
        client: DockerClient | None = None
        try:
            client = DockerClient(base_url=socket)
            try:
                return client.images.get(value)
            except DockerException:
                _LOGGER.info("Pulling Docker image: %s", value)
                return client.images.pull(value)
        except DockerException as exc:
            raise ValueError(
                f"Failed to resolve Docker image {value!r}: {exc}"
            ) from exc
        finally:
            if client is not None:
                client.close()

    @field_serializer("image")
    def _serialize_image(self, value: Image) -> str:
        return value.tags[0] if value.tags else value.id or value.short_id or ""


class _PromptTemplate(BaseModel):
    user: str


class Configuration(BaseModel):
    id: str
    model: _Model
    credential: _AnthropicCompatibleApiKeyCredential | _ClaudeCodeOauthTokenCredential
    directory: _Directory
    container: _Container
    prompt: _PromptTemplate


class ClaudeCodeEvent(Event[Message]):
    format: str = "claude-code-message"

    @field_validator("raw", mode="before")
    @classmethod
    def _deserialize_raw(cls, value: object) -> Message:
        if isinstance(value, Message):
            return value
        if not isinstance(value, str):
            raise ValidationError("raw value must be a string or Message")
        return json.loads(value, object_hook=message_json_object_hook)

    @field_serializer("raw")
    def _serialize_raw(self, value: Message) -> str:
        return json.dumps(value, default=message_json_default, ensure_ascii=False)


class ClaudeCodeAgent(Agent[Message]):
    key: str = "claude-code"

    @staticmethod
    def from_configuration(configuration: Configuration) -> ClaudeCodeAgent:
        match configuration.credential:
            case _AnthropicCompatibleApiKeyCredential() as credential:
                environment = {
                    "ANTHROPIC_API_KEY": credential.api_key,
                    **(
                        {"ANTHROPIC_BASE_URL": credential.base_url}
                        if credential.base_url
                        else {}
                    ),
                }
            case _ClaudeCodeOauthTokenCredential() as credential:
                environment = {
                    "CLAUDE_OAUTH_TOKEN": credential.token,
                }

        return ClaudeCodeAgent(
            id=configuration.id,
            model_name=configuration.model.name,
            container_environment=environment,
            container_challenge_directory=configuration.directory.challenge,
            container_workspace_directory=configuration.directory.workspace,
            container_metadata_directory=configuration.directory.metadata,
            docker_image=configuration.container.image,
            docker_client=DockerClient(base_url=configuration.container.socket),
            user_prompt_template=configuration.prompt.user,
            docker_socket=configuration.container.socket,
        )

    def __init__(
        self,
        id: str,
        model_name: str,
        container_environment: dict[str, str],
        container_challenge_directory: str,
        container_workspace_directory: str,
        container_metadata_directory: str,
        docker_image: Image,
        docker_client: DockerClient,
        user_prompt_template: str,
        docker_socket: str = "/var/run/docker.sock",
    ):
        self._id = id
        self._model_name = model_name
        self._container_environment = container_environment
        self._container_challenge_directory = container_challenge_directory
        self._container_workspace_directory = container_workspace_directory
        self._container_metadata_directory = container_metadata_directory
        self._docker_image = docker_image
        self._docker_client = docker_client
        self._docker_socket = docker_socket
        self._user_prompt_template = user_prompt_template

        image_name = docker_image.tags[0] if docker_image.tags else docker_image.id
        try:
            container = self._docker_client.containers.run(
                self._docker_image,
                command=["sh", "-c", "command -v claude >/dev/null 2>&1"],
                detach=True,
                remove=True,
            )
            result = container.wait()
            exit_code = result.get("StatusCode", 1)
            if exit_code != 0:
                logs = container.logs().decode()
                raise RuntimeError(
                    f"Claude Code executable was not found in Docker image "
                    f"{image_name}: {logs}"
                )
        except DockerException as error:
            raise RuntimeError(
                f"Failed to check Claude Code executable in Docker image "
                f"{image_name}: {error}"
            ) from error

    @property
    def id(self) -> str:
        return self._id

    @property
    def configuration(self) -> Configuration:
        if "ANTHROPIC_API_KEY" in self._container_environment:
            credential = _AnthropicCompatibleApiKeyCredential(
                api_key=self._container_environment["ANTHROPIC_API_KEY"],
                base_url=self._container_environment.get("ANTHROPIC_BASE_URL", None),
            )
        elif "CLAUDE_OAUTH_TOKEN" in self._container_environment:
            credential = _ClaudeCodeOauthTokenCredential(
                token=self._container_environment["CLAUDE_OAUTH_TOKEN"]
            )
        else:
            raise ValueError("No valid credential found in container environment")

        return Configuration(
            id=self._id,
            model=_Model(name=self._model_name),
            credential=credential,
            directory=_Directory(
                challenge=self._container_challenge_directory,
                workspace=self._container_workspace_directory,
                metadata=self._container_metadata_directory,
            ),
            container=_Container(
                socket=self._docker_socket,
                image=self._docker_image,
            ),
            prompt=_PromptTemplate(
                user=self._user_prompt_template,
            ),
        )

    async def stream(
        self,
        challenge: Challenge,
        workspace_directory: Path,
        metadata_directory: Path,
        webhook: Webhook | None = None,
        additional_prompt: str | None = None,
    ) -> AsyncGenerator[Event[Message], Interrupt]:
        if not workspace_directory.exists():
            raise ValueError(f"workspace does not exist: {workspace_directory}")
        if not workspace_directory.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace_directory}")
        if not metadata_directory.exists():
            raise ValueError(f"metadata directory does not exist: {metadata_directory}")
        if not metadata_directory.is_dir():
            raise ValueError(
                f"metadata directory is not a directory: {metadata_directory}"
            )

        OmegaConf.save(
            config=OmegaConf.create(self.configuration.model_dump(mode="json")),
            f=metadata_directory / "configuration.yaml",
        )

        next_prompt = Template(self._user_prompt_template).render(
            challenge=challenge,
            webhook=webhook,
        )
        stripped_additional_prompt = (
            additional_prompt.strip() if additional_prompt is not None else ""
        )
        if stripped_additional_prompt:
            next_prompt = f"{next_prompt}\n\n{stripped_additional_prompt}"

        with self._docker_container(
            challenge=challenge,
            workspace_directory=workspace_directory,
            metadata_directory=metadata_directory,
        ) as container:
            container_claude_configuration_directory = (
                f"{self._container_metadata_directory}/.claude"
            )
            wrapper_path = self._write_cli_wrapper(
                metadata_directory=metadata_directory,
                container_id=container.id,
                container_claude_configuration_directory=container_claude_configuration_directory,
                container_user=f"{os.getuid() or 1000}:{os.getgid() or 1000}",
            )

            options = ClaudeAgentOptions(
                cli_path=wrapper_path,
                env=self._claude_code_environment,
                model=self._model_name,
                system_prompt={
                    "type": "preset",
                    "preset": "claude_code",
                },
                tools={"type": "preset", "preset": "claude_code"},
                permission_mode="bypassPermissions",
                include_partial_messages=True,
                continue_conversation=self._has_claude_sessions(metadata_directory),
                stderr=lambda line: _LOGGER.debug(
                    "(%s)(%s) Claude Code stderr: %s",
                    self._id,
                    challenge.id,
                    line,
                ),
            )

            async with ClaudeSDKClient(options=options) as client:
                while next_prompt is not None:
                    await client.query(next_prompt)
                    next_prompt = None

                    async for message in client.receive_response():
                        interrupt = yield ClaudeCodeEvent(raw=message)
                        restart_turn = False

                        match interrupt:
                            case Steer() as steer:
                                await client.query(steer.text)
                            case Prompt() as prompt_interrupt:
                                await client.interrupt()
                                next_prompt = prompt_interrupt.text
                                restart_turn = True
                            case Stop():
                                await client.interrupt()
                                return
                            case Nop():
                                ...

                        if restart_turn:
                            break

    @contextmanager
    def _docker_container(
        self, challenge: Challenge, workspace_directory: Path, metadata_directory: Path
    ) -> Generator[Any]:
        assert workspace_directory.is_dir()
        assert metadata_directory.is_dir()

        container_claude_configuration_directory = (
            f"{self._container_metadata_directory}/.claude"
        )

        container = self._docker_client.containers.run(
            self._docker_image,
            detach=True,
            stdin_open=True,
            environment={
                "HOME": self._container_workspace_directory,
                "CLAUDE_CONFIG_DIR": container_claude_configuration_directory,
                "CHROME_REMOTE_DEBUGGING_PORT": "9222",
            },
            ports={"9222/tcp": None},
            volumes={
                str(challenge.directory): {
                    "bind": self._container_challenge_directory,
                    "mode": "ro",
                },
                str(workspace_directory): {
                    "bind": self._container_workspace_directory,
                    "mode": "rw",
                },
                str(metadata_directory): {
                    "bind": self._container_metadata_directory,
                    "mode": "rw",
                },
            },
        )

        try:
            _LOGGER.info("(%s) Started Docker container: %s", self._id, container.id)
            self._prepare_claude_runtime(
                container, container_claude_configuration_directory
            )
            container.reload()
            yield container
        finally:
            _LOGGER.info(
                "(%s) Stopping and removing Docker container: %s",
                self._id,
                container.id,
            )
            container.remove(force=True)
            _LOGGER.info("(%s) Docker container removed: %s", self._id, container.id)

    def _prepare_claude_runtime(
        self, container: Container, container_claude_configuration_directory: str
    ) -> None:
        home = shlex.quote(self._container_workspace_directory)
        config_dir = shlex.quote(container_claude_configuration_directory)
        script = """
set -eu
uid=__CATCHY_UID__
gid=__CATCHY_GID__
home=__CATCHY_HOME__
config_dir=__CATCHY_CONFIG_DIR__

test ! -d /root || chmod 755 /root

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is not installed in the Claude Code image" >&2
  exit 1
fi

group_name="$(getent group "$gid" | cut -d: -f1 || true)"
if [ -z "$group_name" ]; then
  group_name="catchy_$gid"
  groupadd -g "$gid" "$group_name"
fi

user_name="$(getent passwd "$uid" | cut -d: -f1 || true)"
if [ -z "$user_name" ]; then
  user_name="catchy_$uid"
  useradd -u "$uid" -g "$gid" -M -d "$home" -s /bin/bash "$user_name"
fi

mkdir -p /etc/sudoers.d
printf '%s ALL=(ALL) NOPASSWD:ALL\\n' "$user_name" > /etc/sudoers.d/catchy-claude-code
chmod 0440 /etc/sudoers.d/catchy-claude-code

mkdir -p "$config_dir"
chown -R "$uid:$gid" "$config_dir"
"""
        script = (
            script.replace("__CATCHY_UID__", str(os.getuid() or 1000))
            .replace("__CATCHY_GID__", str(os.getgid() or 1000))
            .replace("__CATCHY_HOME__", home)
            .replace("__CATCHY_CONFIG_DIR__", config_dir)
        )
        result = container.exec_run(["sh", "-c", script])
        if result.exit_code != 0:
            raw_output = result.output if isinstance(result.output, bytes) else b""
            raise RuntimeError(
                f"Failed to prepare Claude Code runtime: {raw_output.decode()}"
            )

    def _configure_claude_configuration_directory(
        self, container: Any, container_claude_configuration_directory: str
    ) -> None:
        settings_path = (
            f"{shlex.quote(container_claude_configuration_directory)}/settings.json"
        )
        result = container.exec_run(
            [
                "sh",
                "-c",
                f"cat {settings_path} 2>/dev/null || true",
            ]
        )
        raw_output = getattr(result, "output", b"")
        existing_settings = self._json_object_from_bytes(
            raw_output if isinstance(raw_output, bytes) else b""
        )
        image_settings = self._image_claude_settings()
        settings = _deep_merge(image_settings, existing_settings)

        container.exec_run(["mkdir", "-p", container_claude_configuration_directory])
        archive = io.BytesIO()
        encoded_settings = json.dumps(settings, ensure_ascii=False).encode()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            info = tarfile.TarInfo("settings.json")
            info.size = len(encoded_settings)
            tar.addfile(info, io.BytesIO(encoded_settings))
        archive.seek(0)
        container.put_archive(
            container_claude_configuration_directory,
            archive.read(),
        )

    def _image_claude_settings(self) -> dict[str, Any]:
        try:
            raw_settings = cast(
                object, self._docker_client.containers.run(self._docker_image)
            )
        except DockerException:
            return {}
        if isinstance(raw_settings, bytes):
            return self._json_object_from_bytes(raw_settings)
        if isinstance(raw_settings, str):
            return self._json_object_from_bytes(raw_settings.encode())
        return {}

    def _json_object_from_bytes(self, value: bytes) -> dict[str, Any]:
        if not value:
            return {}
        try:
            parsed = cast(object, json.loads(value.decode()))
        except UnicodeDecodeError, json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {
            str(key): value for key, value in cast(dict[object, Any], parsed).items()
        }

    def _write_cli_wrapper(
        self,
        *,
        metadata_directory: Path,
        container_id: str,
        container_claude_configuration_directory: str,
        container_user: str,
    ) -> Path:
        wrapper_path = metadata_directory / "claude-docker-cli"
        environment = {
            "HOME": self._container_workspace_directory,
            "CLAUDE_CONFIG_DIR": container_claude_configuration_directory,
            "CHROME_REMOTE_DEBUGGING_PORT": os.environ.get(
                "CHROME_REMOTE_DEBUGGING_PORT", "9222"
            ),
            **self._claude_code_environment,
        }

        lines = [
            "#!/bin/sh",
            "set -eu",
            f"exec docker exec -i --user {shlex.quote(container_user)} --workdir {shlex.quote(self._container_workspace_directory)} "
            + " ".join(
                f"--env {shlex.quote(key)}={shlex.quote(value)}"
                for key, value in environment.items()
            )
            + f' {shlex.quote(container_id)} claude "$@"',
        ]

        wrapper_path.write_text("\n".join(lines) + "\n")
        wrapper_path.chmod(0o700)
        return wrapper_path

    @property
    def _claude_code_environment(self) -> dict[str, str]:
        container_environment = getattr(self, "_container_environment", None)
        if not isinstance(container_environment, dict):
            container_environment = {}
            model_api_key = getattr(self, "_model_api_key", None)
            model_base_url = getattr(self, "_model_base_url", None)
            if isinstance(model_api_key, str):
                container_environment["ANTHROPIC_API_KEY"] = model_api_key
            if isinstance(model_base_url, str):
                container_environment["ANTHROPIC_BASE_URL"] = model_base_url

        model_name = getattr(self, "_model_name", None)
        return {
            **cast(dict[str, str], container_environment),
            **({"ANTHROPIC_MODEL": model_name} if isinstance(model_name, str) else {}),
        }

    def _has_claude_sessions(self, metadata_directory: Path) -> bool:
        projects_directory = metadata_directory / ".claude" / "projects"
        return projects_directory.exists() and any(projects_directory.rglob("*.jsonl"))


class ClaudeCodeEventRenderer(EventRenderer[Message]):
    def __init__(
        self,
        *,
        model_name: str | None = None,
        challenge_id: str = "",
    ) -> None:
        self._model_name = model_name
        self._challenge_id = challenge_id
        self._stream_block_types: dict[int, str] = {}
        self._latest_usage: dict[str, int] = {}

    def render(self, event: Event[Message]) -> Iterator[Component]:
        match event.raw:
            case StreamEvent() as stream_event:
                payload: dict[str, Any] = stream_event.event
                event_type = payload.get("type")

                if event_type == "message_start":
                    message = payload.get("message")
                    if isinstance(message, dict):
                        message_dict = cast(dict[str, Any], message)
                        raw_usage = message_dict.get("usage")
                        if isinstance(raw_usage, dict):
                            yield self._token_usage_from(
                                cast(dict[str, Any], raw_usage)
                            )
                elif event_type == "message_delta":
                    raw_usage = payload.get("usage")
                    if isinstance(raw_usage, dict):
                        yield self._token_usage_from(cast(dict[str, Any], raw_usage))
                elif event_type == "content_block_start":
                    content_block = payload.get("content_block")
                    index = payload.get("index")
                    if isinstance(content_block, dict) and isinstance(index, int):
                        block_type = cast(dict[str, Any], content_block).get("type")
                        if isinstance(block_type, str):
                            self._stream_block_types[index] = block_type
                elif event_type == "content_block_delta":
                    delta = payload.get("delta")
                    if isinstance(delta, dict):
                        delta_dict = cast(dict[str, Any], delta)
                        delta_type = delta_dict.get("type")
                        if delta_type == "text_delta":
                            text = delta_dict.get("text")
                            if isinstance(text, str):
                                yield Delta(tag="agent", text=text)
                        elif delta_type == "thinking_delta":
                            thinking = delta_dict.get("thinking")
                            if isinstance(thinking, str):
                                yield Delta(tag="thinking", text=thinking)
                        elif delta_type == "input_json_delta":
                            partial_json = delta_dict.get("partial_json")
                            if isinstance(partial_json, str):
                                yield Delta(tag="tool_input", text=partial_json)
                elif event_type == "content_block_stop":
                    yield ItemCompleted()
                elif event_type == "error":
                    error = payload.get("error")
                    if isinstance(error, dict):
                        error_dict = cast(dict[str, Any], error)
                        error_type = error_dict.get("type", "unknown_error")
                        error_message = error_dict.get("message", "")
                        raise RuntimeError(
                            f"Claude Code stream error: {error_type}: {error_message}"
                        )
                    raise RuntimeError("Claude Code stream error")
            case ResultMessage() as result_message:
                yield Delta(tag="agent", text=result_message.result or "")
                yield TurnCompleted()
            case RateLimitEvent() as rate_limit_event:
                yield JsonLog(
                    tag="rate_limit",
                    data={
                        "status": rate_limit_event.rate_limit_info.status,
                        "resets_at": rate_limit_event.rate_limit_info.resets_at,
                        "rate_limit_type": rate_limit_event.rate_limit_info.rate_limit_type,
                        "utilization": rate_limit_event.rate_limit_info.utilization,
                        "overage_status": rate_limit_event.rate_limit_info.overage_status,
                        "overage_resets_at": rate_limit_event.rate_limit_info.overage_resets_at,
                        "overage_disabled_reason": rate_limit_event.rate_limit_info.overage_disabled_reason,
                        "raw": dict(rate_limit_event.rate_limit_info.raw),
                        "session_id": rate_limit_event.session_id,
                        "uuid": rate_limit_event.uuid,
                    },
                )

                if rate_limit_event.rate_limit_info.status == "rejected":
                    raise RuntimeError(
                        f"Claude Code request was rate limited: {rate_limit_event.rate_limit_info.utilization}% utilization"
                    )
            case TaskStartedMessage() as system_message:
                yield TurnStarted()
                yield Delta(tag="agent", text=system_message.description)
            case TaskProgressMessage() as system_message:
                yield Delta(tag="agent", text=system_message.description)
            case TaskNotificationMessage() as task_notification_message:
                yield Delta(tag="agent", text=task_notification_message.summary)
                match task_notification_message.status:
                    case "completed":
                        yield TurnCompleted()
                    case "failed":
                        raise RuntimeError(task_notification_message.summary)
                    case "stopped":
                        yield TurnCompleted()
            case MirrorErrorMessage() as mirror_error_message:
                yield TextLog(tag="mirror_error", text=mirror_error_message.error)
            case HookEventMessage() as hook_event_message:
                yield JsonLog(
                    tag=f"hook_{hook_event_message.hook_event_name or hook_event_message.subtype}",
                    data=hook_event_message.data,
                )
            case SystemMessage() as system_message:
                yield JsonLog(
                    tag=f"system_{system_message.subtype}",
                    data=system_message.data,
                )
            case AssistantMessage() as assistant_message:
                if isinstance(assistant_message.content, str):
                    yield Delta(tag="agent", text=assistant_message.content)
                else:
                    for content_block in assistant_message.content:
                        yield self._from_content_block(content_block)
            case UserMessage() as user_message:
                if isinstance(user_message.content, str):
                    yield Delta(tag="user", text=user_message.content)
                else:
                    for content_block in user_message.content:
                        yield self._from_content_block(content_block)
            case _:
                _LOGGER.debug(
                    "(%s) Unhandled Claude Code message type: %s",
                    self._challenge_id,
                    type(event.raw).__name__,
                )
                yield Nop()

    def _token_usage_from(self, usage: dict[str, Any]) -> TokenUsage:
        for key in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
        ):
            value = usage.get(key)
            if isinstance(value, int):
                self._latest_usage[key] = value

        input_tokens = self._latest_usage.get("input_tokens", 0)
        cache_read = self._latest_usage.get("cache_read_input_tokens", 0)
        cache_creation = self._latest_usage.get("cache_creation_input_tokens", 0)
        output_tokens = self._latest_usage.get("output_tokens", 0)

        return TokenUsage(
            cached_input_tokens=cache_read,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_output_tokens=0,
            total_tokens=input_tokens + cache_creation + cache_read + output_tokens,
        )

    def _from_content_block(self, content_block: ContentBlock) -> Component:
        match content_block:
            case TextBlock() as text_block:
                return Delta(tag="agent", text=text_block.text)
            case ThinkingBlock() as thinking_block:
                return Delta(tag="thinking", text=thinking_block.thinking)
            case ToolUseBlock() as tool_use_block:
                return JsonLog(
                    tag=tool_use_block.name,
                    data=tool_use_block.input,
                )
            case ServerToolUseBlock() as server_tool_use_block:
                return JsonLog(
                    tag=server_tool_use_block.name,
                    data=server_tool_use_block.input,
                )
            case ToolResultBlock() as tool_result_block:
                match tool_result_block.content:
                    case None:
                        text = ""
                    case str() as content:
                        text = content
                    case content:
                        text = json.dumps(content, indent=2)

                return Delta(tag="observation", text=text)
            case ServerToolResultBlock() as server_tool_result_block:
                return Delta(
                    tag="observation",
                    text=json.dumps(
                        server_tool_result_block.content,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    ),
                )
            case _:
                _LOGGER.info(
                    f"({self._challenge_id}) Unhandled Claude Code content block type: {type(content_block).__name__}"
                )
                return Nop()


class ClaudeCodeHandoffExporter(HandoffExporter[Message]):
    def __init__(self, *, model_name: str | None = None) -> None:
        self._renderer = ClaudeCodeEventRenderer(model_name=model_name)

    def export(self, event: Event[Message]) -> str:
        chunks: list[str] = []
        for component in self._renderer.render(event):
            markdown = self._component_markdown(component)
            if markdown:
                chunks.append(markdown)
        return "".join(chunks)

    def _component_markdown(self, component: Component) -> str:
        match component:
            case Delta() as delta:
                if not delta.text.strip():
                    return ""
                label = self._delta_label(delta.tag)
                if label is None:
                    return ""
                return f"- **{label}:** {delta.text.strip()}\n"
            case TextLog() | JsonLog() | TokenUsage() | ItemCompleted() | TurnStarted() | TurnCompleted() | ThreadStarted():
                return ""
            case Nop():
                return ""
        return ""

    def _delta_label(self, tag: str) -> str | None:
        return {
            "agent": "Assistant",
            "user": "User",
        }.get(tag)
