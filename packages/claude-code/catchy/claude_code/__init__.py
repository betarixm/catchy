from __future__ import annotations

import io
import json
import logging
import os
import shlex
import tarfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, cast

from catchy.core.agent.models import (
    Chunk,
    Event,
    Interrupt,
    ItemCompleted,
    Nop,
    Prompt,
    Steer,
    Stop,
    TokenUsage,
    TurnCompleted,
)
from catchy.core.agent.protocols import Agent
from catchy.core.challenge.models import Challenge
from catchy.core.webhook.models import Webhook
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    ServerToolUseBlock,
    StreamEvent,
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
    ValidationInfo,
    field_serializer,
    field_validator,
)

_LOGGER = logging.getLogger(__name__)


def _list_of_strings() -> list[str]:
    return []


@dataclass
class _StreamBlock:
    index: int
    type: str
    content_block: dict[str, object]
    input_json_parts: list[str] = field(default_factory=_list_of_strings)
    text_parts: list[str] = field(default_factory=_list_of_strings)
    thinking_parts: list[str] = field(default_factory=_list_of_strings)
    signature: str | None = None


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


class ClaudeCodeAgent(Agent):
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
        self._stream_blocks: dict[int, _StreamBlock] = {}
        self._stream_message_usage: dict[str, object] = {}
        self._stream_message_stop_reason: str | None = None
        self._saw_stream_event_in_turn = False

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
        workspace: Path,
        metadata_directory: Path,
        webhook: Webhook | None = None,
        prompt: str | None = None,
    ) -> AsyncGenerator[Event, Interrupt]:
        if not workspace.exists():
            raise ValueError(f"workspace does not exist: {workspace}")
        if not workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
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

        default_prompt = Template(self._user_prompt_template).render(
            challenge=challenge,
            webhook=webhook,
        )
        next_prompt: str | None = prompt or default_prompt

        with self._docker_container(
            challenge=challenge,
            workspace=workspace,
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
                    self._reset_stream_accumulator()
                    await client.query(next_prompt)
                    next_prompt = None

                    async for message in client.receive_response():
                        restart_turn = False
                        for event in self._events_from_message(message):
                            interrupt = yield event

                            match interrupt:
                                case Steer() as steer:
                                    await client.query(steer.text)
                                case Prompt() as prompt_interrupt:
                                    await client.interrupt()
                                    next_prompt = prompt_interrupt.text
                                    restart_turn = True
                                    break
                                case Stop():
                                    await client.interrupt()
                                    return
                                case Nop():
                                    ...

                        if restart_turn:
                            break

    @contextmanager
    def _docker_container(
        self, challenge: Challenge, workspace: Path, metadata_directory: Path
    ) -> Generator[Any]:
        assert workspace.is_dir()
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
                str(workspace): {
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
            chrome_devtools_bindings = container.attrs["NetworkSettings"]["Ports"].get(
                "9222/tcp"
            )
            if chrome_devtools_bindings:
                chrome_devtools_url = (
                    f"http://{chrome_devtools_bindings[0]['HostIp']}:"
                    f"{chrome_devtools_bindings[0]['HostPort']}"
                )
                _LOGGER.info(
                    "(%s) Chrome DevTools Protocol available after running "
                    "`chrome-devtools`: %s",
                    self._id,
                    chrome_devtools_url,
                )

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

    def _reset_stream_accumulator(self) -> None:
        self._stream_blocks = {}
        self._stream_message_usage = {}
        self._stream_message_stop_reason = None
        self._saw_stream_event_in_turn = False

    def _stream_block_state(self) -> dict[int, _StreamBlock]:
        raw_blocks = getattr(self, "_stream_blocks", None)
        if not isinstance(raw_blocks, dict):
            self._stream_blocks = {}
            return self._stream_blocks
        return cast(dict[int, _StreamBlock], raw_blocks)

    def _event_from_message(self, message: object) -> Event:
        events = self._events_from_message(message)
        return events[-1] if events else Nop()

    def _events_from_message(self, message: object) -> list[Event]:
        if isinstance(message, StreamEvent):
            return self._events_from_stream_event(message)
        if isinstance(message, AssistantMessage):
            if getattr(self, "_saw_stream_event_in_turn", False):
                return []
            events = self._events_from_assistant_message(message)
            usage_event = self._usage_event_from_assistant_message(message)
            if usage_event is not None:
                events.append(usage_event)
            return events
        if isinstance(message, UserMessage):
            return [self._event_from_user_message(message)]
        if isinstance(message, ResultMessage):
            if message.is_error:
                details = message.result or ", ".join(message.errors or [])
                raise RuntimeError(
                    f"Claude Code turn failed: {details or message.subtype}"
                )
            events: list[Event] = []
            usage_event = self._usage_event_from_result_message(message)
            if usage_event is not None:
                events.append(usage_event)
            events.append(TurnCompleted())
            return events
        return []

    def _event_from_stream_event(self, message: StreamEvent) -> Event:
        events = self._events_from_stream_event(message)
        return events[-1] if events else Nop()

    def _events_from_stream_event(self, message: StreamEvent) -> list[Event]:
        self._saw_stream_event_in_turn = True
        event = cast(dict[str, object], message.event)
        event_type = event.get("type")

        if event_type == "message_start":
            self._stream_blocks = {}
            self._stream_message_usage = self._usage_from_message_start(event)
            self._stream_message_stop_reason = None
            return self._usage_events_from_stream_state(message, event_type=event_type)

        if event_type == "ping" or event_type == "message_stop":
            if event_type == "message_stop":
                self._stream_blocks = {}
            return []

        if event_type == "error":
            raise RuntimeError(self._stream_error_text(event))

        if event_type == "content_block_start":
            raw_content_block = event.get("content_block")
            if isinstance(raw_content_block, dict):
                content_block = cast(dict[str, object], raw_content_block)
            else:
                content_block = {}
            index = self._event_index(event)
            content_block_type = self._string_value(content_block.get("type"))
            if index is not None:
                self._stream_block_state()[index] = _StreamBlock(
                    index=index,
                    type=content_block_type or "unknown",
                    content_block=content_block,
                )

            if content_block_type in {"tool_use", "server_tool_use"}:
                raw_tool_name = content_block.get("name")
                tool_name = raw_tool_name if isinstance(raw_tool_name, str) else "tool"
                return [
                    Chunk(
                        tag="tool_use",
                        text=self._tool_use_text(
                            content_block, fallback_name=tool_name
                        ),
                    )
                ]

            if content_block_type.endswith("_tool_result"):
                text = self._server_tool_result_text(content_block)
                return [Chunk(tag="observation", text=text)] if text else []

        if event_type == "content_block_delta":
            raw_delta = event.get("delta")
            if isinstance(raw_delta, dict):
                delta = cast(dict[str, object], raw_delta)
                block = self._block_for_delta(event)
                if delta.get("type") == "text_delta":
                    raw_text = delta.get("text")
                    text = raw_text if isinstance(raw_text, str) else ""
                    if block is not None:
                        block.text_parts.append(text)
                    return [Chunk(tag="action", text=text)] if text else []
                if delta.get("type") == "input_json_delta":
                    raw_partial_json = delta.get("partial_json")
                    partial_json = (
                        raw_partial_json if isinstance(raw_partial_json, str) else ""
                    )
                    if block is not None:
                        block.input_json_parts.append(partial_json)
                    return (
                        [Chunk(tag="tool_input", text=partial_json)]
                        if partial_json
                        else []
                    )
                if delta.get("type") == "thinking_delta":
                    raw_thinking = delta.get("thinking")
                    thinking = raw_thinking if isinstance(raw_thinking, str) else ""
                    if block is not None:
                        block.thinking_parts.append(thinking)
                    return [Chunk(tag="thinking", text=thinking)] if thinking else []
                if delta.get("type") == "signature_delta":
                    raw_signature = delta.get("signature")
                    if block is not None and isinstance(raw_signature, str):
                        block.signature = raw_signature
                    return []

        if event_type == "content_block_stop":
            events: list[Event] = []
            block = self._block_for_delta(event)
            if block is not None:
                final_tool_input = self._final_tool_input_text(block)
                if final_tool_input:
                    events.append(Chunk(tag="tool_input", text=final_tool_input))
                self._stream_block_state().pop(block.index, None)
            events.append(ItemCompleted())
            return events

        if event_type == "message_delta":
            self._apply_message_delta(event)
            return self._usage_events_from_stream_state(message, event_type=event_type)

        _LOGGER.debug("Ignoring unknown Claude stream event type: %r", event_type)
        return []

    def _events_from_assistant_message(self, message: AssistantMessage) -> list[Event]:
        events: list[Event] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text:
                    events.append(Chunk(tag="action", text=block.text))
                events.append(ItemCompleted())
            elif isinstance(block, ThinkingBlock):
                if block.thinking:
                    events.append(Chunk(tag="thinking", text=block.thinking))
                events.append(ItemCompleted())
            elif isinstance(block, ToolUseBlock):
                events.append(
                    Chunk(
                        tag="tool_use",
                        text=self._tool_use_text(
                            {
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            },
                            fallback_name=block.name,
                        ),
                    )
                )
                events.append(ItemCompleted())
            elif isinstance(block, ServerToolUseBlock):
                events.append(
                    Chunk(
                        tag="tool_use",
                        text=self._tool_use_text(
                            {
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            },
                            fallback_name=block.name,
                        ),
                    )
                )
                events.append(ItemCompleted())
            elif isinstance(block, ToolResultBlock):
                text = self._tool_result_text(block.content)
                if text:
                    events.append(Chunk(tag="observation", text=text))
                events.append(ItemCompleted())
            else:
                text = self._tool_result_text(block.content)
                if text:
                    events.append(Chunk(tag="observation", text=text))
                events.append(ItemCompleted())
        return events

    def _event_index(self, event: dict[str, object]) -> int | None:
        raw_index = event.get("index")
        if isinstance(raw_index, bool):
            return None
        if isinstance(raw_index, int):
            return raw_index
        return None

    def _block_for_delta(self, event: dict[str, object]) -> _StreamBlock | None:
        index = self._event_index(event)
        if index is None:
            return None
        return self._stream_block_state().get(index)

    def _final_tool_input_text(self, block: _StreamBlock) -> str:
        if block.type not in {"tool_use", "server_tool_use"}:
            return ""
        raw_input = "".join(block.input_json_parts)
        if not raw_input:
            return ""
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            return ""
        if not isinstance(parsed, dict):
            return ""
        block.content_block["input"] = parsed
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)

    def _apply_message_delta(self, event: dict[str, object]) -> None:
        raw_delta = event.get("delta")
        if isinstance(raw_delta, dict):
            delta = cast(dict[str, object], raw_delta)
            stop_reason = delta.get("stop_reason")
            if isinstance(stop_reason, str):
                self._stream_message_stop_reason = stop_reason

        raw_usage = event.get("usage")
        if isinstance(raw_usage, dict):
            usage = cast(dict[str, object], raw_usage)
            self._stream_message_usage = {
                str(key): value for key, value in usage.items()
            }

    def _usage_from_message_start(self, event: dict[str, object]) -> dict[str, object]:
        raw_message = event.get("message")
        if not isinstance(raw_message, dict):
            return {}
        message = cast(dict[str, object], raw_message)
        raw_usage = message.get("usage")
        if not isinstance(raw_usage, dict):
            return {}
        usage = cast(dict[str, object], raw_usage)
        return {str(key): value for key, value in usage.items()}

    def _usage_events_from_stream_state(
        self,
        message: StreamEvent,
        *,
        event_type: object,
    ) -> list[Event]:
        usage = getattr(self, "_stream_message_usage", {})
        if not isinstance(usage, dict) or not usage:
            return []
        typed_usage = cast(dict[object, object], usage)
        raw: dict[str, object] = {
            "event_type": event_type if isinstance(event_type, str) else "event",
            "session_id": message.session_id,
            "usage": {str(key): value for key, value in typed_usage.items()},
        }
        stop_reason = getattr(self, "_stream_message_stop_reason", None)
        if isinstance(stop_reason, str):
            raw["stop_reason"] = stop_reason
        event = self._usage_event_from_usage(
            cast(dict[str, object], raw["usage"]),
            source="stream_event",
            raw=raw,
        )
        return [event] if event is not None else []

    def _usage_event_from_assistant_message(
        self, message: AssistantMessage
    ) -> TokenUsage | None:
        if not isinstance(message.usage, dict) or not message.usage:
            return None
        raw: dict[str, object] = {
            "usage": {str(key): value for key, value in message.usage.items()},
        }
        if message.stop_reason:
            raw["stop_reason"] = message.stop_reason
        if message.session_id:
            raw["session_id"] = message.session_id
        if message.message_id:
            raw["message_id"] = message.message_id
        return self._usage_event_from_usage(
            cast(dict[str, object], raw["usage"]),
            source="assistant_message",
            model=message.model,
            raw=raw,
        )

    def _usage_event_from_result_message(
        self, message: ResultMessage
    ) -> TokenUsage | None:
        raw_usage = message.usage or message.model_usage
        if not isinstance(raw_usage, dict) or not raw_usage:
            return None
        raw: dict[str, object] = {
            "subtype": message.subtype,
            "session_id": message.session_id,
            "duration_ms": message.duration_ms,
            "duration_api_ms": message.duration_api_ms,
            "num_turns": message.num_turns,
            "usage": {str(key): value for key, value in raw_usage.items()},
        }
        if message.stop_reason:
            raw["stop_reason"] = message.stop_reason
        if isinstance(message.model_usage, dict) and message.model_usage:
            raw["model_usage"] = {
                str(key): value for key, value in message.model_usage.items()
            }
        return self._usage_event_from_usage(
            cast(dict[str, object], raw["usage"]),
            source="result_message",
            raw=raw,
        )

    def _usage_event_from_usage(
        self,
        usage: dict[str, object],
        *,
        source: str,
        raw: dict[str, object],
        model: str | None = None,
    ) -> TokenUsage | None:
        if not usage:
            return None
        return TokenUsage(
            provider="anthropic",
            model=model or getattr(self, "_model_name", None),
            source=source,
            input_tokens=self._int_value(usage.get("input_tokens")),
            cache_creation_input_tokens=self._int_value(
                usage.get("cache_creation_input_tokens")
            ),
            cache_read_input_tokens=self._int_value(
                usage.get("cache_read_input_tokens")
            ),
            output_tokens=self._int_value(usage.get("output_tokens")),
            raw=raw,
        )

    def _int_value(self, value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdecimal():
            return int(value)
        return 0

    def _stream_error_text(self, event: dict[str, object]) -> str:
        raw_error = event.get("error")
        if not isinstance(raw_error, dict):
            return "Claude Code stream error"

        error = cast(dict[str, object], raw_error)
        error_type = error.get("type")
        message = error.get("message")
        if isinstance(error_type, str) and isinstance(message, str):
            return f"Claude Code stream error: {error_type}: {message}"
        if isinstance(message, str):
            return f"Claude Code stream error: {message}"
        if isinstance(error_type, str):
            return f"Claude Code stream error: {error_type}"
        return "Claude Code stream error"

    def _string_value(self, value: object) -> str:
        return value if isinstance(value, str) else ""

    def _server_tool_result_text(self, content_block: dict[str, object]) -> str:
        content_block_type = self._string_value(content_block.get("type"))
        payload: dict[str, object] = {"type": content_block_type}

        raw_tool_use_id = content_block.get("tool_use_id")
        if isinstance(raw_tool_use_id, str):
            payload["tool_use_id"] = raw_tool_use_id

        raw_content = content_block.get("content")
        if isinstance(raw_content, list):
            payload["content"] = self._summarize_tool_result_content(
                cast(list[object], raw_content)
            )
        elif isinstance(raw_content, dict):
            payload["content"] = raw_content
        elif isinstance(raw_content, str):
            payload["content"] = raw_content

        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _summarize_tool_result_content(
        self, content: list[object]
    ) -> list[dict[str, object]]:
        summarized: list[dict[str, object]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            typed_item = cast(dict[str, object], item)
            item_type = typed_item.get("type")
            if item_type == "web_search_result":
                summary: dict[str, object] = {"type": "web_search_result"}
                for key in ("title", "url", "page_age"):
                    value = typed_item.get(key)
                    if isinstance(value, str) or value is None:
                        summary[key] = value
                summarized.append(summary)
            else:
                summarized.append(
                    {str(key): value for key, value in typed_item.items()}
                )
        return summarized

    def _tool_use_text(
        self, content_block: dict[str, object], *, fallback_name: str
    ) -> str:
        payload: dict[str, object] = {"name": fallback_name}
        raw_id = content_block.get("id")
        if isinstance(raw_id, str):
            payload["id"] = raw_id
        raw_input = content_block.get("input")
        if isinstance(raw_input, dict):
            payload["input"] = raw_input
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _event_from_user_message(self, message: UserMessage) -> Event:
        if not isinstance(message.content, list):
            return Nop()

        chunks: list[str] = []
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                text = self._tool_result_text(block.content)
                if text:
                    chunks.append(text)

        if chunks:
            return Chunk(tag="observation", text="\n".join(chunks))
        return Nop()

    def _tool_result_text(self, content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
