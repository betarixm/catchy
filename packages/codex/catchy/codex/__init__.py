from __future__ import annotations

import io
import json
import logging
import shlex
import tarfile
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, cast

import tomli_w
from catchy.core.agent.models import (
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
from catchy.core.agent.protocols import Agent
from catchy.core.challenge.models import Challenge
from catchy.core.webhook.models import Webhook
from codex_app_server import (
    AppServerConfig,
    AsyncCodex,
    TextInput,
)
from codex_app_server.generated.v2_all import (
    AgentMessageDeltaNotification,
    CommandExecutionOutputDeltaNotification,
    ErrorNotification,
    FileChangeOutputDeltaNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    McpToolCallProgressNotification,
    PlanDeltaNotification,
    ReasoningSummaryPartAddedNotification,
    ReasoningSummaryTextDeltaNotification,
    ReasoningTextDeltaNotification,
    TerminalInteractionNotification,
    ThreadListCwdFilter,
    ThreadTokenUsageUpdatedNotification,
    TurnDiffUpdatedNotification,
    TurnCompletedNotification,
    TurnPlanUpdatedNotification,
    TurnStatus,
)
from docker import DockerClient
from docker.errors import DockerException
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


class _OpenAICompatibleApiKeyCredential(BaseModel):
    api_key: str
    base_url: str | None = None
    organization_id: str | None = None


class _CodexAuthJsonCredential(BaseModel):
    json_string: str
    base_url: str | None = None

    @field_validator("json_string")
    @classmethod
    def _validate_json_string(cls, value: str) -> str:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("json_string must contain valid JSON") from exc

        if not isinstance(payload, dict):
            raise ValueError("json_string must contain a JSON object")

        return value


class _Model(BaseModel):
    name: str = "gpt-5.5"


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
    credential: _OpenAICompatibleApiKeyCredential | _CodexAuthJsonCredential
    directory: _Directory
    container: _Container
    prompt: _PromptTemplate


class CodexAgent(Agent):
    key: str = "codex"

    @staticmethod
    def from_configuration(configuration: Configuration) -> CodexAgent:
        match configuration.credential:
            case _OpenAICompatibleApiKeyCredential() as credential:
                auth_json_string = json.dumps(
                    {
                        "auth_mode": "apikey",
                        "OPENAI_API_KEY": credential.api_key,
                    }
                )
            case _CodexAuthJsonCredential() as credential:
                auth_json_string = credential.json_string

        return CodexAgent(
            id=configuration.id,
            model_name=configuration.model.name,
            container_challenge_directory=configuration.directory.challenge,
            container_workspace_directory=configuration.directory.workspace,
            container_metadata_directory=configuration.directory.metadata,
            docker_image=configuration.container.image,
            docker_client=DockerClient(base_url=configuration.container.socket),
            user_prompt_template=configuration.prompt.user,
            model_base_url=configuration.credential.base_url,
            auth_json_string=auth_json_string,
            docker_socket=configuration.container.socket,
        )

    def __init__(
        self,
        id: str,
        model_name: str,
        container_challenge_directory: str,
        container_workspace_directory: str,
        container_metadata_directory: str,
        docker_image: Image,
        docker_client: DockerClient,
        user_prompt_template: str,
        model_base_url: str | None = None,
        auth_json_string: str | None = None,
        model_api_key: str | None = None,
        model_organization_id: str | None = None,
        docker_socket: str = "/var/run/docker.sock",
    ):
        self._id = id
        self._model_name = model_name
        self._model_base_url = model_base_url
        self._auth_json_string = auth_json_string or self._auth_json_from_api_key(
            model_api_key=model_api_key,
            model_organization_id=model_organization_id,
        )
        self._model_api_key = model_api_key
        self._model_organization_id = model_organization_id
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
                command=["sh", "-c", "command -v codex >/dev/null 2>&1"],
                detach=True,
                remove=True,
            )
            result = container.wait()
            exit_code = result.get("StatusCode", 1)
            if exit_code != 0:
                logs = container.logs().decode()
                raise RuntimeError(
                    f"Codex executable was not found in Docker image {image_name}: "
                    f"{logs}"
                )
        except DockerException as error:
            raise RuntimeError(
                f"Failed to check Codex executable in Docker image {image_name}: {error}"
            ) from error

    @staticmethod
    def _auth_json_from_api_key(
        *,
        model_api_key: str | None,
        model_organization_id: str | None = None,
    ) -> str:
        if not model_api_key:
            raise ValueError("auth_json_string or model_api_key is required")

        payload = {
            "auth_mode": "apikey",
            "OPENAI_API_KEY": model_api_key,
        }
        if model_organization_id:
            payload["OPENAI_ORGANIZATION"] = model_organization_id
            payload["OPENAI_ORG_ID"] = model_organization_id
        return json.dumps(payload)

    @property
    def id(self) -> str:
        return self._id

    @property
    def configuration(self) -> Configuration:
        return Configuration(
            id=self._id,
            model=_Model(
                name=self._model_name,
            ),
            credential=_CodexAuthJsonCredential(
                json_string=self._auth_json_string,
                base_url=self._model_base_url,
            ),
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

        with self._docker_container(
            challenge=challenge,
            workspace=workspace,
            metadata_directory=metadata_directory,
        ) as container:
            async with AsyncCodex(
                config=AppServerConfig(
                    launch_args_override=(  # pyright: ignore[reportArgumentType]
                        "docker",
                        "exec",
                        "-i",
                        "--env",
                        f"HOME={self._container_workspace_directory}",
                        "--env",
                        f"CODEX_HOME={self._container_metadata_directory}/.codex",
                        container.id,
                        "codex",
                        "app-server",
                        "--listen",
                        "stdio://",
                    ),
                    client_name="catchy",
                    client_title="Catchy",
                    experimental_api=True,
                )
            ) as codex:
                thread = await self._resume_or_start_thread(codex, challenge.id)

                default_prompt = Template(self._user_prompt_template).render(
                    challenge=challenge,
                    webhook=webhook,
                )
                next_prompt: str | None = prompt or default_prompt

                while next_prompt is not None:
                    turn = await thread.turn(TextInput(next_prompt))
                    next_prompt = None

                    async for codex_event in turn.stream():
                        restart_turn = False
                        for event in self._events_from_codex_notification(
                            codex_event.method,
                            codex_event.payload,
                            turn_id=turn.id,
                            challenge_id=challenge.id,
                        ):
                            interrupt = yield event

                            match interrupt:
                                case Steer() as steer:
                                    await turn.steer(TextInput(steer.text))
                                case Prompt() as prompt_interrupt:
                                    await turn.interrupt()
                                    next_prompt = prompt_interrupt.text
                                    restart_turn = True
                                    break
                                case Stop():
                                    await turn.interrupt()
                                    return
                                case Nop():
                                    ...

                        if restart_turn:
                            break

    async def _resume_or_start_thread(self, codex: Any, challenge_id: str) -> Any:
        threads = (
            await codex.thread_list(
                archived=False,
                cwd=ThreadListCwdFilter(self._container_workspace_directory),
            )
        ).data

        match threads:
            case [thread]:
                _LOGGER.info(
                    f"({self.id})({challenge_id}) Resuming existing thread: {thread.id}"
                )
                return await codex.thread_resume(thread.id)
            case []:
                _LOGGER.info(f"({self.id})({challenge_id}) Starting new thread")
                return await codex.thread_start(
                    model=self._model_name,
                    cwd=self._container_workspace_directory,
                    service_name="catchy",
                    config={},  # TODO: support custom model config
                )
            case _:
                raise RuntimeError(
                    f"Expected at most one thread, but found {len(threads)}"
                )

    @contextmanager
    def _docker_container(
        self, challenge: Challenge, workspace: Path, metadata_directory: Path
    ):
        assert workspace.is_dir()
        assert metadata_directory.is_dir()

        codex_home = f"{self._container_metadata_directory}/.codex"

        container = self._docker_client.containers.run(
            self._docker_image,
            detach=True,
            stdin_open=True,
            # Codex uses bubblewrap for its Linux sandbox; Docker's default confinement blocks the user/mount namespace setup bwrap needs.
            cap_add=["SYS_ADMIN"],
            security_opt=["seccomp=unconfined", "apparmor=unconfined"],
            environment={
                "HOME": self._container_workspace_directory,
                "CODEX_HOME": codex_home,
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
            _LOGGER.info(f"({self._id}) Started Docker container: {container.id}")
            self._configure_codex_home(container, codex_home)
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
                    f"({self._id}) Chrome DevTools Protocol available after running "
                    f"`chrome-devtools`: {chrome_devtools_url}"
                )

            yield container
        finally:
            _LOGGER.info(
                f"({self._id}) Stopping and removing Docker container: {container.id}"
            )
            container.remove(force=True)
            _LOGGER.info(f"({self._id}) Docker container removed: {container.id}")

    def _events_from_codex_notification(
        self,
        method: str,
        payload: object,
        *,
        turn_id: str,
        challenge_id: str,
    ) -> list[Event]:
        match payload:
            case ItemStartedNotification() as notification if (
                notification.turn_id == turn_id
            ):
                chunk = self._chunk_from_started_item(notification.item)
                return [chunk] if chunk is not None else []
            case ItemCompletedNotification() as notification if (
                notification.turn_id == turn_id
            ):
                events = self._chunks_from_completed_item(notification.item)
                events.append(ItemCompleted())
                return events
            case ErrorNotification() as notification if notification.turn_id == turn_id:
                if notification.will_retry:
                    _LOGGER.warning(
                        "(%s)(%s) Codex turn error; server will retry: %s",
                        self._id,
                        challenge_id,
                        notification.error.message,
                    )
                    return []
                raise RuntimeError(
                    "Codex reported a non-retryable turn error: "
                    f"{notification.error.message}"
                )
            case TurnCompletedNotification() as notification if (
                notification.turn.id == turn_id
            ):
                return [self._event_from_turn_completed(notification)]
            case AgentMessageDeltaNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return self._delta_event("action", notification.delta)
            case PlanDeltaNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return self._delta_event("plan", notification.delta)
            case ReasoningSummaryTextDeltaNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return self._delta_event("thinking", notification.delta)
            case ReasoningTextDeltaNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return self._delta_event("thinking", notification.delta)
            case ReasoningSummaryPartAddedNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return [Chunk(tag="thinking", text="\n\n")]
            case CommandExecutionOutputDeltaNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return self._delta_event("observation", notification.delta)
            case FileChangeOutputDeltaNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return self._delta_event("observation", notification.delta)
            case McpToolCallProgressNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return self._log_event(
                    "tool_progress",
                    notification,
                    text=notification.message,
                )
            case TerminalInteractionNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return self._log_event(
                    "terminal_input",
                    notification,
                    text=notification.stdin,
                )
            case TurnPlanUpdatedNotification() as notification if (
                notification.turn_id == turn_id
            ):
                text = self._turn_plan_text(notification)
                return self._log_event("plan", notification, text=text) if text else []
            case TurnDiffUpdatedNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return (
                    self._log_event("diff", notification, text=notification.diff)
                    if notification.diff
                    else []
                )
            case ThreadTokenUsageUpdatedNotification() as notification if (
                notification.turn_id == turn_id
            ):
                return self._token_usage_event(notification)
            case _:
                _LOGGER.debug(
                    "(%s)(%s) Ignoring Codex event: %s",
                    self._id,
                    challenge_id,
                    method,
                )
                return []

    def _event_from_turn_completed(
        self, notification: TurnCompletedNotification
    ) -> Event:
        match notification.turn.status:
            case TurnStatus.completed:
                return TurnCompleted()
            case TurnStatus.failed:
                raise RuntimeError(
                    "Codex turn failed: "
                    f"{notification.turn.error.message if notification.turn.error else 'unknown error'}"
                )
            case TurnStatus.interrupted:
                raise RuntimeError(
                    "Codex turn was interrupted: "
                    f"{notification.turn.error.message if notification.turn.error else 'unknown error'}"
                )
            case TurnStatus.in_progress:
                raise RuntimeError(
                    "Codex emitted turn/completed while the turn is still in progress"
                )

    def _delta_event(self, tag: str, delta: str) -> list[Event]:
        return [Chunk(tag=tag, text=delta)] if delta else []

    def _log_event(
        self,
        kind: str,
        value: object,
        *,
        text: str | None = None,
    ) -> list[Event]:
        raw = self._json_model_payload(value)
        if text is None:
            text = json.dumps(raw, ensure_ascii=False) if raw else ""
        if not text and not raw:
            return []
        return [Log(kind=kind, text=text, raw=raw)]

    def _token_usage_event(
        self, notification: ThreadTokenUsageUpdatedNotification
    ) -> list[Event]:
        raw = self._json_model_payload(notification)
        usage = self._codex_token_usage_from_raw(raw)
        if usage is None:
            return []
        raw_total_tokens = usage.get("total_tokens") or usage.get("totalTokens")
        return [
            TokenUsage(
                provider="openai",
                model=getattr(self, "_model_name", None),
                source="thread_token_usage_updated",
                input_tokens=_int_value(
                    usage.get("input_tokens") or usage.get("inputTokens")
                ),
                cached_input_tokens=_int_value(
                    usage.get("cached_input_tokens") or usage.get("cachedInputTokens")
                ),
                output_tokens=_int_value(
                    usage.get("output_tokens") or usage.get("outputTokens")
                ),
                reasoning_output_tokens=_int_value(
                    usage.get("reasoning_output_tokens")
                    or usage.get("reasoningOutputTokens")
                ),
                total_tokens=_int_value(raw_total_tokens)
                if raw_total_tokens is not None
                else None,
                raw=raw,
            )
        ]

    def _codex_token_usage_from_raw(
        self, raw: dict[str, object]
    ) -> dict[str, Any] | None:
        token_usage = self._first_dict(
            raw.get("tokenUsage"),
            raw.get("token_usage"),
            raw.get("usage"),
        )
        usage = self._first_dict(
            token_usage.get("total"),
            token_usage.get("total_token_usage"),
            token_usage.get("last"),
            token_usage.get("last_token_usage"),
            token_usage,
        )
        if not usage:
            return None
        return usage

    def _first_dict(self, *values: object) -> dict[str, Any]:
        for value in values:
            if isinstance(value, dict):
                return cast(dict[str, Any], value)
        return {}

    def _chunk_from_started_item(self, item: object) -> Chunk | None:
        payload = self._thread_item_payload(item)
        item_type = payload.get("type")
        if not isinstance(item_type, str):
            return None
        if item_type in {"userMessage", "agentMessage", "plan", "reasoning"}:
            return None

        summary = self._summarize_started_item(payload)
        return Chunk(tag="tool_use", text=json.dumps(summary, ensure_ascii=False))

    def _chunks_from_completed_item(self, item: object) -> list[Event]:
        payload = self._thread_item_payload(item)
        item_type = payload.get("type")
        if item_type == "exitedReviewMode":
            review = payload.get("review")
            return [Chunk(tag="action", text=review)] if isinstance(review, str) and review else []
        if item_type in {"mcpToolCall", "dynamicToolCall", "collabAgentToolCall"}:
            summary = self._summarize_completed_tool_item(payload)
            return [Chunk(tag="observation", text=json.dumps(summary, ensure_ascii=False))]
        return []

    def _thread_item_payload(self, item: object) -> dict[str, Any]:
        root = getattr(item, "root", item)
        if isinstance(root, BaseModel):
            dumped = root.model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
                warnings=False,
            )
            return dumped
        return {}

    def _summarize_started_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_type = payload.get("type")
        summary: dict[str, Any] = {"type": item_type}
        for key in (
            "id",
            "status",
            "command",
            "cwd",
            "server",
            "tool",
            "query",
            "path",
            "review",
            "source",
            "senderThreadId",
            "receiverThreadIds",
        ):
            if key in payload:
                summary[key] = payload[key]

        if item_type == "fileChange" and isinstance(payload.get("changes"), list):
            summary["changes"] = payload["changes"]
        if item_type in {"dynamicToolCall", "mcpToolCall"} and "arguments" in payload:
            summary["arguments"] = payload["arguments"]
        if item_type == "webSearch" and "action" in payload:
            summary["action"] = payload["action"]
        return summary

    def _summarize_completed_tool_item(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        summary = self._summarize_started_item(payload)
        for key in ("success", "durationMs", "error", "result", "contentItems"):
            if key in payload:
                summary[key] = payload[key]
        return summary

    def _turn_plan_text(self, notification: TurnPlanUpdatedNotification) -> str:
        lines: list[str] = []
        if notification.explanation:
            lines.append(notification.explanation)
        for step in notification.plan:
            status = getattr(step.status, "value", str(step.status))
            lines.append(f"- [{status}] {step.step}")
        return "\n".join(lines)

    def _json_model_text(self, value: object) -> str:
        dumped = self._json_model_payload(value)
        return json.dumps(dumped, ensure_ascii=False) if dumped else ""

    def _json_model_payload(self, value: object) -> dict[str, object]:
        if isinstance(value, BaseModel):
            return value.model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
                warnings=False,
            )
        return {}

    def _configure_codex_home(self, container: Any, codex_home: str) -> None:
        runtime_config = self._read_container_toml(
            container, f"{codex_home}/config.toml"
        )

        self._put_container_files(
            container,
            codex_home,
            {
                "auth.json": self._codex_auth_json_string(),
                "config.toml": tomli_w.dumps(self._build_codex_config(runtime_config)),
            },
        )

    def _codex_auth_json_string(self) -> str:
        auth_json_string = getattr(self, "_auth_json_string", None)
        if isinstance(auth_json_string, str) and auth_json_string:
            return auth_json_string
        model_api_key = getattr(self, "_model_api_key", None)
        model_organization_id = getattr(self, "_model_organization_id", None)
        return self._auth_json_from_api_key(
            model_api_key=model_api_key if isinstance(model_api_key, str) else None,
            model_organization_id=model_organization_id
            if isinstance(model_organization_id, str)
            else None,
        )

    def _build_codex_config(
        self, runtime_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        config = _deep_merge(
            self._load_container_codex_config(),
            runtime_config or {},
        )
        config["model"] = self._model_name
        if self._model_base_url:
            config["openai_base_url"] = self._model_base_url
        return config

    def _read_container_toml(self, container: Any, path: str) -> dict[str, Any]:
        output = self._run_container_command(
            container,
            ["sh", "-c", f"cat {shlex.quote(path)} 2>/dev/null || true"],
        )
        return tomllib.loads(output.decode())

    def _put_container_files(
        self, container: Any, directory: str, files: dict[str, str]
    ) -> None:
        self._run_container_command(container, ["mkdir", "-p", directory])
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            for name, content in files.items():
                encoded = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(encoded)
                archive.addfile(info, io.BytesIO(encoded))
        archive_buffer.seek(0)
        if not container.put_archive(directory, archive_buffer.getvalue()):
            raise RuntimeError(f"failed to copy Codex configuration into {directory}")

    def _run_container_command(self, container: Any, command: list[str]) -> bytes:
        result = container.exec_run(command)
        exit_code = int(getattr(result, "exit_code", 1))
        output = getattr(result, "output", b"")
        if exit_code != 0:
            text = output.decode() if isinstance(output, bytes) else str(output)
            raise RuntimeError(f"container command failed: {command!r}: {text}")
        return output if isinstance(output, bytes) else str(output).encode()

    def _load_container_codex_config(self) -> dict[str, Any]:
        paths = [
            f"{self._container_metadata_directory}/.codex/config.toml",
            "/metadata/.codex/config.toml",
        ]
        for path in dict.fromkeys(paths):
            try:
                output = self._docker_client.containers.run(
                    self._docker_image,
                    command=["cat", path],
                    remove=True,
                )
            except DockerException:
                continue
            text = output.decode()
            return tomllib.loads(text)
        return {}
