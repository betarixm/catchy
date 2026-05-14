from __future__ import annotations

import io
import json
import logging
import tarfile
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, AsyncGenerator, Iterator, Literal

import tomli_w
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
from catchy.core.agent.protocols import Agent
from catchy.core.challenge.models import Challenge
from catchy.core.webhook.models import Webhook
from codex_app_server import (
    AppServerConfig,
    AsyncCodex,
    TextInput,
)
from codex_app_server.generated.notification_registry import NOTIFICATION_MODELS
from codex_app_server.generated.v2_all import (
    AgentMessageDeltaNotification,
    CommandExecutionOutputDeltaNotification,
    ContextCompactedNotification,
    ErrorNotification,
    FileChangeOutputDeltaNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    McpToolCallProgressNotification,
    PlanDeltaNotification,
    ReasoningSummaryTextDeltaNotification,
    ReasoningTextDeltaNotification,
    TerminalInteractionNotification,
    ThreadStartedNotification,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
    TurnDiffUpdatedNotification,
    TurnPlanUpdatedNotification,
    TurnStartedNotification,
    TurnStatus,
)
from codex_app_server.models import Notification, UnknownNotification
from docker import DockerClient
from docker.errors import DockerException
from docker.models.containers import Container
from docker.models.images import Image
from jinja2 import Template
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    GetPydanticSchema,
    ValidationError,
    ValidationInfo,
    field_serializer,
    field_validator,
)
from pydantic_core import core_schema

_LOGGER = logging.getLogger(__name__)

type NotificationValue = Annotated[
    Notification,
    GetPydanticSchema(lambda _source, _handler: core_schema.any_schema()),
]


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


class CodexEvent(Event[NotificationValue]):
    format: str = "codex-notification"

    @field_validator("raw", mode="before")
    @classmethod
    def _deserialize_raw(cls, value: object) -> Notification:
        if isinstance(value, Notification):
            return value

        if isinstance(value, str):
            value = json.loads(value)

        if not isinstance(value, dict):
            raise ValidationError(
                f"raw value must be a dict, got {type(value).__name__}"
            )

        method, payload = value.get("method"), value.get("payload")  # type: ignore
        if not isinstance(method, str):
            raise ValidationError(
                f"raw value must contain a 'method' field of type str, got {method}"
            )

        Model = NOTIFICATION_MODELS[method]

        return Notification(method=method, payload=Model.model_validate(payload))  # type: ignore

    @field_serializer("raw")
    def _serialize_raw(self, value: Notification) -> str:
        if isinstance(value.payload, UnknownNotification):
            raise ValueError(
                f"Cannot serialize Notification with unknown payload: {value}"
            )

        return json.dumps(
            {"method": value.method, "payload": value.payload.model_dump(mode="json")},
            ensure_ascii=False,
        )


class CodexAgent(Agent[Notification]):
    key: str = "codex"

    @staticmethod
    def from_configuration(configuration: Configuration) -> CodexAgent:
        match configuration.credential:
            case _OpenAICompatibleApiKeyCredential() as credential:
                auth_json_string = json.dumps(
                    {
                        "auth_mode": "apikey",
                        "OPENAI_API_KEY": credential.api_key,
                        **(
                            {
                                "OPENAI_ORGANIZATION": credential.organization_id,
                                "OPENAI_ORG_ID": credential.organization_id,
                            }
                            if credential.organization_id
                            else {}
                        ),
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
        auth_json_string: str,
        model_base_url: str | None = None,
        model_organization_id: str | None = None,
        docker_socket: str = "/var/run/docker.sock",
    ):
        self._id = id
        self._model_name = model_name
        self._model_base_url = model_base_url
        self._auth_json_string = auth_json_string
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
        workspace_directory: Path,
        metadata_directory: Path,
        webhook: Webhook | None = None,
    ) -> AsyncGenerator[Event[Notification], Interrupt]:
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

        self._seed_metadata_directory_from_image(metadata_directory)

        OmegaConf.save(
            config=OmegaConf.create(self.configuration.model_dump(mode="json")),
            f=metadata_directory / "configuration.yaml",
        )

        with self._docker_container(
            challenge=challenge,
            workspace_directory=workspace_directory,
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
                match (await codex.thread_list()).data:
                    case []:
                        thread = await codex.thread_start(
                            model=self._model_name,
                            cwd=self._container_workspace_directory,
                            service_name="catchy",
                            config={},  # TODO: support custom model config
                        )
                    case [thread]:
                        _LOGGER.info(
                            f"({self.id})({challenge.id}) Resuming existing thread: {thread.id}"
                        )
                        thread = await codex.thread_resume(thread.id)
                    case threads:
                        raise RuntimeError(
                            f"Expected at most one thread, but found {len(threads)}"
                        )

                next_prompt = Template(self._user_prompt_template).render(
                    challenge=challenge,
                    webhook=webhook,
                )

                while next_prompt is not None:
                    turn = await thread.turn(TextInput(next_prompt))
                    next_prompt = None

                    async for codex_event in turn.stream():
                        interrupt = yield CodexEvent(raw=codex_event)

                        restart_turn = False
                        match interrupt:
                            case Steer() as steer:
                                await turn.steer(TextInput(steer.text))
                            case Prompt() as prompt_interrupt:
                                await turn.interrupt()
                                next_prompt = prompt_interrupt.text
                                restart_turn = True
                            case Stop():
                                await turn.interrupt()
                                return
                            case Nop():
                                ...

                        if restart_turn:
                            break

    @contextmanager
    def _docker_container(
        self, challenge: Challenge, workspace_directory: Path, metadata_directory: Path
    ):
        assert workspace_directory.is_dir()
        assert metadata_directory.is_dir()

        container_codex_home = f"{self._container_metadata_directory}/.codex"

        container = self._docker_client.containers.run(
            self._docker_image,
            detach=True,
            stdin_open=True,
            # Codex uses bubblewrap for its Linux sandbox; Docker's default confinement blocks the user/mount namespace setup bwrap needs.
            cap_add=["SYS_ADMIN"],
            security_opt=["seccomp=unconfined", "apparmor=unconfined"],
            environment={
                "HOME": self._container_workspace_directory,
                "CODEX_HOME": container_codex_home,
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
            _LOGGER.info(f"({self._id}) Started Docker container: {container.id}")
            self._configure_codex_home(container, container_codex_home)
            container.reload()
            yield container
        finally:
            _LOGGER.info(
                f"({self._id}) Stopping and removing Docker container: {container.id}"
            )
            container.remove(force=True)
            _LOGGER.info(f"({self._id}) Docker container removed: {container.id}")

    def _seed_metadata_directory_from_image(self, metadata_directory: Path) -> None:
        host_codex_home = metadata_directory / ".codex"
        host_config_toml = host_codex_home / "config.toml"
        if host_config_toml.exists():
            return

        host_codex_home.mkdir(parents=True, exist_ok=True)
        container_config_toml = (
            f"{self._container_metadata_directory}/.codex/config.toml"
        )
        probe_container = self._docker_client.containers.create(
            self._docker_image,
            command=["cat", container_config_toml],
        )

        try:
            probe_container.start()
            result = probe_container.wait()
            exit_code = result.get("StatusCode", 1)
            output = probe_container.logs()
            if exit_code != 0:
                raise RuntimeError(
                    "Failed to seed metadata .codex/config.toml from Docker image at "
                    f"{container_config_toml}: {output.decode()}"
                )

            host_config_toml.write_bytes(output)
            _LOGGER.info(
                "(%s) Seeded metadata .codex/config.toml from Docker image path %s",
                self._id,
                container_config_toml,
            )
        finally:
            probe_container.remove(force=True)

    def _configure_codex_home(
        self, container: Container, container_codex_home: str
    ) -> None:
        result = container.exec_run(["cat", f"{container_codex_home}/config.toml"])
        if result.exit_code != 0:
            raise RuntimeError(
                f"Codex home config.toml not found in container; initializing Codex home at {container_codex_home}: {result.output.decode()}"
            )

        config_toml = tomllib.loads(result.output.decode())

        self._put_container_files(
            container,
            container_codex_home,
            {
                "auth.json": self._auth_json_string,
                "config.toml": tomli_w.dumps(
                    {
                        **config_toml,
                        "model": self._model_name,
                        **(
                            {"openai_base_url": self._model_base_url}
                            if self._model_base_url
                            else {}
                        ),
                    }
                ),
            },
        )

    def _put_container_files(
        self, container: Container, directory: str, files: dict[str, str]
    ) -> None:
        result = container.exec_run(["mkdir", "-p", directory])
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to create directory {directory} in container: {result.output.decode()}"
            )
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            for name, content in files.items():
                encoded = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(encoded)
                archive.addfile(info, io.BytesIO(encoded))
        archive_buffer.seek(0)
        if not container.put_archive(directory, archive_buffer.getvalue()):  # pyright: ignore[reportUnknownMemberType]
            raise RuntimeError(f"failed to copy Codex configuration into {directory}")


class CodexEventRenderer(EventRenderer[Notification]):
    def __init__(
        self,
        *,
        model_name: str | None = None,
        turn_id: str | None = None,
        challenge_id: str = "",
    ) -> None:
        self._model_name = model_name
        self._challenge_id = challenge_id

    def render(self, event: Event[Notification]) -> Iterator[Component]:
        match event.raw.payload:
            case AgentMessageDeltaNotification() as notification:
                yield Delta(tag="agent", text=notification.delta)
            case PlanDeltaNotification() as notification:
                yield Delta(tag="thinking", text=notification.delta)
            case ReasoningSummaryTextDeltaNotification() as notification:
                yield Delta(tag="thinking", text=notification.delta)
            case ReasoningTextDeltaNotification() as notification:
                yield Delta(tag="thinking", text=notification.delta)
            case CommandExecutionOutputDeltaNotification() as notification:
                yield Delta(tag="observation", text=notification.delta)
            case FileChangeOutputDeltaNotification() as notification:
                yield Delta(tag="observation", text=notification.delta)
            case McpToolCallProgressNotification() as notification:
                yield TextLog(
                    tag="mcp_tool_call_progress",
                    text=notification.message,
                )

            case TerminalInteractionNotification() as notification:
                yield Delta(
                    tag="tool_input",
                    text=notification.stdin,
                )

            case TurnPlanUpdatedNotification() as notification:
                lines: list[str] = []
                if notification.explanation:
                    lines.append(notification.explanation)
                for step in notification.plan:
                    status = getattr(step.status, "value", str(step.status))
                    lines.append(f"- [{status}] {step.step}")

                yield TextLog(tag="turn_plan_updated", text="\n".join(lines))
            case TurnDiffUpdatedNotification() as notification:
                yield TextLog(
                    tag="turn_diff_updated",
                    text=notification.diff or "",
                )

            case ThreadTokenUsageUpdatedNotification() as notification:
                yield TokenUsage(
                    cached_input_tokens=notification.token_usage.last.cached_input_tokens,
                    input_tokens=notification.token_usage.last.input_tokens,
                    output_tokens=notification.token_usage.last.output_tokens,
                    reasoning_output_tokens=notification.token_usage.last.reasoning_output_tokens,
                    total_tokens=notification.token_usage.last.total_tokens,
                )

            case ItemStartedNotification() as notification:
                yield JsonLog(
                    tag="item_started",
                    data=notification.item.root.model_dump(),
                )

            case ItemCompletedNotification():
                yield ItemCompleted()
            case TurnCompletedNotification() as notification:
                match notification.turn.status:
                    case TurnStatus.completed:
                        yield TurnCompleted()
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
            case ErrorNotification() as notification if notification.will_retry:
                yield Nop()
            case ErrorNotification() as notification:
                raise RuntimeError(notification.error.message)
            case ContextCompactedNotification() as notification:
                yield TextLog(
                    tag="context_compacted",
                    text="",
                )

            case ThreadStartedNotification():
                yield ThreadStarted()
            case TurnStartedNotification():
                yield TurnStarted()
            case _:
                _LOGGER.debug(
                    f"({self._challenge_id}) Unhandled Codex notification type: {type(event.raw.payload).__name__}"
                )
                yield Nop()
