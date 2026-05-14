from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import select
import shutil
import socket
import tarfile
import threading
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from docker import DockerClient
from docker.errors import DockerException

from catchy.codex import CodexAgent
from catchy.core.agent.models import (
    Chunk,
    ItemCompleted,
    Log,
    TokenUsage,
    TurnCompleted,
)
from catchy.core.challenge.models import Challenge
from codex_app_server.generated.v2_all import (
    AgentMessageDeltaNotification,
    CommandExecutionOutputDeltaNotification,
    ErrorNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    ReasoningTextDeltaNotification,
    ThreadTokenUsageUpdatedNotification,
    TurnDiffUpdatedNotification,
    TurnCompletedNotification,
)

_CODEX_IMAGE = "ghcr.io/betarixm/catchy-codex:latest"
_DOCKER_SOCKET = "/var/run/docker.sock"
_CHALLENGE_ROOT = (
    Path(__file__).parent / "fixtures" / "challenges" / "lets-change"
).resolve()
_STREAM_OUTPUT_PATH = (
    Path(__file__).parent / "fixtures" / "stream_outputs" / "lets_change_stream.json"
)
_STREAM_OK_MARKER = "CATCHY_STREAM_OK"


def test_codex_config_merges_container_and_runtime_toml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(CodexAgent)
    setattr(agent, "_model_name", "gpt-5.5")
    setattr(agent, "_model_base_url", "https://example.test/v1")
    setattr(agent, "_model_organization_id", "org_123")

    def container_config(self: CodexAgent) -> dict[str, Any]:
        return {
            "sandbox_mode": "danger-full-access",
            "approval_policy": "never",
            "otel": {
                "environment": "dev",
                "exporter": "none",
                "log_user_prompt": True,
            },
        }

    monkeypatch.setattr(CodexAgent, "_load_container_codex_config", container_config)

    config = agent._build_codex_config(  # pyright: ignore[reportPrivateUsage]
        {"approval_policy": "on-request", "otel": {"log_user_prompt": False}}
    )

    assert config["sandbox_mode"] == "danger-full-access"
    assert config["approval_policy"] == "on-request"
    assert config["model"] == "gpt-5.5"
    assert config["openai_base_url"] == "https://example.test/v1"
    assert config["otel"] == {
        "environment": "dev",
        "exporter": "none",
        "log_user_prompt": False,
    }


def test_codex_home_configuration_is_copied_into_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(CodexAgent)
    setattr(agent, "_model_name", "gpt-5.5")
    setattr(agent, "_model_api_key", "test-key")
    setattr(agent, "_model_base_url", "https://example.test/v1")
    setattr(agent, "_model_organization_id", "org_123")
    container = _FakeContainer(
        {"/metadata/.codex/config.toml": 'approval_policy = "on-request"\n'}
    )

    def container_config(self: CodexAgent) -> dict[str, Any]:
        return {"sandbox_mode": "danger-full-access", "approval_policy": "never"}

    monkeypatch.setattr(CodexAgent, "_load_container_codex_config", container_config)

    agent._configure_codex_home(  # pyright: ignore[reportPrivateUsage]
        container, "/metadata/.codex"
    )

    assert container.commands == [
        ["sh", "-c", "cat /metadata/.codex/config.toml 2>/dev/null || true"],
        ["mkdir", "-p", "/metadata/.codex"],
    ]
    assert container.files["auth.json"] == json.dumps(
        {
            "auth_mode": "apikey",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_ORGANIZATION": "org_123",
            "OPENAI_ORG_ID": "org_123",
        }
    )
    config = tomllib.loads(container.files["config.toml"])
    assert config["sandbox_mode"] == "danger-full-access"
    assert config["approval_policy"] == "on-request"
    assert config["model"] == "gpt-5.5"
    assert config["openai_base_url"] == "https://example.test/v1"


def test_invalid_runtime_codex_config_raises_toml_decode_error() -> None:
    agent = object.__new__(CodexAgent)
    container = _FakeContainer({"/metadata/.codex/config.toml": "not = [valid"})

    with pytest.raises(tomllib.TOMLDecodeError):
        agent._read_container_toml(  # pyright: ignore[reportPrivateUsage]
            container, "/metadata/.codex/config.toml"
        )


def test_codex_agent_resumes_existing_thread_without_source_filter() -> None:
    agent = object.__new__(CodexAgent)
    setattr(agent, "_id", "codex-test")
    setattr(agent, "_model_name", "gpt-5.5")
    setattr(agent, "_container_workspace_directory", "/workspace")
    codex = _FakeCodexAppServer([SimpleNamespace(id="thread-1")])

    thread = asyncio.run(
        agent._resume_or_start_thread(  # pyright: ignore[reportPrivateUsage]
            codex, "challenge-1"
        )
    )

    assert thread.id == "resumed-thread-1"
    assert codex.thread_list_kwargs == {
        "archived": False,
        "cwd": "/workspace",
    }
    assert codex.resumed_thread_ids == ["thread-1"]
    assert codex.started_threads == []


def test_codex_agent_starts_thread_when_no_existing_thread() -> None:
    agent = object.__new__(CodexAgent)
    setattr(agent, "_id", "codex-test")
    setattr(agent, "_model_name", "gpt-5.5")
    setattr(agent, "_container_workspace_directory", "/workspace")
    codex = _FakeCodexAppServer([])

    thread = asyncio.run(
        agent._resume_or_start_thread(  # pyright: ignore[reportPrivateUsage]
            codex, "challenge-1"
        )
    )

    assert thread.id == "started-thread"
    assert codex.resumed_thread_ids == []
    assert codex.started_threads == [
        {
            "model": "gpt-5.5",
            "cwd": "/workspace",
            "service_name": "catchy",
            "config": {},
        }
    ]


def test_codex_notification_yields_agent_and_reasoning_deltas() -> None:
    agent = object.__new__(CodexAgent)
    setattr(agent, "_id", "codex-test")

    agent_delta = agent._events_from_codex_notification(  # pyright: ignore[reportPrivateUsage]
        "item/agentMessage/delta",
        AgentMessageDeltaNotification.model_validate(
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "itemId": "item-1",
                "delta": "hello",
            }
        ),
        turn_id="turn-1",
        challenge_id="challenge-1",
    )
    reasoning_delta = agent._events_from_codex_notification(  # pyright: ignore[reportPrivateUsage]
        "item/reasoning/textDelta",
        ReasoningTextDeltaNotification.model_validate(
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "itemId": "item-2",
                "contentIndex": 0,
                "delta": "thinking",
            }
        ),
        turn_id="turn-1",
        challenge_id="challenge-1",
    )

    assert agent_delta == [Chunk(tag="action", text="hello")]
    assert reasoning_delta == [Chunk(tag="thinking", text="thinking")]


def test_codex_notification_yields_tool_start_and_completion() -> None:
    agent = object.__new__(CodexAgent)
    setattr(agent, "_id", "codex-test")

    started = agent._events_from_codex_notification(  # pyright: ignore[reportPrivateUsage]
        "item/started",
        ItemStartedNotification.model_validate(
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {
                    "type": "commandExecution",
                    "id": "item-1",
                    "command": "pytest -q",
                    "cwd": "/workspace",
                    "status": "inProgress",
                    "commandActions": [],
                },
            }
        ),
        turn_id="turn-1",
        challenge_id="challenge-1",
    )
    output = agent._events_from_codex_notification(  # pyright: ignore[reportPrivateUsage]
        "item/commandExecution/outputDelta",
        CommandExecutionOutputDeltaNotification.model_validate(
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "itemId": "item-1",
                "delta": "1 passed",
            }
        ),
        turn_id="turn-1",
        challenge_id="challenge-1",
    )
    completed = agent._events_from_codex_notification(  # pyright: ignore[reportPrivateUsage]
        "item/completed",
        ItemCompletedNotification.model_validate(
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {
                    "type": "commandExecution",
                    "id": "item-1",
                    "command": "pytest -q",
                    "cwd": "/workspace",
                    "status": "completed",
                    "commandActions": [],
                    "exitCode": 0,
                    "aggregatedOutput": "1 passed",
                },
            }
        ),
        turn_id="turn-1",
        challenge_id="challenge-1",
    )

    assert started == [
        Chunk(
            tag="tool_use",
            text=json.dumps(
                {
                    "type": "commandExecution",
                    "id": "item-1",
                    "status": "inProgress",
                    "command": "pytest -q",
                    "cwd": "/workspace",
                    "source": "agent",
                }
            ),
        )
    ]
    assert output == [Chunk(tag="observation", text="1 passed")]
    assert completed == [ItemCompleted()]


def test_codex_notification_yields_turn_completed() -> None:
    agent = object.__new__(CodexAgent)
    setattr(agent, "_id", "codex-test")

    events = agent._events_from_codex_notification(  # pyright: ignore[reportPrivateUsage]
        "turn/completed",
        TurnCompletedNotification.model_validate(
            {
                "threadId": "thread-1",
                "turn": {
                    "id": "turn-1",
                    "status": "completed",
                    "items": [],
                    "error": None,
                },
            }
        ),
        turn_id="turn-1",
        challenge_id="challenge-1",
    )

    assert events == [TurnCompleted()]


def test_codex_notification_yields_structured_log_events() -> None:
    agent = object.__new__(CodexAgent)
    setattr(agent, "_id", "codex-test")

    diff_events = agent._events_from_codex_notification(  # pyright: ignore[reportPrivateUsage]
        "turn/diff/updated",
        TurnDiffUpdatedNotification.model_validate(
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "diff": "--- a/file\n+++ b/file\n",
            }
        ),
        turn_id="turn-1",
        challenge_id="challenge-1",
    )
    usage_events = agent._events_from_codex_notification(  # pyright: ignore[reportPrivateUsage]
        "thread/tokenUsage/updated",
        ThreadTokenUsageUpdatedNotification.model_validate(
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "tokenUsage": {
                    "last": {
                        "inputTokens": 1,
                        "cachedInputTokens": 0,
                        "outputTokens": 2,
                        "reasoningOutputTokens": 0,
                        "totalTokens": 3,
                    },
                    "total": {
                        "inputTokens": 5,
                        "cachedInputTokens": 1,
                        "outputTokens": 3,
                        "reasoningOutputTokens": 0,
                        "totalTokens": 8,
                    },
                    "modelContextWindow": 1000,
                },
            }
        ),
        turn_id="turn-1",
        challenge_id="challenge-1",
    )

    assert diff_events == [
        Log(
            kind="diff",
            text="--- a/file\n+++ b/file\n",
            raw={
                "threadId": "thread-1",
                "turnId": "turn-1",
                "diff": "--- a/file\n+++ b/file\n",
            },
        )
    ]
    assert len(usage_events) == 1
    usage = usage_events[0]
    assert isinstance(usage, TokenUsage)
    assert usage.provider == "openai"
    assert usage.source == "thread_token_usage_updated"
    assert usage.input_tokens == 5
    assert usage.cached_input_tokens == 1
    assert usage.output_tokens == 3
    assert usage.total_tokens == 8
    token_usage = cast(dict[str, Any], usage.raw["tokenUsage"])
    total = cast(dict[str, Any], token_usage["total"])
    assert total["inputTokens"] == 5


def test_codex_notification_raises_non_retryable_error() -> None:
    agent = object.__new__(CodexAgent)
    setattr(agent, "_id", "codex-test")

    with pytest.raises(RuntimeError, match="non-retryable turn error"):
        agent._events_from_codex_notification(  # pyright: ignore[reportPrivateUsage]
            "error",
            ErrorNotification.model_validate(
                {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "willRetry": False,
                    "error": {"message": "boom"},
                }
            ),
            turn_id="turn-1",
            challenge_id="challenge-1",
        )


class _ExecResult:
    def __init__(self, exit_code: int, output: bytes = b"") -> None:
        self.exit_code = exit_code
        self.output = output


class _FakeContainer:
    def __init__(self, readable_files: dict[str, str]) -> None:
        self._readable_files = readable_files
        self.commands: list[list[str]] = []
        self.files: dict[str, str] = {}

    def exec_run(self, command: list[str]) -> _ExecResult:
        self.commands.append(command)
        if command[:2] == ["sh", "-c"]:
            path = command[2].removeprefix("cat ").split(" ", 1)[0]
            return _ExecResult(0, self._readable_files.get(path, "").encode())
        return _ExecResult(0)

    def put_archive(self, directory: str, data: bytes) -> bool:
        assert directory == "/metadata/.codex"
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
            for member in archive.getmembers():
                file = archive.extractfile(member)
                assert file is not None
                self.files[member.name] = file.read().decode()
        return True


class _FakeCodexAppServer:
    def __init__(self, threads: list[Any]) -> None:
        self._threads = threads
        self.thread_list_kwargs: dict[str, Any] = {}
        self.resumed_thread_ids: list[str] = []
        self.started_threads: list[dict[str, Any]] = []

    async def thread_list(self, **kwargs: Any) -> Any:
        self.thread_list_kwargs = {
            key: getattr(value, "root", value) if key == "cwd" else value
            for key, value in kwargs.items()
        }
        return SimpleNamespace(data=self._threads)

    async def thread_resume(self, thread_id: str) -> Any:
        self.resumed_thread_ids.append(thread_id)
        return SimpleNamespace(id=f"resumed-{thread_id}")

    async def thread_start(self, **kwargs: Any) -> Any:
        self.started_threads.append(kwargs)
        return SimpleNamespace(id="started-thread")


class _DockerSocketProxy:
    def __init__(self, unix_socket_path: str) -> None:
        self._unix_socket_path = unix_socket_path
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.getsockname()
        return f"tcp://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stopped.set()
        with contextlib.suppress(OSError):
            with socket.create_connection(self._server.getsockname(), timeout=0.1):
                pass
        self._server.close()
        self._thread.join(timeout=1)

    def _serve(self) -> None:
        while not self._stopped.is_set():
            try:
                client, _address = self._server.accept()
            except OSError:
                break

            thread = threading.Thread(target=self._handle, args=(client,), daemon=True)
            thread.start()

    def _handle(self, client: socket.socket) -> None:
        with client:
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            with upstream:
                upstream.connect(self._unix_socket_path)
                sockets = [client, upstream]

                while True:
                    readable, _writable, _errors = select.select(sockets, [], [], 5)
                    if not readable:
                        return

                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = upstream if source is client else client
                        destination.sendall(data)


@pytest.fixture
def docker_base_url(pytestconfig: pytest.Config) -> Iterator[str]:
    record_mode = str(pytestconfig.getoption("--record-mode"))
    if record_mode == "none":
        yield "tcp://127.0.0.1:1"
        return

    proxy = _DockerSocketProxy(_DOCKER_SOCKET)
    proxy.start()
    try:
        yield proxy.base_url
    finally:
        proxy.close()


@pytest.fixture
def run_directories(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    root = tmp_path / "lets-change"
    workspace = root / "workspace"
    metadata = root / "metadata"
    workspace.mkdir(parents=True)
    metadata.mkdir(parents=True)

    try:
        yield workspace, metadata
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _lets_change_challenge() -> Challenge:
    return Challenge(
        id="lets-change",
        description="nc sol.plus.or.kr 25001",
        directory=_CHALLENGE_ROOT / "source",
    )


def _redact_stream_message(message: str) -> str:
    redacted = message
    for name, value in os.environ.items():
        if not value or len(value) < 8:
            continue
        if any(marker in name.upper() for marker in ("API_KEY", "AUTH", "SECRET")):
            redacted = redacted.replace(value, "<REDACTED>")
    return redacted


def _record_stream_output(messages: list[str]) -> None:
    _STREAM_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "challenge_id": "lets-change",
        "expected_marker": _STREAM_OK_MARKER,
        "messages": [_redact_stream_message(message) for message in messages],
    }
    _STREAM_OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")


@pytest.mark.default_cassette("lets_change_docker_container.yaml")
@pytest.mark.vcr(match_on=["method", "path", "query"])
def test_codex_agent_runs_lets_change_in_recorded_docker_container(
    docker_base_url: str,
    monkeypatch: pytest.MonkeyPatch,
    run_directories: tuple[Path, Path],
) -> None:
    monkeypatch.setenv("CATCHY_TEST_OPENAI_API_KEY", "test-openai-api-key")
    configured_codex_homes: list[str] = []

    def configure_codex_home(self: CodexAgent, container: Any, codex_home: str) -> None:
        configured_codex_homes.append(codex_home)

    monkeypatch.setattr(CodexAgent, "_configure_codex_home", configure_codex_home)
    workspace, metadata_directory = run_directories
    docker_client = DockerClient(base_url=docker_base_url, timeout=30)

    try:
        agent = CodexAgent(
            id="codex-test",
            model_name="gpt-test",
            model_api_key=os.environ["CATCHY_TEST_OPENAI_API_KEY"],
            container_challenge_directory="/challenge",
            container_workspace_directory="/workspace",
            container_metadata_directory="/metadata",
            docker_image=docker_client.images.get(_CODEX_IMAGE),
            docker_client=docker_client,
            user_prompt_template="Solve {{ challenge.id }}",
        )
        challenge = _lets_change_challenge()

        with agent._docker_container(  # pyright: ignore[reportPrivateUsage]
            challenge=challenge,
            workspace=workspace,
            metadata_directory=metadata_directory,
        ) as container:
            mounts = cast(list[dict[str, Any]], container.attrs["Mounts"])
            destinations = {str(mount["Destination"]) for mount in mounts}

        assert "/challenge" in destinations
        assert "/workspace" in destinations
        assert "/metadata" in destinations
        assert (_CHALLENGE_ROOT / "source" / "challenge.c").is_file()
        assert configured_codex_homes == ["/metadata/.codex"]
    finally:
        docker_client.close()


def test_codex_agent_stream_reaches_openai_when_enabled(
    pytestconfig: pytest.Config,
    run_directories: tuple[Path, Path],
) -> None:
    record_mode = str(pytestconfig.getoption("--record-mode"))
    if record_mode == "none":
        pytest.skip("pass --record-mode=once to refresh stream output")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for the live OpenAI stream test")

    async def run_stream() -> list[str]:
        workspace, metadata_directory = run_directories
        docker_client = DockerClient(base_url=f"unix://{_DOCKER_SOCKET}", timeout=30)
        try:
            try:
                docker_image = docker_client.images.get(_CODEX_IMAGE)
            except DockerException as exc:
                pytest.skip(f"Docker image is not available locally: {exc}")

            agent = CodexAgent(
                id="codex-live-openai-test",
                model_name=os.environ.get("CATCHY_TEST_OPENAI_MODEL", "gpt-5.5"),
                model_api_key=api_key,
                container_challenge_directory="/challenge",
                container_workspace_directory="/workspace",
                container_metadata_directory="/metadata",
                docker_image=docker_image,
                docker_client=docker_client,
                user_prompt_template=(
                    f"Reply with exactly {_STREAM_OK_MARKER}. "
                    "Do not run commands or edit files."
                ),
            )

            messages: list[str] = []
            stream = agent.stream(
                challenge=_lets_change_challenge(),
                workspace=workspace,
                metadata_directory=metadata_directory,
            )
            async for message in stream:
                if isinstance(message, Chunk):
                    messages.append(message.text)
            return messages
        finally:
            docker_client.close()

    messages = asyncio.run(asyncio.wait_for(run_stream(), timeout=180))

    assert any(_STREAM_OK_MARKER in message for message in messages)
    _record_stream_output(messages)


def test_recorded_stream_output_has_expected_shape() -> None:
    if not _STREAM_OUTPUT_PATH.exists():
        pytest.skip("stream output fixture has not been recorded yet")

    raw_payload = json.loads(_STREAM_OUTPUT_PATH.read_text())
    assert isinstance(raw_payload, dict)
    payload = cast(dict[str, Any], raw_payload)
    assert payload["challenge_id"] == "lets-change"
    assert payload["expected_marker"] == _STREAM_OK_MARKER
    raw_messages = payload["messages"]
    assert isinstance(raw_messages, list)
    messages = cast(list[Any], raw_messages)
    assert any(
        isinstance(message, str) and _STREAM_OK_MARKER in message
        for message in messages
    )
