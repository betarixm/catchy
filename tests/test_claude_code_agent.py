from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

from catchy.claude_code import ClaudeCodeAgent
from catchy.core.agent.models import Chunk, ItemCompleted, Nop, TokenUsage, TurnCompleted
from claude_agent_sdk import ResultMessage, StreamEvent


def test_claude_home_configuration_is_copied_into_container() -> None:
    agent = object.__new__(ClaudeCodeAgent)
    setattr(agent, "_container_metadata_directory", "/metadata")
    setattr(agent, "_docker_image", object())
    setattr(agent, "_docker_client", _FakeDockerClient({"permissions": {"allow": []}}))
    container = _FakeContainer(
        {"/metadata/.claude/settings.json": '{"permissions": {"deny": []}}'}
    )

    agent._configure_claude_configuration_directory(  # pyright: ignore[reportPrivateUsage]
        container, "/metadata/.claude"
    )

    assert container.commands == [
        ["sh", "-c", "cat /metadata/.claude/settings.json 2>/dev/null || true"],
        ["mkdir", "-p", "/metadata/.claude"],
    ]
    settings = json.loads(container.files["settings.json"])
    assert settings == {"permissions": {"allow": [], "deny": []}}


def test_prepare_claude_runtime_creates_claude_config_directory() -> None:
    agent = object.__new__(ClaudeCodeAgent)
    setattr(agent, "_container_workspace_directory", "/workspace")
    container = _FakeContainer({})

    agent._prepare_claude_runtime(  # pyright: ignore[reportPrivateUsage]
        container, "/metadata/.claude"  # pyright: ignore[reportArgumentType]
    )

    script = container.commands[-1][2]
    assert 'config_dir=/metadata/.claude' in script
    assert 'mkdir -p "$config_dir"\nchown -R "$uid:$gid" "$config_dir"' in script


def test_claude_cli_wrapper_execs_claude_inside_container(tmp_path: Path) -> None:
    agent = object.__new__(ClaudeCodeAgent)
    setattr(agent, "_model_api_key", "test-key")
    setattr(agent, "_model_base_url", "https://example.test")
    setattr(agent, "_container_workspace_directory", "/workspace")

    wrapper = agent._write_cli_wrapper(  # pyright: ignore[reportPrivateUsage]
        metadata_directory=tmp_path,
        container_id="container-123",
        container_claude_configuration_directory="/metadata/.claude",
        container_user="1001:1001",
    )

    script = wrapper.read_text()
    assert script.startswith("#!/bin/sh\nset -eu\nexec docker exec -i")
    assert "--user 1001:1001" in script
    assert "--workdir /workspace" in script
    assert "--env ANTHROPIC_API_KEY" in script
    assert "--env ANTHROPIC_BASE_URL" in script
    assert 'container-123 claude "$@"' in script


def test_claude_stream_event_yields_text_delta() -> None:
    agent = object.__new__(ClaudeCodeAgent)
    event = agent._event_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-1",
            session_id="session-1",
            event={
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "hello"},
            },
        )
    )

    assert event == Chunk(tag="action", text="hello")


def test_claude_stream_event_yields_tool_start_and_input_delta() -> None:
    agent = object.__new__(ClaudeCodeAgent)

    start = agent._event_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-1",
            session_id="session-1",
            event={
                "type": "content_block_start",
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "Read",
                    "input": {},
                },
            },
        )
    )
    input_delta = agent._event_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-2",
            session_id="session-1",
            event={
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"file_path": "README.md"}',
                },
            },
        )
    )
    stop = agent._event_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-3",
            session_id="session-1",
            event={"type": "content_block_stop"},
        )
    )

    assert start == Chunk(
        tag="tool_use",
        text='{"id": "toolu_123", "input": {}, "name": "Read"}',
    )
    assert input_delta == Chunk(
        tag="tool_input",
        text='{"file_path": "README.md"}',
    )
    assert isinstance(stop, ItemCompleted)


def test_claude_stream_event_accumulates_tool_input_on_stop() -> None:
    agent = object.__new__(ClaudeCodeAgent)

    start_events = agent._events_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-1",
            session_id="session-1",
            event={
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "Read",
                    "input": {},
                },
            },
        )
    )
    first_delta = agent._events_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-2",
            session_id="session-1",
            event={
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"file_'},
            },
        )
    )
    second_delta = agent._events_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-3",
            session_id="session-1",
            event={
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": 'path": "README.md"}',
                },
            },
        )
    )
    stop_events = agent._events_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-4",
            session_id="session-1",
            event={"type": "content_block_stop", "index": 1},
        )
    )

    assert start_events == [
        Chunk(tag="tool_use", text='{"id": "toolu_123", "input": {}, "name": "Read"}')
    ]
    assert first_delta == [Chunk(tag="tool_input", text='{"file_')]
    assert second_delta == [Chunk(tag="tool_input", text='path": "README.md"}')]
    assert stop_events == [
        Chunk(tag="tool_input", text='{"file_path": "README.md"}'),
        ItemCompleted(),
    ]


def test_claude_stream_event_handles_ping_and_signature_delta() -> None:
    agent = object.__new__(ClaudeCodeAgent)

    ping = agent._event_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-1",
            session_id="session-1",
            event={"type": "ping"},
        )
    )
    start = agent._events_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-2",
            session_id="session-1",
            event={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        )
    )
    signature = agent._event_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-3",
            session_id="session-1",
            event={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig_123"},
            },
        )
    )

    assert isinstance(ping, Nop)
    assert start == []
    assert isinstance(signature, Nop)


def test_claude_stream_event_yields_token_usage_events() -> None:
    agent = object.__new__(ClaudeCodeAgent)

    start_events = agent._events_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-1",
            session_id="session-1",
            event={
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 2,
                        "output_tokens": 1,
                    }
                },
            },
        )
    )
    delta_events = agent._events_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-2",
            session_id="session-1",
            event={
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 2,
                    "output_tokens": 5,
                },
            },
        )
    )

    assert start_events == [
        TokenUsage(
            provider="anthropic",
            source="stream_event",
            input_tokens=10,
            cache_read_input_tokens=2,
            output_tokens=1,
            raw={
                "event_type": "message_start",
                "session_id": "session-1",
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 2,
                    "output_tokens": 1,
                },
            },
        )
    ]
    assert delta_events == [
        TokenUsage(
            provider="anthropic",
            source="stream_event",
            input_tokens=10,
            cache_read_input_tokens=2,
            output_tokens=5,
            raw={
                "event_type": "message_delta",
                "session_id": "session-1",
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 2,
                    "output_tokens": 5,
                },
                "stop_reason": "end_turn",
            },
        )
    ]


def test_claude_result_message_yields_usage_event_before_turn_completed() -> None:
    agent = object.__new__(ClaudeCodeAgent)

    events = agent._events_from_message(  # pyright: ignore[reportPrivateUsage]
        ResultMessage(
            subtype="success",
            duration_ms=1200,
            duration_api_ms=900,
            is_error=False,
            num_turns=1,
            session_id="session-1",
            stop_reason="end_turn",
            usage={
                "input_tokens": 10,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2,
                "output_tokens": 5,
            },
        )
    )

    assert events == [
        TokenUsage(
            provider="anthropic",
            source="result_message",
            input_tokens=10,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=2,
            output_tokens=5,
            raw={
                "subtype": "success",
                "session_id": "session-1",
                "duration_ms": 1200,
                "duration_api_ms": 900,
                "num_turns": 1,
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 2,
                    "output_tokens": 5,
                },
                "stop_reason": "end_turn",
            },
        ),
        TurnCompleted(),
    ]


def test_claude_stream_event_yields_server_tool_use() -> None:
    agent = object.__new__(ClaudeCodeAgent)

    event = agent._event_from_stream_event(  # pyright: ignore[reportPrivateUsage]
        StreamEvent(
            uuid="event-1",
            session_id="session-1",
            event={
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_123",
                    "name": "web_search",
                    "input": {},
                },
            },
        )
    )

    assert event == Chunk(
        tag="tool_use",
        text='{"id": "srvtoolu_123", "input": {}, "name": "web_search"}',
    )


def test_claude_stream_event_raises_on_error_event() -> None:
    agent = object.__new__(ClaudeCodeAgent)

    try:
        agent._event_from_stream_event(  # pyright: ignore[reportPrivateUsage]
            StreamEvent(
                uuid="event-1",
                session_id="session-1",
                event={
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Overloaded",
                    },
                },
            )
        )
    except RuntimeError as error:
        assert str(error) == "Claude Code stream error: overloaded_error: Overloaded"
    else:
        raise AssertionError("expected stream error to raise")


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
        assert directory == "/metadata/.claude"
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
            for member in archive.getmembers():
                file = archive.extractfile(member)
                assert file is not None
                self.files[member.name] = file.read().decode()
        return True


class _FakeDockerClient:
    def __init__(self, image_settings: dict[str, Any]) -> None:
        self.containers = _FakeContainers(image_settings)


class _FakeContainers:
    def __init__(self, image_settings: dict[str, Any]) -> None:
        self._image_settings = image_settings

    def run(self, *_args: object, **_kwargs: object) -> bytes:
        return json.dumps(self._image_settings).encode()
