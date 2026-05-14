from __future__ import annotations

import pytest
from catchy.claude_code import ClaudeCodeEvent, ClaudeCodeEventRenderer
from catchy.core.agent.models import (
    Component,
    Delta,
    ItemCompleted,
    JsonLog,
    Nop,
    TokenUsage,
    TurnCompleted,
)
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    ServerToolResultBlock,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


def _render(message: object) -> list[Component]:
    event = ClaudeCodeEvent(raw=message)
    return list(ClaudeCodeEventRenderer().render(event))


def test_stream_text_delta_is_agent_tag() -> None:
    components = _render(
        StreamEvent(
            uuid="e",
            session_id="s",
            event={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            parent_tool_use_id=None,
        )
    )
    assert components == [Delta(tag="agent", text="hello")]


def test_stream_thinking_delta_is_thinking_tag() -> None:
    components = _render(
        StreamEvent(
            uuid="e",
            session_id="s",
            event={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "musing"},
            },
            parent_tool_use_id=None,
        )
    )
    assert components == [Delta(tag="thinking", text="musing")]


def test_stream_input_json_delta_is_tool_input_tag() -> None:
    components = _render(
        StreamEvent(
            uuid="e",
            session_id="s",
            event={
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"file_path":',
                },
            },
            parent_tool_use_id=None,
        )
    )
    assert components == [Delta(tag="tool_input", text='{"file_path":')]


def test_stream_content_block_stop_yields_item_completed() -> None:
    components = _render(
        StreamEvent(
            uuid="e",
            session_id="s",
            event={"type": "content_block_stop", "index": 0},
            parent_tool_use_id=None,
        )
    )
    assert components == [ItemCompleted()]


def test_stream_message_start_yields_token_usage() -> None:
    components = _render(
        StreamEvent(
            uuid="e",
            session_id="s",
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
            parent_tool_use_id=None,
        )
    )
    assert len(components) == 1
    usage = components[0]
    assert isinstance(usage, TokenUsage)
    assert usage.input_tokens == 10
    assert usage.cached_input_tokens == 2
    assert usage.output_tokens == 1


def test_stream_error_event_raises() -> None:
    with pytest.raises(RuntimeError, match="overloaded_error"):
        _render(
            StreamEvent(
                uuid="e",
                session_id="s",
                event={
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                },
                parent_tool_use_id=None,
            )
        )


def test_result_message_yields_agent_delta_and_turn_completed() -> None:
    components = _render(
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            result="done",
            total_cost_usd=None,
            usage=None,
        )
    )
    assert components == [Delta(tag="agent", text="done"), TurnCompleted()]


def test_assistant_text_block_is_agent_tag() -> None:
    components = _render(
        AssistantMessage(
            content=[TextBlock(text="reply")],
            model="claude",
            parent_tool_use_id=None,
        )
    )
    assert components == [Delta(tag="agent", text="reply")]


def test_assistant_thinking_block_is_thinking_tag() -> None:
    components = _render(
        AssistantMessage(
            content=[ThinkingBlock(thinking="hmm", signature="sig")],
            model="claude",
            parent_tool_use_id=None,
        )
    )
    assert components == [Delta(tag="thinking", text="hmm")]


def test_assistant_tool_use_block_is_json_log() -> None:
    components = _render(
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="Read", input={"path": "a"})],
            model="claude",
            parent_tool_use_id=None,
        )
    )
    assert len(components) == 1
    log = components[0]
    assert isinstance(log, JsonLog)
    assert log.tag == "Read"
    assert log.data == {"path": "a"}


def test_user_tool_result_block_is_observation_tag() -> None:
    components = _render(
        UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="t1",
                    content="file contents",
                    is_error=False,
                )
            ],
            parent_tool_use_id=None,
        )
    )
    assert components == [Delta(tag="observation", text="file contents")]


def test_user_server_tool_result_block_is_observation_tag() -> None:
    components = _render(
        UserMessage(
            content=[
                ServerToolResultBlock(
                    tool_use_id="t1",
                    content={"results": []},
                )
            ],
            parent_tool_use_id=None,
        )
    )
    assert len(components) == 1
    delta = components[0]
    assert isinstance(delta, Delta)
    assert delta.tag == "observation"


def test_user_message_with_string_content_is_user_tag() -> None:
    components = _render(
        UserMessage(content="hello", parent_tool_use_id=None)
    )
    assert components == [Delta(tag="user", text="hello")]


def test_assistant_message_with_string_content_is_agent_tag() -> None:
    components = _render(
        AssistantMessage(
            content="quick reply",
            model="claude",
            parent_tool_use_id=None,
        )
    )
    assert components == [Delta(tag="agent", text="quick reply")]
