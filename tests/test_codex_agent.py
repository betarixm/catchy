from __future__ import annotations

import pytest
from catchy.codex import CodexEvent, CodexEventRenderer
from catchy.core.agent.models import (
    Component,
    Delta,
    ItemCompleted,
    JsonLog,
    Nop,
    TextLog,
    ThreadStarted,
    TokenUsage,
    TurnCompleted,
    TurnStarted,
)
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
)
from codex_app_server.models import Notification


def _render(method: str, payload_dict: dict[str, object]) -> list[Component]:
    """Validate the payload into a Notification and render via CodexEventRenderer."""
    from codex_app_server.generated.notification_registry import NOTIFICATION_MODELS

    Model = NOTIFICATION_MODELS[method]
    payload = Model.model_validate(payload_dict)  # type: ignore[attr-defined]
    notification = Notification(method=method, payload=payload)  # type: ignore[arg-type]
    event = CodexEvent(raw=notification)
    return list(CodexEventRenderer(model_name="gpt-5.5").render(event))


def test_agent_message_delta_is_agent_tag() -> None:
    components = _render(
        "item/agentMessage/delta",
        {"threadId": "t", "turnId": "u", "itemId": "i", "delta": "hello"},
    )
    assert components == [Delta(tag="agent", text="hello")]


def test_command_execution_output_delta_is_observation() -> None:
    components = _render(
        "item/commandExecution/outputDelta",
        {"threadId": "t", "turnId": "u", "itemId": "i", "delta": "1 passed"},
    )
    assert components == [Delta(tag="observation", text="1 passed")]


def test_file_change_output_delta_is_observation() -> None:
    components = _render(
        "item/fileChange/outputDelta",
        {"threadId": "t", "turnId": "u", "itemId": "i", "delta": "+new line\n"},
    )
    assert components == [Delta(tag="observation", text="+new line\n")]


def test_reasoning_text_delta_is_thinking() -> None:
    components = _render(
        "item/reasoning/textDelta",
        {
            "threadId": "t",
            "turnId": "u",
            "itemId": "i",
            "contentIndex": 0,
            "delta": "thinking out loud",
        },
    )
    assert components == [Delta(tag="thinking", text="thinking out loud")]


def test_reasoning_summary_text_delta_is_thinking() -> None:
    components = _render(
        "item/reasoning/summaryTextDelta",
        {
            "threadId": "t",
            "turnId": "u",
            "itemId": "i",
            "summaryIndex": 0,
            "delta": "summary",
        },
    )
    assert components == [Delta(tag="thinking", text="summary")]


def test_plan_delta_is_thinking() -> None:
    components = _render(
        "item/plan/delta",
        {"threadId": "t", "turnId": "u", "itemId": "i", "delta": "plan step"},
    )
    assert components == [Delta(tag="thinking", text="plan step")]


def test_terminal_interaction_is_tool_input() -> None:
    components = _render(
        "item/commandExecution/terminalInteraction",
        {
            "threadId": "t",
            "turnId": "u",
            "itemId": "i",
            "processId": "p1",
            "stdin": "ls -la\n",
        },
    )
    assert components == [Delta(tag="tool_input", text="ls -la\n")]


def test_item_started_is_json_log() -> None:
    components = _render(
        "item/started",
        {
            "threadId": "t",
            "turnId": "u",
            "item": {
                "type": "commandExecution",
                "id": "i",
                "command": "pytest -q",
                "cwd": "/workspace",
                "status": "inProgress",
                "commandActions": [],
            },
        },
    )
    assert len(components) == 1
    component = components[0]
    assert isinstance(component, JsonLog)
    assert component.tag == "item_started"


def test_item_completed_yields_item_completed() -> None:
    components = _render(
        "item/completed",
        {
            "threadId": "t",
            "turnId": "u",
            "item": {
                "type": "commandExecution",
                "id": "i",
                "command": "pytest -q",
                "cwd": "/workspace",
                "status": "completed",
                "commandActions": [],
                "exitCode": 0,
                "aggregatedOutput": "1 passed",
            },
        },
    )
    assert components == [ItemCompleted()]


def test_turn_completed_yields_turn_completed() -> None:
    components = _render(
        "turn/completed",
        {
            "threadId": "t",
            "turn": {
                "id": "u",
                "status": "completed",
                "items": [],
                "error": None,
            },
        },
    )
    assert components == [TurnCompleted()]


def test_turn_started_yields_turn_started() -> None:
    components = _render(
        "turn/started",
        {
            "threadId": "t",
            "turn": {"id": "u", "status": "inProgress", "items": []},
        },
    )
    assert components == [TurnStarted()]


def test_token_usage_yields_token_usage_component() -> None:
    components = _render(
        "thread/tokenUsage/updated",
        {
            "threadId": "t",
            "turnId": "u",
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
        },
    )
    assert len(components) == 1
    usage = components[0]
    assert isinstance(usage, TokenUsage)
    assert usage.input_tokens == 1
    assert usage.output_tokens == 2
    assert usage.total_tokens == 3


def test_turn_diff_updated_yields_text_log() -> None:
    components = _render(
        "turn/diff/updated",
        {"threadId": "t", "turnId": "u", "diff": "--- a\n+++ b\n"},
    )
    assert components == [TextLog(tag="turn_diff_updated", text="--- a\n+++ b\n")]


def test_mcp_tool_call_progress_yields_text_log() -> None:
    components = _render(
        "item/mcpToolCall/progress",
        {
            "threadId": "t",
            "turnId": "u",
            "itemId": "i",
            "message": "downloading…",
        },
    )
    assert components == [TextLog(tag="mcp_tool_call_progress", text="downloading…")]


def test_error_retryable_yields_nop() -> None:
    components = _render(
        "error",
        {
            "threadId": "t",
            "turnId": "u",
            "willRetry": True,
            "error": {"message": "transient"},
        },
    )
    assert components == [Nop()]


def test_error_non_retryable_raises() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _render(
            "error",
            {
                "threadId": "t",
                "turnId": "u",
                "willRetry": False,
                "error": {"message": "boom"},
            },
        )


def test_context_compacted_yields_text_log() -> None:
    components = _render(
        "thread/compacted",
        {"threadId": "t", "turnId": "u"},
    )
    assert components == [TextLog(tag="context_compacted", text="")]
