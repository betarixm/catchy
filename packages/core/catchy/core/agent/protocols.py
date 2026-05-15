from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator

from ..challenge.models import Challenge
from ..webhook.models import Webhook
from .models import Event, Interrupt


class Agent[T](ABC):
    tag: str

    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def stream(
        self,
        challenge: Challenge,
        workspace_directory: Path,
        metadata_directory: Path,
        webhook: Webhook | None = None,
        additional_prompt: str | None = None,
    ) -> AsyncGenerator[Event[T], Interrupt]: ...

    @abstractmethod
    def last_agent_message_from_events(self, events: list[Event[T]]) -> str: ...

    @staticmethod
    def from_configuration(configuration: dict[str, Any]) -> Agent[Any]:
        match configuration["class"]:
            case "catchy.codex.CodexAgent":
                from catchy.codex import CodexAgent

                return CodexAgent.from_configuration(configuration)
            case "catchy.claude_code.ClaudeCodeAgent":
                from catchy.claude_code import ClaudeCodeAgent

                return ClaudeCodeAgent.from_configuration(configuration)
            case name:
                module_name, class_name = name.rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name])
                klass = getattr(module, class_name)
                if issubclass(klass, Agent):
                    return klass.from_configuration(configuration)
                else:
                    raise ValueError(f"Class {name} is not a subclass of Agent")


class HandoffExporter[T](ABC):
    def export(self, event: Event[T]) -> str: ...
