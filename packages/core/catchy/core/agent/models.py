from abc import ABC, abstractmethod
from typing import Iterator, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
)


class Event[T](ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    format: str
    raw: T

    @abstractmethod
    @field_validator("raw", mode="before")
    @classmethod
    def _deserialize_raw(cls, value: object) -> T: ...

    @abstractmethod
    @field_serializer("raw")
    def _serialize_raw(self, value: T) -> str: ...


class Nop(BaseModel): ...


DeltaTag = Literal["agent", "thinking", "observation", "tool_input", "user"]


class Delta(BaseModel):
    tag: DeltaTag
    text: str


class TextLog(BaseModel):
    tag: str
    text: str


class JsonLog(BaseModel):
    tag: str
    data: dict[str, object]


class TokenUsage(BaseModel):
    cached_input_tokens: int
    input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int


class ItemCompleted(BaseModel): ...


class TurnCompleted(BaseModel): ...


class TurnStarted(BaseModel): ...


class ThreadStarted(BaseModel): ...


Component = (
    Delta
    | TextLog
    | JsonLog
    | TokenUsage
    | ItemCompleted
    | Nop
    | TurnCompleted
    | TurnStarted
    | ThreadStarted
)


class EventRenderer[T](ABC):
    @abstractmethod
    def render(self, event: Event[T]) -> Iterator[Component]: ...


class Prompt(BaseModel):
    text: str


class Steer(BaseModel):
    text: str


class Stop(BaseModel): ...


Interrupt = Nop | Steer | Stop | Prompt
