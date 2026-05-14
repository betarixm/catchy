from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class Nop(BaseModel): ...


class Chunk(BaseModel):
    tag: Literal["observation", "action"] | str
    text: str


class Log(BaseModel):
    kind: str
    text: str = ""
    raw: dict[str, object] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    provider: str | None = None
    model: str | None = None
    source: str | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int | None = None
    raw: dict[str, object] = Field(default_factory=dict)

    @field_validator(
        "input_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
        mode="before",
    )
    @classmethod
    def _deserialize_token_count(
        cls, value: object, info: ValidationInfo
    ) -> int | None:
        if value is None:
            return None if info.field_name == "total_tokens" else 0
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdecimal():
            return int(value)
        return 0

    def usage_dict(self) -> dict[str, int]:
        usage = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_token_count,
        }
        for key in (
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "reasoning_output_tokens",
        ):
            value = getattr(self, key)
            if value:
                usage[key] = value
        return usage

    @property
    def total_token_count(self) -> int:
        if self.total_tokens is not None:
            return self.total_tokens
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.output_tokens
        )

    def event_raw(self) -> dict[str, object]:
        raw: dict[str, object] = {"usage": self.usage_dict()}
        if self.provider:
            raw["provider"] = self.provider
        if self.model:
            raw["model"] = self.model
        if self.source:
            raw["source"] = self.source
        if self.raw:
            raw["raw"] = self.raw
        return raw


class ItemCompleted(BaseModel): ...


class TurnCompleted(BaseModel): ...


Event = Chunk | Log | TokenUsage | ItemCompleted | Nop | TurnCompleted


class Prompt(BaseModel):
    text: str


class Steer(BaseModel):
    text: str


class Stop(BaseModel): ...


Interrupt = Nop | Steer | Stop | Prompt
