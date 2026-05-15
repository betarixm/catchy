import operator
from pathlib import Path
from typing import Annotated, Any, Callable, NotRequired, TypedDict
from collections.abc import Awaitable

from ..agent.models import Event
from ..challenge.models import Challenge
from ..webhook.models import Webhook


class State(TypedDict):
    challenge: Challenge
    workspace_directory: Path
    metadata_directory: Path
    metadata_directory_factory: NotRequired[Callable[[str], Path]]
    webhook: Webhook | None
    event_observer: Callable[[Event[Any]], None | Awaitable[None]]
    messages: Annotated[list[str], operator.add]
