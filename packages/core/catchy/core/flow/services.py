import inspect
from typing import Any

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel

from ..agent.models import Event, Nop
from ..agent.protocols import Agent
from .models import State


class AgentNode[T]:
    def __init__(
        self,
        agent: Agent[T],
    ) -> None:
        self._agent = agent

    @property
    def id(self) -> str:
        return self._agent.id

    async def __call__(self, state: State):
        metadata_directory_factory = state.get("metadata_directory_factory")
        metadata_directory = (
            metadata_directory_factory(self.id)
            if callable(metadata_directory_factory)
            else state["metadata_directory"]
        )
        metadata_directory.mkdir(parents=True, exist_ok=True)
        stream = self._agent.stream(
            challenge=state["challenge"],
            workspace_directory=state["workspace_directory"],
            metadata_directory=metadata_directory,
            webhook=state["webhook"],
            additional_prompt="\n\n".join(state["messages"]),
        )

        events: list[Event[T]] = []
        is_started = False
        while True:
            try:
                if not is_started:
                    event = await stream.__anext__()
                    is_started = True
                else:
                    event = await stream.asend(Nop())  # TODO: Support other interrupts
                observer_result = state["event_observer"](event)
                if inspect.isawaitable(observer_result):
                    await observer_result
                events.append(event)
            except StopAsyncIteration:
                break

        return {"messages": [self._agent.last_agent_message_from_events(events)]}


class FlowConfiguration(BaseModel):
    agents: list[dict[str, Any]]  # list of agent_config
    edges: list[tuple[str, str]]  # list of (from_id, to_id), supports __start__/__end__


class Flow:
    @staticmethod
    def from_configuration(configuration: FlowConfiguration):
        agents = [
            Agent.from_configuration(agent_config)
            for agent_config in configuration.agents
        ]
        agent_ids = {agent.id for agent in agents}
        edges = [tuple(edge) for edge in configuration.edges]

        if not edges:
            raise ValueError("flow must have at least one edge")

        for source, target in edges:
            if source == "__end__":
                raise ValueError("__end__ cannot be used as an edge source")
            if target == "__start__":
                raise ValueError("__start__ cannot be used as an edge target")
            if source not in agent_ids and source != "__start__":
                raise ValueError(f"unknown edge source: {source}")
            if target not in agent_ids and target != "__end__":
                raise ValueError(f"unknown edge target: {target}")

        if not any(source == "__start__" for source, _ in edges):
            incoming: dict[str, int] = {node_id: 0 for node_id in agent_ids}
            for _, target in edges:
                if target in incoming:
                    incoming[target] += 1
            roots = [node_id for node_id, degree in incoming.items() if degree == 0]
            edges.extend(("__start__", node_id) for node_id in roots)

        if not any(target == "__end__" for _, target in edges):
            outgoing: dict[str, int] = {node_id: 0 for node_id in agent_ids}
            for source, _ in edges:
                if source in outgoing:
                    outgoing[source] += 1
            leaves = [node_id for node_id, degree in outgoing.items() if degree == 0]
            edges.extend((node_id, "__end__") for node_id in leaves)

        return Flow(nodes=[AgentNode(agent) for agent in agents], edges=edges)

    def __init__(
        self, nodes: list[AgentNode[Any]], edges: list[tuple[str, str]]
    ) -> None:
        builder = StateGraph(State)
        for node in nodes:
            builder.add_node(node.id, node)  # pyright: ignore[reportUnknownMemberType]
        for from_id, to_id in edges:
            source = START if from_id == "__start__" else from_id
            target = END if to_id == "__end__" else to_id
            builder.add_edge(source, target)

        self.graph = builder.compile()  # pyright: ignore
