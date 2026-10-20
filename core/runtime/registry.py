from collections.abc import Iterable, Iterator
from typing import Any, Protocol

from core.runtime.agent import Tool


class ToolDefinition(Protocol):
    """A tool implementation: a name, a JSON Schema for its input, and an async run()."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def input_schema(self) -> dict[str, Any]: ...

    async def run(self, args: dict[str, Any]) -> str: ...


class ToolRegistry:
    def __init__(self, definitions: Iterable[ToolDefinition] = ()) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool {definition.name!r} is already registered")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError:
            raise KeyError(f"No tool named {name!r}") from None

    def tools(self) -> list[Tool]:
        """The registered tools in the form AgentRuntime takes."""
        return [
            Tool(d.name, d.description, d.input_schema, d.run) for d in self._definitions.values()
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._definitions

    def __iter__(self) -> Iterator[str]:
        return iter(self._definitions)

    def __len__(self) -> int:
        return len(self._definitions)
