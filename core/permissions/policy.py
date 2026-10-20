from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from core.llm import ToolCall

_RULE_KEYS = {
    "decision",
    "max_calls_per_task",
    "allowed_paths",
    "path_arguments",
    "allowed_domains",
    "url_arguments",
}
_POLICY_KEYS = {"default", "tools"}


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class ToolRule:
    decision: Decision
    max_calls_per_task: int | None = None
    # When set, every path argument must resolve inside one of these prefixes.
    allowed_paths: tuple[str, ...] = ()
    path_arguments: tuple[str, ...] = ("path",)
    # When set, every URL argument must be http(s) on one of these domains or a subdomain.
    allowed_domains: tuple[str, ...] = ()
    url_arguments: tuple[str, ...] = ("url",)


@dataclass(frozen=True)
class PolicyCheck:
    decision: Decision
    reason: str


@dataclass(frozen=True)
class Policy:
    tools: Mapping[str, ToolRule] = field(default_factory=dict)
    default: Decision = Decision.DENY

    @classmethod
    def allow_all(cls) -> "Policy":
        return cls(default=Decision.ALLOW)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Policy":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        try:
            return cls.from_dict(data)
        except ValueError as exc:
            raise ValueError(f"Invalid policy file {path}: {exc}") from None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Policy":
        _reject_unknown_keys(data, _POLICY_KEYS, "policy")
        tools = data.get("tools") or {}
        if not isinstance(tools, Mapping):
            raise ValueError("'tools' must be a mapping of tool name to rule")  # noqa: TRY004 - all config errors are ValueError
        return cls(
            tools={name: _parse_rule(name, rule) for name, rule in tools.items()},
            default=_parse_decision(data.get("default", Decision.DENY.value), "default"),
        )

    @property
    def requires_approval(self) -> bool:
        return self.default is Decision.REQUIRE_APPROVAL or any(
            rule.decision is Decision.REQUIRE_APPROVAL for rule in self.tools.values()
        )

    def check(self, call: ToolCall, calls_made: Mapping[str, int]) -> PolicyCheck:
        """Decide a tool call, given how many times each tool already ran in this task."""
        rule = self.tools.get(call.name)
        if rule is None:
            return PolicyCheck(self.default, f"tool {call.name!r} is not listed in the policy")
        if rule.decision is Decision.DENY:
            return PolicyCheck(Decision.DENY, f"tool {call.name!r} is denied by policy")

        limit = rule.max_calls_per_task
        if limit is not None and calls_made.get(call.name, 0) >= limit:
            return PolicyCheck(
                Decision.DENY, f"tool {call.name!r} reached its limit of {limit} call(s) per task"
            )

        for checker, enabled in (
            (_check_paths, rule.allowed_paths),
            (_check_urls, rule.allowed_domains),
        ):
            denial = checker(call, rule) if enabled else None
            if denial is not None:
                return PolicyCheck(Decision.DENY, denial)

        return PolicyCheck(rule.decision, f"tool {call.name!r} is {rule.decision.value} by policy")


def _check_paths(call: ToolCall, rule: ToolRule) -> str | None:
    paths = [call.input[arg] for arg in rule.path_arguments if arg in call.input]
    if not paths:
        return f"tool {call.name!r} needs a path argument ({', '.join(rule.path_arguments)})"
    for path in paths:
        if not isinstance(path, str) or not _within(path, rule.allowed_paths):
            return f"path {path!r} is outside the allowed prefixes for tool {call.name!r}"
    return None


def _check_urls(call: ToolCall, rule: ToolRule) -> str | None:
    urls = [call.input[arg] for arg in rule.url_arguments if arg in call.input]
    if not urls:
        return f"tool {call.name!r} needs a URL argument ({', '.join(rule.url_arguments)})"
    for url in urls:
        if not isinstance(url, str) or not url_allowed(url, rule.allowed_domains):
            return f"URL {url!r} is not on the allowed domains for tool {call.name!r}"
    return None


def url_allowed(url: str, domains: tuple[str, ...]) -> bool:
    """True for an http(s) URL whose host is one of `domains` or a subdomain of one."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not host:
        return False
    return any(host == d or host.endswith("." + d) for d in (d.lower() for d in domains))


def _within(path: str, prefixes: tuple[str, ...]) -> bool:
    # resolve() collapses '..' and follows symlinks, so neither can escape a prefix.
    resolved = Path(path).resolve()
    return any(resolved.is_relative_to(Path(prefix).resolve()) for prefix in prefixes)


def _parse_rule(name: str, raw: Any) -> ToolRule:
    if not isinstance(raw, Mapping):
        raise ValueError(f"rule for tool {name!r} must be a mapping")  # noqa: TRY004 - all config errors are ValueError
    _reject_unknown_keys(raw, _RULE_KEYS, f"tool {name!r}")
    if "decision" not in raw:
        raise ValueError(f"tool {name!r} is missing 'decision'")

    limit = raw.get("max_calls_per_task")
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError(f"tool {name!r}: max_calls_per_task must be a non-negative integer")

    return ToolRule(
        decision=_parse_decision(raw["decision"], f"tool {name!r}"),
        max_calls_per_task=limit,
        allowed_paths=_string_tuple(raw.get("allowed_paths", ()), name, "allowed_paths"),
        path_arguments=_string_tuple(raw.get("path_arguments", ("path",)), name, "path_arguments"),
        allowed_domains=_string_tuple(raw.get("allowed_domains", ()), name, "allowed_domains"),
        url_arguments=_string_tuple(raw.get("url_arguments", ("url",)), name, "url_arguments"),
    )


def _parse_decision(value: Any, where: str) -> Decision:
    try:
        return Decision(value)
    except ValueError:
        options = ", ".join(d.value for d in Decision)
        raise ValueError(
            f"{where}: unknown decision {value!r} (expected one of {options})"
        ) from None


def _string_tuple(value: Any, name: str, key: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"tool {name!r}: {key} must be a list of strings")
    return tuple(value)


def _reject_unknown_keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
