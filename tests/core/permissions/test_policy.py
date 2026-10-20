from pathlib import Path

import pytest

from core.llm import ToolCall
from core.permissions import Decision, Policy, ToolRule

REPO_ROOT = Path(__file__).resolve().parents[3]


def call(name: str, **args: object) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, input=dict(args))


def test_default_policy_file_loads() -> None:
    policy = Policy.from_yaml(REPO_ROOT / "policies" / "default.yaml")

    assert policy.default is Decision.DENY
    assert policy.tools["shell"].decision is Decision.DENY
    assert policy.tools["write_file"] == ToolRule(
        decision=Decision.REQUIRE_APPROVAL, max_calls_per_task=20, allowed_paths=("./workspace",)
    )
    assert policy.requires_approval


def test_decisions_by_tool() -> None:
    policy = Policy.from_dict(
        {
            "tools": {
                "search": {"decision": "allow"},
                "shell": {"decision": "deny"},
                "send_email": {"decision": "require_approval"},
            }
        }
    )

    assert policy.check(call("search"), {}).decision is Decision.ALLOW
    assert policy.check(call("shell"), {}).decision is Decision.DENY
    assert policy.check(call("send_email"), {}).decision is Decision.REQUIRE_APPROVAL
    unlisted = policy.check(call("unlisted"), {})
    assert unlisted.decision is Decision.DENY
    assert unlisted.reason == "tool 'unlisted' is not listed in the policy"


def test_unlisted_tools_follow_the_default() -> None:
    assert Policy.from_dict({"default": "allow"}).check(call("x"), {}).decision is Decision.ALLOW


def test_max_calls_per_task() -> None:
    policy = Policy.from_dict({"tools": {"search": {"decision": "allow", "max_calls_per_task": 2}}})

    assert policy.check(call("search"), {"search": 1}).decision is Decision.ALLOW
    exceeded = policy.check(call("search"), {"search": 2})
    assert exceeded.decision is Decision.DENY
    assert exceeded.reason == "tool 'search' reached its limit of 2 call(s) per task"


def test_allowed_path_prefixes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    policy = Policy.from_dict(
        {
            "tools": {
                "copy_file": {
                    "decision": "allow",
                    "allowed_paths": [str(workspace)],
                    "path_arguments": ["source", "target"],
                }
            }
        }
    )

    def decide(**args: object) -> Decision:
        return policy.check(call("copy_file", **args), {}).decision

    assert decide(source=f"{workspace}/a.txt", target=f"{workspace}/sub/b.txt") is Decision.ALLOW
    assert decide(source=f"{workspace}/a.txt", target="/etc/passwd") is Decision.DENY
    assert decide(source=f"{workspace}/../secrets.txt") is Decision.DENY
    assert decide(source=f"{workspace}-evil/a.txt") is Decision.DENY
    assert decide(source=42) is Decision.DENY
    assert decide(other="x") is Decision.DENY  # no path argument at all


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"tools": {"x": {"decision": "maybe"}}}, "unknown decision 'maybe'"),
        ({"tools": {"x": {}}}, "missing 'decision'"),
        ({"tools": {"x": {"decision": "allow", "max_calls": 3}}}, "unknown key(s) max_calls"),
        ({"tools": {"x": {"decision": "allow", "max_calls_per_task": -1}}}, "non-negative"),
        ({"tools": {"x": {"decision": "allow", "allowed_paths": "/tmp"}}}, "list of strings"),
        ({"default": "allow", "tool": {}}, "unknown key(s) tool"),
    ],
)
def test_invalid_policies_are_rejected(data: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        Policy.from_dict(data)


def test_invalid_policy_file_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("tools:\n  x:\n    decision: nope\n")

    with pytest.raises(ValueError, match="Invalid policy file .*bad.yaml"):
        Policy.from_yaml(path)


def test_allowed_domains() -> None:
    policy = Policy.from_dict(
        {"tools": {"fetch_url": {"decision": "allow", "allowed_domains": ["python.org"]}}}
    )

    def decide(url: object) -> Decision:
        return policy.check(call("fetch_url", url=url), {}).decision

    assert decide("https://python.org/") is Decision.ALLOW
    assert decide("http://docs.PYTHON.org/3/library") is Decision.ALLOW
    assert decide("https://evilpython.org/") is Decision.DENY
    assert decide("https://python.org.evil.com/") is Decision.DENY
    assert decide("ftp://python.org/") is Decision.DENY
    assert decide("https://user@evil.com/?python.org") is Decision.DENY
    assert decide(42) is Decision.DENY
    reason = policy.check(call("fetch_url"), {}).reason
    assert reason == "tool 'fetch_url' needs a URL argument (url)"
