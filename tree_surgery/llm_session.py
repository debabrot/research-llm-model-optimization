"""One LLM session: a short tool loop that ends in a JSON proposal (PLAN §5, changelog 2026-11-18).

    system: static block            (cacheable prefix)
    user:   history block           (previous candidates + calibration RMSE)
    -> up to ``max_queries`` query_leaves calls (tool results appended)
    -> final answer: JSON with ``candidates_per_session`` candidates
    -> if invalid: one retry with the validation error; still invalid = failed session

Once the query limit is reached the next call is made with ``tool_choice="none"``.
API errors, :class:`~tree_surgery.llm_client.BudgetExceeded` and
:class:`~tree_surgery.llm_client.CacheMiss` propagate: they stop the run rather than
being counted as a failed candidate (a resumed run picks up the cached calls for free).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .design_matrix import LeafIndex
from .llm_client import LLMClient, LLMResponse, ToolCall, Usage
from .prompts import QUERY_TOOL_NAME, LeafView, PromptSpec, ProposalError, parse_proposal, query_tool_schema, run_query

RETRY_TEMPLATE = "Your answer could not be used: {error}\nReply again with only the corrected JSON object."


@dataclass
class SessionResult:
    proposal: Any = None                 # SelectProposal | DirectProposal | None
    error: Optional[str] = None          # set when the session failed
    n_queries: int = 0
    n_calls: int = 0
    retried: bool = False
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    queries: list[str] = field(default_factory=list)   # raw query arguments, for analysis

    @property
    def ok(self) -> bool:
        return self.proposal is not None


def run_session(
    client: LLMClient,
    static: str,
    history: str,
    view: LeafView,
    spec: PromptSpec,
    index: LeafIndex,
    params: Optional[dict[str, Any]] = None,
) -> SessionResult:
    params = dict(params or {})
    tools = [query_tool_schema(spec.rows_per_query)]
    messages: list[dict[str, Any]] = [{"role": "system", "content": static}, {"role": "user", "content": history}]
    res = SessionResult()
    max_calls = spec.max_queries + 3  # queries + answer + retry + slack for a model that ignores tool_choice

    def call(tool_choice: Optional[str]) -> LLMResponse:
        r = client.chat(messages, tools, tool_choice, **params)
        res.n_calls += 1
        res.usage = res.usage + r.usage
        res.cost_usd += r.cost_usd
        return r

    while res.n_calls < max_calls:
        exhausted = res.n_queries >= spec.max_queries
        r = call("none" if exhausted or res.retried else None)
        if r.tool_calls and not exhausted and not res.retried:
            messages.append(r.assistant_message())
            for tc in r.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": _tool_result(tc, view, spec, res)})
            continue

        try:
            if r.tool_calls and not (r.content or "").strip():
                raise ProposalError("expected the final JSON answer, got another tool call (query limit reached)")
            res.proposal = parse_proposal(r.content, spec.approach, index, spec.max_leaves,
                                          spec.candidates_per_session, spec.delta_bound)
            return res
        except ProposalError as e:
            if res.retried:
                res.error = f"invalid after retry: {e}"
                return res
            res.retried = True
            messages.append({"role": "assistant", "content": r.content or ""})
            messages.append({"role": "user", "content": RETRY_TEMPLATE.format(error=e)})
    res.error = "call limit reached without a valid answer"
    return res


def _tool_result(tc: ToolCall, view: LeafView, spec: PromptSpec, res: SessionResult) -> str:
    if tc.name != QUERY_TOOL_NAME:
        return f"ERROR: unknown tool {tc.name!r}; the only tool is {QUERY_TOOL_NAME}."
    if res.n_queries >= spec.max_queries:
        return f"ERROR: query limit ({spec.max_queries}) reached. Answer now with the JSON object."
    res.n_queries += 1
    res.queries.append(tc.arguments)
    return run_query(view, tc.arguments, spec.rows_per_query)


# --------------------------------------------------------------------------- #
# Mock policy (offline dry runs and tests)
# --------------------------------------------------------------------------- #
_LEAF_LINE = re.compile(r"^T(\d+)/L(\d+) \|.*?\| n_cal=(\d+) \|.*?mean_resid=([+-]?[\d.eE+-]+|nan)", re.M)


def make_mock_policy(invalid_rate: float = 0.0, queries: int = 2):
    """A scripted stand-in for an LLM. Not a baseline: it only exercises the machinery.

    It makes ``queries`` tool calls (largest |mean_resid| among leaves with >= 30 rows, then
    leaves using the first listed feature), then proposes candidates drawn from the leaves it
    saw. Direct deltas are half the leaf's mean residual, clipped to the bound. With
    probability ``invalid_rate`` the first answer names a leaf that doesn't exist, to
    exercise the retry path.
    """

    def policy(messages, tools, tool_choice, rng: random.Random) -> LLMResponse:
        system = messages[0]["content"]
        n_tool = sum(m["role"] == "tool" for m in messages)
        is_retry = messages[-1]["role"] == "user" and messages[-1]["content"].startswith("Your answer could not")
        if tools and tool_choice != "none" and n_tool < queries:
            feats = re.findall(r"^- ([A-Za-z0-9_]+):", system, re.M)
            args = ({"sort_by": "abs_mean_resid", "min_n_cal": 30} if n_tool == 0
                    else {"features": feats[:1], "sort_by": "n_cal"})
            return LLMResponse(None, [ToolCall(f"call_{n_tool}", QUERY_TOOL_NAME, json.dumps(args))], "tool_calls")

        seen: dict[tuple[int, int], float] = {}
        for m in messages:
            if m["role"] == "tool":
                for t, l, n, mr in _LEAF_LINE.findall(m["content"]):
                    if int(n) > 0 and mr != "nan":
                        seen[(int(t), int(l))] = float(mr)
        n_cand = int(re.search(r"Give exactly (\d+) candidates", system).group(1))
        max_leaves = int(re.search(r"each with 1 to (\d+) distinct", system).group(1))
        bound_m = re.search(r"\|delta\| must be <= (\d+(?:\.\d+)?(?:e[+-]?\d+)?)", system)
        keys = sorted(seen) or [(0, 0)]
        cands = []
        for _ in range(n_cand):
            pick = rng.sample(keys, k=min(len(keys), rng.randint(3, max_leaves)))
            leaves = []
            for t, l in pick:
                item = {"tree_id": t, "leaf_id": l, "reason": "large residual" if abs(seen.get((t, l), 0)) > 0 else ""}
                if bound_m:
                    b = float(bound_m.group(1))
                    item["delta"] = max(-b, min(b, 0.5 * seen.get((t, l), 0.0)))
                leaves.append(item)
            cands.append({"leaves": leaves, "rationale": "mock"})
        if not is_retry and rng.random() < invalid_rate:
            cands[0]["leaves"][0]["tree_id"] = 10**6
        return LLMResponse(json.dumps({"candidates": cands}), [], "stop")

    return policy


mock_policy = make_mock_policy()
