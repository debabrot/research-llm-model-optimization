"""Phase 3: LLM client, cache/replay, budget guard, prompts, query tool, sessions, LLM arms. No network."""

import copy
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tree_surgery.arms import build_context, get_arm
from tree_surgery.config import load_config
from tree_surgery.llm_client import (
    AnthropicClient,
    BudgetExceeded,
    BudgetGuard,
    CachedClient,
    CacheMiss,
    LLMError,
    LLMResponse,
    MockClient,
    OpenAICompatClient,
    Price,
    ToolCall,
    Usage,
    make_client,
    request_key,
    to_anthropic,
)
from tree_surgery.llm_session import make_mock_policy, run_session
from tree_surgery.prompts import (
    LeafView,
    PromptSpec,
    ProposalError,
    _feature_pattern,
    history_block,
    HistoryItem,
    parse_proposal,
    run_query,
    static_block,
)
from tree_surgery.runner import run_grid

CFG = load_config()
PRICE = Price(0.10, 0.60, 0.03)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ctx(delivery):
    data, booster = delivery
    return build_context(data, booster, CFG)


def mock(invalid_rate=0.0, queries=2):
    return CachedClient(MockClient(make_mock_policy(invalid_rate, queries), price=PRICE), "mock", "live", None)


class Recorder:
    """Wraps a client and records every request."""

    def __init__(self, inner):
        self.inner, self.model, self.requests = inner, inner.model, []
        self.log = getattr(inner, "log", [])

    @property
    def cache_tag(self):
        return self.inner.cache_tag

    @cache_tag.setter
    def cache_tag(self, v):
        self.inner.cache_tag = v

    def chat(self, messages, tools=None, tool_choice=None, **params):
        self.requests.append({"messages": copy.deepcopy(messages), "tool_choice": tool_choice, "params": params})
        return self.inner.chat(messages, tools, tool_choice, **params)


class Scripted:
    """Inner client that returns queued responses."""

    def __init__(self, responses, cost=0.001):
        self.model, self.queue, self.calls, self.cost = "scripted", list(responses), 0, cost

    def chat(self, messages, tools=None, tool_choice=None, **params):
        self.calls += 1
        r = self.queue.pop(0) if self.queue else LLMResponse("{}")
        r.cost_usd = self.cost
        return r


MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


# --------------------------------------------------------------------------- #
# client: keys, cache, replay, budget
# --------------------------------------------------------------------------- #
def test_request_key_sensitivity():
    k = request_key("m", MSGS, None, {"max_tokens": 10}, "run0")
    assert k == request_key("m", copy.deepcopy(MSGS), None, {"max_tokens": 10}, "run0")
    assert k != request_key("m", MSGS, None, {"max_tokens": 10}, "run1")
    assert k != request_key("m", MSGS, None, {"max_tokens": 11}, "run0")
    assert k != request_key("m2", MSGS, None, {"max_tokens": 10}, "run0")
    assert k != request_key("m", MSGS + [{"role": "user", "content": "x"}], None, {"max_tokens": 10}, "run0")


def test_cache_live_then_replay(tmp_path):
    inner = Scripted([LLMResponse("hi", [ToolCall("c1", "query_leaves", '{"limit": 3}')], "tool_calls",
                                  Usage(100, 20, 50, 5), model="m-2026")])
    live = CachedClient(inner, "m", "live", tmp_path)
    r1 = live.chat(MSGS, max_tokens=10)
    r2 = live.chat(MSGS, max_tokens=10)
    assert inner.calls == 1 and not r1.from_cache and r2.from_cache
    assert live.log[0]["billed_usd"] == pytest.approx(0.001) and live.log[1]["billed_usd"] == 0
    assert live.log[1]["cost_usd"] == pytest.approx(0.001)  # original cost kept for accounting

    replay = CachedClient(None, "m", "replay", tmp_path)
    r3 = replay.chat(MSGS, max_tokens=10)
    assert r3.content == "hi" and r3.tool_calls[0].arguments == '{"limit": 3}'
    assert r3.usage == Usage(100, 20, 50, 5) and r3.model == "m-2026"
    with pytest.raises(CacheMiss):
        replay.chat(MSGS, max_tokens=11)
    replay.cache_tag = "run1"
    with pytest.raises(CacheMiss):
        replay.chat(MSGS, max_tokens=10)
    rec = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])
    assert "request" not in rec and rec["tag"] == ""


def test_cache_modes_validated(tmp_path):
    with pytest.raises(ValueError):
        CachedClient(None, "m", "live", tmp_path)
    with pytest.raises(ValueError):
        CachedClient(None, "m", "record", tmp_path)


def test_budget_guard_refuses_before_calling(tmp_path):
    inner = Scripted([LLMResponse("a"), LLMResponse("b"), LLMResponse("c")], cost=0.0045)
    ledger = tmp_path / "ledger.json"
    g = BudgetGuard(inner, max_usd=0.010, price=Price(1.0, 1.0), ledger=ledger, default_max_tokens=1000)
    g.chat(MSGS)          # worst case ~$0.001 + spent 0
    g.chat(MSGS)          # spent 0.0045
    assert g.spent_usd == pytest.approx(0.009)
    with pytest.raises(BudgetExceeded):
        g.chat(MSGS)      # 0.009 + worst case (~1000 output tokens = $0.001) > 0.010
    assert inner.calls == 2
    again = BudgetGuard(inner, max_usd=0.010, price=Price(1.0, 1.0), ledger=ledger)
    assert again.spent_usd == pytest.approx(0.009) and again.calls == 2


def test_budget_guard_counts_max_tokens():
    g = BudgetGuard(Scripted([]), max_usd=1.0, price=Price(0.0, 1e6))  # $1 per output token
    with pytest.raises(BudgetExceeded):
        g.chat(MSGS, max_tokens=2)


def test_price_cost_with_cache():
    u = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, cached_tokens=400_000)
    assert Price(1.0, 2.0, 0.1).cost(u) == pytest.approx(0.6 + 0.04 + 2.0)
    assert Price(1.0, 2.0).cost(u) == pytest.approx(3.0)


def test_openai_compat_request_and_parse(monkeypatch):
    c = OpenAICompatClient("deepseek/x", base_url="https://openrouter.ai/api/v1", api_key="k",
                           price=PRICE, reasoning_effort="low")
    seen = {}

    def create(**kw):
        seen.update(kw)
        msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            id="t1", function=SimpleNamespace(name="query_leaves", arguments='{"limit": 2}'))])
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, cost=0.00042,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=600),
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=150))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
                               usage=usage, model="deepseek/x-20261101")

    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    r = c.chat(MSGS, tools=[{"type": "function", "function": {"name": "q", "parameters": {}}}],
               tool_choice="none", max_tokens=99)
    assert seen["extra_body"] == {"usage": {"include": True}, "reasoning": {"effort": "low"}}
    assert seen["tool_choice"] == "none" and seen["max_tokens"] == 99
    assert r.cost_usd == pytest.approx(0.00042) and r.model == "deepseek/x-20261101"
    assert r.usage == Usage(1000, 200, 600, 150)
    assert r.tool_calls[0].name == "query_leaves"


def test_openai_compat_needs_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        OpenAICompatClient("m", base_url="https://openrouter.ai/api/v1")


def test_to_anthropic_conversion():
    msgs = [
        {"role": "system", "content": "static"},
        {"role": "user", "content": "history"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "query_leaves", "arguments": '{"limit": 1}'}},
            {"id": "b", "type": "function", "function": {"name": "query_leaves", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "r1"},
        {"role": "tool", "tool_call_id": "b", "content": "r2"},
    ]
    tools = [{"type": "function", "function": {"name": "query_leaves", "description": "d", "parameters": {"type": "object"}}}]
    system, out, a_tools = to_anthropic(msgs, tools)
    assert system == [{"type": "text", "text": "static", "cache_control": {"type": "ephemeral"}}]
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["tool_use", "tool_use"]
    assert out[1]["content"][0]["input"] == {"limit": 1}
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]
    assert a_tools == [{"name": "query_leaves", "description": "d", "input_schema": {"type": "object"}}]


def test_anthropic_adapter_parse(monkeypatch):
    c = AnthropicClient.__new__(AnthropicClient)
    c.model, c.price = "claude-x", Price(4.0, 20.0, 0.2)
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok"),
                 SimpleNamespace(type="tool_use", id="t", name="query_leaves", input={"limit": 1})],
        stop_reason="tool_use", model="claude-x-1",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=900,
                              cache_creation_input_tokens=0))
    c._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
    r = c.chat(MSGS, max_tokens=10)
    assert r.content == "ok" and r.tool_calls[0].arguments == '{"limit": 1}'
    assert r.usage == Usage(1000, 50, 900, 0)


def test_make_client_modes(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    r = make_client(CFG["llm"], mode="replay", cache_dir=tmp_path)
    assert r.mode == "replay" and r.model == CFG["llm"]["models"]["cheap"]
    m = make_client(CFG["llm"], mode="mock", cache_dir=tmp_path)
    assert m.model == "mock" and m.path is None
    with pytest.raises(LLMError):
        make_client(CFG["llm"], mode="live", cache_dir=tmp_path)
    with pytest.raises(LLMError, match="no price"):
        make_client(CFG["llm"], mode="live", model="unknown/model", cache_dir=tmp_path)


def test_mock_cost_is_notional():
    c = CachedClient(MockClient(lambda *a: LLMResponse("ok"), price=PRICE), "mock", "live", None)
    c.chat(MSGS)
    assert c.log[0]["cost_usd"] > 0 and c.log[0]["billed_usd"] == 0


# --------------------------------------------------------------------------- #
# query tool and leaf views
# --------------------------------------------------------------------------- #
def test_feature_pattern_whole_token():
    p = _feature_pattern("f1")
    assert p.search("f1 > 2 AND f10 <= 3") and not p.search("f10 <= 3 AND f11 > 1")
    assert not _feature_pattern("hour").search("hours <= 3")


def test_query_filters_sort_limit(ctx):
    view = ctx.leaf_view("named")
    out = run_query(view, {"features": ["rain_mm"], "min_n_cal": 20, "sort_by": "abs_mean_resid", "limit": 100}, 30)
    head, *lines = out.splitlines()
    n_match = int(re.match(r"(\d+) leaves matched; showing (\d+)", head).group(1))
    t = view.table
    expected = t[t.rule.str.contains(_feature_pattern("rain_mm")) & (t.n_cal >= 20)]
    assert n_match == len(expected) and len(lines) == 30
    vals = [abs(float(re.search(r"mean_resid=([+-][\d.eE+-]+)", l).group(1))) for l in lines]
    assert vals == sorted(vals, reverse=True)
    top = expected.sort_values("abs_mean_resid", ascending=False, kind="stable").head(30)  # ties: twin leaves
    assert lines == [view.line(r) for r in top.to_dict("records")]


def test_query_errors_are_messages(ctx):
    view = ctx.leaf_view("named")
    assert run_query(view, {"sort_by": "bogus"}, 30).startswith("ERROR")
    assert run_query(view, {"features": ["nope"]}, 30).startswith("ERROR: unknown feature")
    assert run_query(view, "{not json", 30).startswith("ERROR")
    assert run_query(view, {"limit": 5, "extra": 1}, 30).startswith("ERROR")


def test_query_tree_and_ranges(ctx):
    view = ctx.leaf_view("named")
    out = run_query(view, {"tree_id": 3, "sort_by": "tree_id"}, 30).splitlines()
    assert all(l.startswith("T3/") for l in out[1:]) and len(out) - 1 == (view.table.tree_id == 3).sum()
    out = run_query(view, {"min_mean_resid": 1.0, "max_n_cal": 50}, 30).splitlines()[1:]
    for l in out:
        assert float(re.search(r"mean_resid=([+-][\d.eE+-]+)", l).group(1)) >= 1.0
        assert int(re.search(r"n_cal=(\d+)", l).group(1)) <= 50


def test_anon_hides_names(ctx):
    data = ctx.data
    spec = PromptSpec("select", 10, 5, 3, 30)
    view = ctx.leaf_view("anon")
    static = static_block(spec, data, view, ctx.base, ctx.base_pred["cal"])
    text = static + "\n".join(view.table.rule)
    for name in data.feature_names + [data.target_name]:
        assert name not in text
    for f in data.features:
        assert f.description not in static
    assert sorted(view.features_shown) == sorted(f"f{i}" for i in range(len(data.feature_names)))
    # same leaves, same numbers: only names differ
    named = ctx.leaf_view("named").table
    pd.testing.assert_frame_equal(named.drop(columns="rule"), view.table.drop(columns="rule"))


def test_named_and_anon_share_instructions(ctx):
    spec = PromptSpec("direct", 10, 5, 3, 30, ctx.delta_bound)
    a = static_block(spec, ctx.data, ctx.leaf_view("named"), ctx.base, ctx.base_pred["cal"])
    b = static_block(spec, ctx.data, ctx.leaf_view("anon"), ctx.base, ctx.base_pred["cal"])
    strip = lambda s: s[: s.index("# Data")] + s[s.index("# Model"):]
    assert strip(a) == strip(b)


def test_static_block_has_no_leaf_list(ctx):
    for approach in ("select", "direct"):
        spec = PromptSpec(approach, 10, 5, 3, 30, ctx.delta_bound if approach == "direct" else None)
        s = static_block(spec, ctx.data, ctx.leaf_view("named"), ctx.base, ctx.base_pred["cal"])
        assert not re.search(r"T\d+/L\d+", s)


def test_history_block():
    h = [HistoryItem([(1, 2), (3, 4)], None, 4.21234), HistoryItem([], None, None, "invalid answer"),
         HistoryItem([(5, 6)], [0.5], 4.1)]
    s = history_block(h, 4.25, 2, 2, 5)
    assert "Round 2 of 2" in s and "T1/L2, T3/L4" in s and "FAILED" in s and "T5/L6 +0.5" in s
    assert "validation" not in s.lower() and "test" not in s.lower()


# --------------------------------------------------------------------------- #
# proposal parsing
# --------------------------------------------------------------------------- #
def _prop(cands):
    return json.dumps({"candidates": cands})


def test_parse_proposal_valid_and_fenced(ctx):
    k = ctx.index.key(0)
    good = _prop([{"leaves": [{"tree_id": k[0], "leaf_id": k[1]}]}])
    assert len(parse_proposal(good, "select", ctx.index, 10, 5).candidates) == 1
    assert parse_proposal(f"Here:\n```json\n{good}\n```", "select", ctx.index, 10, 5)


@pytest.mark.parametrize("bad,match", [
    ("not json", "not valid JSON"),
    ("", "empty"),
    (_prop([{"leaves": [{"tree_id": 99999, "leaf_id": 0}]}]), "do not exist"),
    (_prop([{"leaves": [{"tree_id": 0, "leaf_id": 0}, {"tree_id": 0, "leaf_id": 0}]}]), "twice"),
    (_prop([{"leaves": [{"tree_id": 0, "leaf_id": i % 3, "x": 1} for i in range(1)]}] * 6), "at most 5"),
    (_prop([{"leaves": []}]), "schema"),
    (json.dumps({"cands": []}), "schema"),
])
def test_parse_proposal_rejects(ctx, bad, match):
    with pytest.raises(ProposalError, match=match):
        parse_proposal(bad, "select", ctx.index, 10, 5)


def test_parse_proposal_leaf_limit_and_delta_bound(ctx):
    keys = [ctx.index.key(c) for c in range(11)]
    with pytest.raises(ProposalError, match="at most 10"):
        parse_proposal(_prop([{"leaves": [{"tree_id": t, "leaf_id": l} for t, l in keys]}]), "select", ctx.index, 10, 5)
    t, l = keys[0]
    with pytest.raises(ProposalError, match="delta"):
        parse_proposal(_prop([{"leaves": [{"tree_id": t, "leaf_id": l, "delta": 99.0}]}]), "direct", ctx.index, 10, 5, 1.0)
    with pytest.raises(ProposalError, match="schema"):  # direct needs delta
        parse_proposal(_prop([{"leaves": [{"tree_id": t, "leaf_id": l}]}]), "direct", ctx.index, 10, 5, 1.0)


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #
def _session(ctx, client, approach="select", queries=3):
    spec = PromptSpec(approach, 10, 5, queries, 30, ctx.delta_bound if approach == "direct" else None)
    view = ctx.leaf_view("named")
    static = static_block(spec, ctx.data, view, ctx.base, ctx.base_pred["cal"])
    return run_session(client, static, history_block([], 4.0, 1, 2, 5), view, spec, ctx.index, {"max_tokens": 100})


def test_session_mock_happy_path(ctx):
    res = _session(ctx, mock())
    assert res.ok and res.n_queries == 2 and res.n_calls == 3 and not res.retried
    assert len(res.proposal.candidates) == 5 and res.usage.prompt_tokens > 0


def test_session_retries_once_then_succeeds(ctx):
    res = _session(ctx, mock(invalid_rate=1.0))
    assert res.ok and res.retried and res.n_calls == 4


def test_session_fails_after_second_invalid(ctx):
    inner = Scripted([LLMResponse("garbage"), LLMResponse("still garbage")])
    res = _session(ctx, CachedClient(inner, "s", "live", None))
    assert not res.ok and res.retried and "after retry" in res.error and inner.calls == 2


def test_session_enforces_query_limit(ctx):
    rec = Recorder(mock(queries=99))
    res = _session(ctx, rec, queries=3)
    assert res.ok and res.n_queries == 3
    assert [r["tool_choice"] for r in rec.requests] == [None, None, None, "none"]


def test_session_propagates_cache_miss(ctx, tmp_path):
    with pytest.raises(CacheMiss):
        _session(ctx, CachedClient(None, "nothing", "replay", tmp_path))


# --------------------------------------------------------------------------- #
# LLM arms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ["llm_select", "llm_direct"])
@pytest.mark.parametrize("variant", ["named", "anon"])
def test_llm_arm_budget_selection_linearity(ctx, arm, variant):
    out = get_arm(arm, mock(), variant, 0).run(ctx)
    assert len(out.candidates) == CFG["edit_budget"]["max_candidates"]
    assert all(len(c.leaves) <= CFG["edit_budget"]["max_leaves"] for c in out.candidates)
    ok = [c for c in out.candidates if c.status == "ok"]
    assert out.candidates[out.chosen].val_rmse == min(c.val_rmse for c in ok)
    assert out.info["linearity_err"] < 1e-9 and out.within_budget
    assert out.info["llm_calls"] == sum(s["n_calls"] for s in out.info["sessions"])
    assert [s["round"] for s in out.info["sessions"]] == [0, 1]
    if arm == "llm_direct":
        assert all(abs(d) <= ctx.delta_bound for c in ok for d in c.deltas)


def test_llm_select_feedback_is_calibration_only(delivery):
    """The conversation must not change when validation labels change."""
    data, booster = delivery

    def conversation(d):
        c = build_context(d, booster, CFG)
        rec = Recorder(mock())
        get_arm("llm_select", rec, "named", 0).run_sessions(c)
        return [r["messages"] for r in rec.requests]

    poisoned = copy.deepcopy(data)
    poisoned.splits["val"].y = poisoned["val"].y + 100.0
    poisoned.splits["test"].y = poisoned["test"].y * np.nan
    assert conversation(data) == conversation(poisoned)


def test_llm_runs_differ_and_failed_sessions_count(ctx):
    a = get_arm("llm_select", mock(), "named", 0).run(ctx)
    b = get_arm("llm_select", mock(), "named", 1).run(ctx)
    assert [c.leaves for c in a.candidates] != [c.leaves for c in b.candidates]

    inner = Scripted([LLMResponse("x")] * 10)
    out = get_arm("llm_select", CachedClient(inner, "s", "live", None), "named", 0).run(ctx)
    assert all(c.status == "failed" for c in out.candidates) and len(out.candidates) == 10
    assert out.chosen is None and out.leaves_edited == 0


@pytest.mark.parametrize("arm", ["llm_select", "llm_direct"])
def test_llm_arm_replay_reproduces_live(ctx, tmp_path, arm):
    live = CachedClient(MockClient(make_mock_policy(0.2), price=PRICE), "m", "live", tmp_path)
    a = get_arm(arm, live, "anon", 2).run(ctx)
    replay = CachedClient(None, "m", "replay", tmp_path)
    b = get_arm(arm, replay, "anon", 2).run(ctx)
    assert [c.to_dict() for c in a.candidates] == [c.to_dict() for c in b.candidates]
    assert a.info["llm_tokens_in"] == b.info["llm_tokens_in"] and a.info["llm_cost_usd"] == b.info["llm_cost_usd"]
    assert b.info["llm_billed_usd"] == 0 and b.info["llm_cache_hits"] == b.info["llm_calls"]
    with pytest.raises(CacheMiss):
        get_arm(arm, replay, "anon", 3).run(ctx)


def test_llm_arm_requires_client():
    with pytest.raises(ValueError, match="needs an LLM client"):
        get_arm("llm_select")


def test_runner_with_llm_arms(tmp_path):
    df = run_grid(["delivery"], [0], ["original", "sparse_opt", "llm_select", "llm_direct"], CFG, tmp_path,
                  verbose=False, client=mock(), llm_runs=2)
    assert len(df) == 2 + 2 * 2 * 2
    llm = df[df.arm.str.startswith("llm")]
    assert set(llm.variant) == {"named", "anon"} and set(llm.run) == {0, 1}
    assert (llm.llm_tokens_in > 0).all() and (llm.llm_billed_usd == 0).all()
    recs = [json.loads(l) for l in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert sum("sessions" in r for r in recs) == 8
    summary = (tmp_path / "summary.md").read_text()
    assert "llm_select/named − sparse_opt" in summary and "## LLM usage" in summary
