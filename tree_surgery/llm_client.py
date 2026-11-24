"""Provider-agnostic LLM client with cache / replay, budget guard and a mock (PLAN §7).

Messages and tools use the OpenAI chat format everywhere in this package::

    {"role": "system" | "user" | "assistant" | "tool", "content": ..., "tool_calls": [...], "tool_call_id": ...}
    tools = [{"type": "function", "function": {"name", "description", "parameters"}}]

Every client returns an :class:`LLMResponse` (content, tool calls, token usage, cost, latency).

Layers, outermost first::

    CachedClient(mode="live" | "replay")    # llm_cache/<model>.jsonl; replay fails on a miss
      -> BudgetGuard                        # refuses a call that could cross the spend cap
        -> OpenAICompatClient (OpenRouter, DeepSeek, ...) | AnthropicClient | MockClient

Use :func:`make_client` to build the stack from config + environment.

Cache key = sha256 of (model, messages, tools, sampling params, ``cache_tag``). ``cache_tag``
is not sent to the API; the runner sets it to the LLM run index, so the 3 runs per cell are
3 independent samples rather than one cached answer replayed three times.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .config import REPO_ROOT

DEFAULT_CACHE_DIR = REPO_ROOT / "llm_cache"
CHARS_PER_TOKEN = 4.0  # rough estimate used only for budget pre-checks and mock usage


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model produced it


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0      # prompt tokens served from the provider's prompt cache
    reasoning_tokens: int = 0   # included in completion_tokens (billed as output)

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(*(getattr(self, f) + getattr(o, f) for f in self.__dataclass_fields__))


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0           # actual provider charge for this call (0 for cache hits)
    latency_s: float = 0.0
    model: str = ""                 # model id the provider reports (pins the exact version)
    from_cache: bool = False

    def assistant_message(self) -> dict[str, Any]:
        """The message to append to the conversation before sending tool results back."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}}
                for t in self.tool_calls
            ]
        return msg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("from_cache")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], from_cache: bool = False) -> "LLMResponse":
        return cls(
            content=d.get("content"),
            tool_calls=[ToolCall(**t) for t in d.get("tool_calls", [])],
            finish_reason=d.get("finish_reason"),
            usage=Usage(**d.get("usage", {})),
            cost_usd=float(d.get("cost_usd", 0.0)),
            latency_s=float(d.get("latency_s", 0.0)),
            model=d.get("model", ""),
            from_cache=from_cache,
        )


@dataclass
class Price:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: Optional[float] = None

    def cost(self, u: Usage) -> float:
        cached = u.cached_tokens if self.cache_read is not None else 0
        cr = self.cache_read if self.cache_read is not None else self.input
        return ((u.prompt_tokens - cached) * self.input + cached * cr + u.completion_tokens * self.output) / 1e6


class LLMError(RuntimeError):
    pass


class CacheMiss(LLMError):
    """Replay mode found no cached response for a request."""


class BudgetExceeded(LLMError):
    """The next call could push total spend over the configured cap."""


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **params: Any,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def estimate_tokens(obj: Any) -> int:
    """Crude token estimate (chars / 4) of any JSON-serialisable object."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return int(len(text) / CHARS_PER_TOKEN) + 1


def request_key(model: str, messages: list[dict], tools: Optional[list], params: dict, cache_tag: str = "") -> str:
    payload = {"model": model, "messages": messages, "tools": tools or [], "params": params, "tag": cache_tag}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class OpenAICompatClient:
    """OpenAI-compatible chat completions (OpenRouter, DeepSeek, vLLM, ...).

    On OpenRouter the request asks for usage accounting, so the reported ``usage.cost``
    (USD) is used as the call's cost; otherwise cost comes from ``price``.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        price: Optional[Price] = None,
        reasoning_effort: Optional[str] = None,
        timeout_s: float = 180.0,
        max_retries: int = 2,
    ):
        from openai import OpenAI

        self.model = model
        self.price = price
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set (see .env.example)")
        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=timeout_s, max_retries=max_retries)

    @property
    def is_openrouter(self) -> bool:
        return "openrouter.ai" in (self.base_url or "")

    def chat(self, messages, tools=None, tool_choice=None, **params) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, **params}
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        extra: dict[str, Any] = {}
        if self.is_openrouter:
            extra["usage"] = {"include": True}
            if self.reasoning_effort:
                extra["reasoning"] = {"effort": self.reasoning_effort}
        elif self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if extra:
            kwargs["extra_body"] = extra

        t0 = time.perf_counter()
        r = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        choice = r.choices[0]
        msg = choice.message
        u = r.usage
        ptd = getattr(u, "prompt_tokens_details", None)
        ctd = getattr(u, "completion_tokens_details", None)
        usage = Usage(
            prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
            cached_tokens=int(getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0,
            reasoning_tokens=int(getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0,
        )
        cost = getattr(u, "cost", None)
        if cost is None and self.price is not None:
            cost = self.price.cost(usage)
        return LLMResponse(
            content=msg.content,
            tool_calls=[ToolCall(t.id, t.function.name, t.function.arguments or "{}") for t in (msg.tool_calls or [])],
            finish_reason=choice.finish_reason,
            usage=usage,
            cost_usd=float(cost or 0.0),
            latency_s=latency,
            model=getattr(r, "model", self.model) or self.model,
        )


def to_anthropic(messages: list[dict], tools: Optional[list[dict]] = None, cache_system: bool = True):
    """Convert OpenAI-format messages/tools to Anthropic Messages API arguments.

    Returns ``(system_blocks, messages, tools)``. The system prompt is marked for prompt
    caching; consecutive tool results are merged into one user turn as Anthropic requires.
    """
    system: list[dict] = []
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system.append({"type": "text", "text": m["content"]})
        elif role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
        elif role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for t in m.get("tool_calls", []) or []:
                blocks.append({"type": "tool_use", "id": t["id"], "name": t["function"]["name"],
                               "input": json.loads(t["function"]["arguments"] or "{}")})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and all(b["type"] == "tool_result" for b in out[-1]["content"]):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            raise ValueError(f"unknown role {role!r}")
    if cache_system and system:
        system[-1]["cache_control"] = {"type": "ephemeral"}
    a_tools = [
        {"name": t["function"]["name"], "description": t["function"].get("description", ""),
         "input_schema": t["function"]["parameters"]}
        for t in (tools or [])
    ]
    return system, out, a_tools


class AnthropicClient:
    """Direct Anthropic API adapter (``ANTHROPIC_API_KEY``). OpenRouter needs no adapter."""

    def __init__(self, model: str, api_key: Optional[str] = None, price: Optional[Price] = None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.price = price
        self._client = anthropic.Anthropic(api_key=key)

    def chat(self, messages, tools=None, tool_choice=None, **params) -> LLMResponse:
        system, msgs, a_tools = to_anthropic(messages, tools)
        kwargs: dict[str, Any] = {"model": self.model, "messages": msgs,
                                  "max_tokens": params.pop("max_tokens", 4000), **params}
        if system:
            kwargs["system"] = system
        if a_tools:
            kwargs["tools"] = a_tools
            if tool_choice == "none":
                kwargs["tool_choice"] = {"type": "none"}
        t0 = time.perf_counter()
        r = self._client.messages.create(**kwargs)
        latency = time.perf_counter() - t0
        text = "".join(b.text for b in r.content if b.type == "text") or None
        calls = [ToolCall(b.id, b.name, json.dumps(b.input)) for b in r.content if b.type == "tool_use"]
        cr = int(getattr(r.usage, "cache_read_input_tokens", 0) or 0)
        cw = int(getattr(r.usage, "cache_creation_input_tokens", 0) or 0)
        usage = Usage(prompt_tokens=r.usage.input_tokens + cr + cw, completion_tokens=r.usage.output_tokens,
                      cached_tokens=cr)
        return LLMResponse(text, calls, r.stop_reason, usage,
                           self.price.cost(usage) if self.price else 0.0, latency, r.model)


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
MockPolicy = Callable[[list[dict], Optional[list[dict]], Optional[str], random.Random], LLMResponse]


class MockClient:
    """Deterministic offline client. ``policy`` decides each reply (see ``tree_surgery.llm_session``).

    Usage is estimated from text length so token and cost projections can be made from a
    dry run; ``price`` turns that into a notional cost (not charged anywhere).
    """

    def __init__(self, policy: MockPolicy, model: str = "mock", price: Optional[Price] = None, seed: int = 0):
        self.model = model
        self.policy = policy
        self.price = price
        self.seed = seed
        self.n_calls = 0
        self.notional = True  # cost_usd is a projection at `price`; nothing is billed
        self.tag = ""         # set by CachedClient to the run tag, so runs differ like real samples

    def chat(self, messages, tools=None, tool_choice=None, **params) -> LLMResponse:
        # Seed from the conversation itself so the reply doesn't depend on call order.
        h = int(request_key(self.model, messages, tools, {"tc": tool_choice}, f"{self.seed}|{self.tag}")[:12], 16)
        r = self.policy(messages, tools, tool_choice, random.Random(h))
        r.usage = Usage(
            prompt_tokens=estimate_tokens(messages) + (estimate_tokens(tools) if tools else 0),
            completion_tokens=estimate_tokens(r.content or "") + sum(estimate_tokens(t.arguments) for t in r.tool_calls),
        )
        r.model = self.model
        if self.price:
            r.cost_usd = self.price.cost(r.usage)
        self.n_calls += 1
        return r


# --------------------------------------------------------------------------- #
# Budget guard
# --------------------------------------------------------------------------- #
class BudgetGuard:
    """Hard spend cap across runs, persisted in a ledger file.

    Before each call it estimates the worst case (estimated input + ``max_tokens`` output,
    priced at ``price``) and raises :class:`BudgetExceeded` if that could cross the cap.
    After each call it adds the provider's actual cost to the ledger.
    """

    def __init__(self, inner: LLMClient, max_usd: float, price: Price, ledger: Optional[Path] = None,
                 default_max_tokens: int = 4000):
        self.inner = inner
        self.model = inner.model
        self.max_usd = float(max_usd)
        self.price = price
        self.ledger = Path(ledger) if ledger else None
        self.default_max_tokens = default_max_tokens
        self.spent_usd = 0.0
        self.calls = 0
        if self.ledger and self.ledger.exists():
            d = json.loads(self.ledger.read_text())
            self.spent_usd, self.calls = float(d.get("total_usd", 0.0)), int(d.get("calls", 0))

    def worst_case_usd(self, messages, tools, max_tokens: int) -> float:
        u = Usage(prompt_tokens=estimate_tokens(messages) + (estimate_tokens(tools) if tools else 0),
                  completion_tokens=max_tokens)
        return Price(self.price.input, self.price.output).cost(u)  # no cache discount: worst case

    def chat(self, messages, tools=None, tool_choice=None, **params) -> LLMResponse:
        worst = self.worst_case_usd(messages, tools, int(params.get("max_tokens", self.default_max_tokens)))
        if self.spent_usd + worst > self.max_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f}; next call could cost up to ${worst:.4f}; cap ${self.max_usd:.4f}"
            )
        r = self.inner.chat(messages, tools, tool_choice, **params)
        self.spent_usd += r.cost_usd
        self.calls += 1
        if self.ledger:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            self.ledger.write_text(json.dumps(
                {"total_usd": round(self.spent_usd, 6), "calls": self.calls, "cap_usd": self.max_usd}, indent=1))
        return r


# --------------------------------------------------------------------------- #
# Cache / replay
# --------------------------------------------------------------------------- #
class CachedClient:
    """Request/response cache in ``<cache_dir>/<model>.jsonl``.

    ``mode="live"``: serve hits from the cache, call the inner client on a miss and append.
    ``mode="replay"``: cache only; a miss raises :class:`CacheMiss`. No inner client needed.
    Every call (hit or miss) is appended to ``self.log`` for token / cost accounting.
    """

    def __init__(self, inner: Optional[LLMClient], model: str, mode: str = "replay",
                 cache_dir: Path | str | None = DEFAULT_CACHE_DIR, store_requests: bool = False):
        if mode not in ("live", "replay"):
            raise ValueError("mode must be 'live' or 'replay'")
        if mode == "live" and inner is None:
            raise ValueError("live mode needs an inner client")
        self.inner, self.model, self.mode = inner, model, mode
        # cache_dir=None: in-memory only (used for mock dry runs)
        self.path = None if cache_dir is None else Path(cache_dir) / f"{model_slug(model)}.jsonl"
        self.store_requests = store_requests
        self.cache_tag = ""
        self.log: list[dict[str, Any]] = []
        self._index: dict[str, dict] = {}
        if self.path is not None and self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self._index[rec["key"]] = rec["response"]

    def __len__(self) -> int:
        return len(self._index)

    def chat(self, messages, tools=None, tool_choice=None, **params) -> LLMResponse:
        key_params = {**params, "tool_choice": tool_choice}
        key = request_key(self.model, messages, tools, key_params, self.cache_tag)
        if key in self._index:
            r = LLMResponse.from_dict(self._index[key], from_cache=True)
            billed = 0.0
        elif self.mode == "replay":
            raise CacheMiss(f"no cached response for request {key[:12]} (model {self.model}, tag {self.cache_tag!r})")
        else:
            if hasattr(self.inner, "tag"):
                self.inner.tag = self.cache_tag
            r = self.inner.chat(messages, tools, tool_choice, **params)
            billed = 0.0 if getattr(self.inner, "notional", False) else r.cost_usd
            rec: dict[str, Any] = {"key": key, "model": self.model, "tag": self.cache_tag, "params": key_params,
                                   "response": r.to_dict(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if self.store_requests:
                rec["request"] = {"messages": messages, "tools": tools}
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._index[key] = rec["response"]
        self.log.append({"key": key, "tag": self.cache_tag, "model": r.model, "from_cache": r.from_cache,
                         "usage": asdict(r.usage),
                         "cost_usd": r.cost_usd, "billed_usd": billed, "latency_s": r.latency_s})
        return r


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def price_for(model: str, llm_cfg: dict[str, Any]) -> Price:
    p = (llm_cfg.get("prices") or {}).get(model)
    if not p:
        raise LLMError(f"no price for {model!r} in config llm.prices (needed for the budget guard)")
    return Price(float(p["input"]), float(p["output"]), p.get("cache_read"))


def make_client(
    llm_cfg: dict[str, Any],
    mode: Optional[str] = None,
    model: Optional[str] = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    mock_policy: Optional[MockPolicy] = None,
    max_spend_inr: Optional[float] = None,
) -> CachedClient:
    """Build the client stack from config + environment (``LLM_MODE``, ``LLM_MODEL``, keys).

    ``mode``: ``replay`` (default, no keys, no spend), ``live`` (real calls, capped), or
    ``mock`` (offline policy, in-memory cache only, nothing billed).
    ``max_spend_inr`` adds a tighter cap for this process on top of the global ledger cap.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    mode = mode or os.environ.get("LLM_MODE", "replay")
    model = model or os.environ.get("LLM_MODEL") or llm_cfg["models"]["cheap"]
    if mode == "mock":
        if mock_policy is None:
            from .llm_session import mock_policy as default_policy

            mock_policy = default_policy
        price = price_for(llm_cfg["models"]["cheap"], llm_cfg)
        inner = MockClient(mock_policy, model="mock", price=price)
        return CachedClient(inner, "mock", "live", cache_dir=None)
    if mode == "replay":
        return CachedClient(None, model, "replay", cache_dir)
    if mode != "live":
        raise ValueError(f"unknown LLM mode {mode!r}")

    price = price_for(model, llm_cfg)
    provider = os.environ.get("LLM_PROVIDER", llm_cfg.get("provider", "openrouter"))
    if provider == "anthropic":
        inner: LLMClient = AnthropicClient(model, price=price)
    else:
        inner = OpenAICompatClient(model, base_url=os.environ.get("OPENAI_BASE_URL", llm_cfg.get("base_url")),
                                   price=price, reasoning_effort=llm_cfg.get("reasoning_effort"))
    budget = llm_cfg.get("budget", {})
    cap_usd = float(budget.get("max_spend_inr", 200)) / float(budget.get("inr_per_usd", 95))
    guard = BudgetGuard(inner, cap_usd, price, Path(cache_dir) / "spend_ledger.json",
                        default_max_tokens=int(llm_cfg.get("max_tokens_per_call", 12000)))
    if max_spend_inr is not None:
        guard.max_usd = min(guard.max_usd, guard.spent_usd + float(max_spend_inr) / float(budget.get("inr_per_usd", 95)))
    return CachedClient(guard, model, "live", cache_dir)
