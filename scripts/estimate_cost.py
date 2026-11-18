"""Back-of-envelope LLM cost model for the query-tool design (PLAN §5, §7).

A *session* is one short agent loop: the LLM reads the static prompt, makes a few
`query_leaves` tool calls (each returns <= `rows_per_query` leaf lines), then
proposes `cands_per_session` candidates. Each tool turn re-sends the conversation,
so input grows turn by turn; with prompt caching the repeated prefix is billed at
the cache-read rate and only new tokens at the (cache-write) input rate.

Everything here is an assumption until Phase 3 measures real token counts.
Usage: python scripts/estimate_cost.py
"""

from dataclasses import dataclass

INR_PER_USD = 95.0  # mid-market, Nov 2026


@dataclass
class Price:  # USD per 1M tokens
    inp: float
    out: float
    cache_read: float
    cache_write: float


PRICES = {
    # OpenRouter list prices, Nov 2026. Flash varies by provider; these are conservative.
    "deepseek-v4.1-flash": Price(inp=0.10, out=0.60, cache_read=0.03, cache_write=0.10),
    "claude-opus-5.5": Price(inp=4.00, out=20.00, cache_read=0.20, cache_write=5.00),
}


@dataclass
class Session:
    static_tokens: int = 4_000        # task, constraints, dataset + model summary, tool schemas
    history_tokens: int = 1_500       # previous candidates + their calibration RMSE (avg)
    queries: int = 4                  # tool calls per session
    rows_per_query: int = 50
    tokens_per_row: int = 42          # measured: 169K tokens / 4,005 leaf lines (chars/4)
    call_tokens: int = 150            # the tool-call JSON the model emits
    out_per_turn: int = 300           # visible output per exploration turn
    reasoning_per_turn: int = 1_000   # thinking tokens per turn (billed as output)
    final_out: int = 1_500            # final answer (candidates + reasons)

    def tokens(self):
        base = self.static_tokens + self.history_tokens
        step = self.rows_per_query * self.tokens_per_row + self.call_tokens
        turns = self.queries + 1
        total_in = sum(base + k * step for k in range(turns))
        new_in = base + self.queries * step            # each token is new once
        cached_in = total_in - new_in
        out = self.queries * (self.out_per_turn + self.reasoning_per_turn) + self.final_out + self.reasoning_per_turn
        return total_in, new_in, cached_in, out

    def cost(self, p: Price, caching: bool = True) -> float:
        total_in, new_in, cached_in, out = self.tokens()
        if caching:
            inp = new_in * p.cache_write + cached_in * p.cache_read
        else:
            inp = total_in * p.inp
        return (inp + out * p.out) / 1e6


def grid_sessions(approaches=2, variants=2, datasets=2, seeds=5, runs=1, candidates=20, cands_per_session=1):
    return approaches * variants * datasets * seeds * runs * (candidates // cands_per_session)


def show(label, n_sessions, model, sess, caching=True):
    c = n_sessions * sess.cost(PRICES[model], caching)
    print(f"  {label:58s} {n_sessions:5d} sessions  ${c:8.2f}  ₹{c * INR_PER_USD:9.0f}")
    return c


if __name__ == "__main__":
    low = Session(reasoning_per_turn=500, queries=3)
    mid = Session()
    high = Session(reasoning_per_turn=3_000, queries=6)

    for name, s in [("low", low), ("mid", mid), ("high", high)]:
        t = s.tokens()
        print(f"{name:5s} session: {t[0]/1e3:.0f}K input ({t[2]/1e3:.0f}K cache-readable), {t[3]/1e3:.1f}K output")
        for m in PRICES:
            print(f"      {m:22s} ${s.cost(PRICES[m]):.4f} cached   ${s.cost(PRICES[m], caching=False):.4f} uncached")

    print("\nPlan as written (1 candidate per session):")
    for runs in (1, 3):
        n = grid_sessions(runs=runs)
        for m in PRICES:
            show(f"{m}, full grid, {runs} run(s), mid", n, m, mid)
