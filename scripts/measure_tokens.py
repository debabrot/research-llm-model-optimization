"""Phase 3 gate: measure real token usage on a handful of sessions, then project the full grid.

    python scripts/measure_tokens.py --dry            # offline: mock structure + assumptions, no key
    python scripts/measure_tokens.py                  # LIVE: ~32 calls, about ₹3-6, hard-capped at ₹15

Live mode runs 4 real LLM runs (8 sessions) that are also real grid cells:

    llm_select + llm_direct  x  (delivery seed 0 named run 0, california seed 0 anon run 0)

Every call goes through the normal cache (llm_cache/), so these calls are reused for free
when the full grid runs later. The spend ledger (llm_cache/spend_ledger.json) counts them
against the overall ₹200 cap. The script prints the exact model id the provider served:
pin it in configs/experiment.yaml (llm.models.cheap) before the full run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tree_surgery.arms import build_context, get_arm  # noqa: E402
from tree_surgery.config import load_config  # noqa: E402
from tree_surgery.datasets import load_dataset  # noqa: E402
from tree_surgery.llm_client import make_client, price_for  # noqa: E402
from tree_surgery.train import fit_base_model  # noqa: E402

CELLS = [("delivery", 0, "named", 0), ("california", 0, "anon", 0)]
APPROACHES = ("llm_select", "llm_direct")


def grid_sessions(cfg) -> int:
    sess = cfg["llm"]["session"]
    rounds = -(-cfg["edit_budget"]["max_candidates"] // sess["candidates_per_session"])
    return len(APPROACHES) * 2 * len(cfg["datasets"]) * len(cfg["seeds"]) * cfg["llm_runs_per_cell"] * rounds


def project(per_session_usd: dict[str, float], cfg, label: str) -> float:
    n = grid_sessions(cfg) // len(APPROACHES)
    b = cfg["llm"]["budget"]
    total = sum(v * n for v in per_session_usd.values())
    inr = total * b["inr_per_usd"]
    print(f"\n{label}: {grid_sessions(cfg)} sessions in the full grid")
    for a, v in per_session_usd.items():
        print(f"  {a:11s} ${v:.5f}/session x {n} = ${v * n:.3f}")
    print(f"  TOTAL ${total:.2f} = ₹{inr:.0f}   (cap ₹{b['max_spend_inr']})  "
          f"{'OK' if inr <= b['max_spend_inr'] else 'OVER CAP: reduce runs or rounds'}")
    return inr


def assumption_projection(cfg) -> None:
    """Per-session tokens from measured prompt sizes + assumed reasoning/answer lengths."""
    p = price_for(cfg["llm"]["models"]["cheap"], cfg["llm"])
    q = cfg["llm"]["session"]["max_queries"]
    prefix = 650 + 300 + 250          # static block + tool schema + history (measured, round average)
    result = 1100 + 60                # one 30-row query result + the call itself (measured, both datasets)
    tin = (q + 1) * prefix + result * q * (q + 1) / 2
    for name, reasoning in (("low reasoning (~700/turn)", 700), ("heavier reasoning (~2000/turn)", 2000)):
        tout = q * (reasoning + 60) + reasoning + 1500   # + final JSON answer
        per = (tin * p.input + tout * p.output) / 1e6     # ignores prompt-cache discount (conservative)
        print(f"\nAssumed session: {tin / 1e3:.1f}K in, {tout / 1e3:.1f}K out ({name})")
        project({a: per for a in APPROACHES}, cfg, "Projection")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="offline: mock client, no key, no spend")
    ap.add_argument("--max-inr", type=float, default=15.0, help="hard cap for THIS script's spend (₹)")
    args = ap.parse_args()
    cfg = load_config()

    if args.dry:
        assumption_projection(cfg)
        client = make_client(cfg["llm"], mode="mock")
    else:
        client = make_client(cfg["llm"], mode="live", max_spend_inr=args.max_inr)
        print(f"LIVE: model {client.model}; this script is capped at ₹{args.max_inr:.0f}")

    per: dict[str, list[float]] = {a: [] for a in APPROACHES}
    served = set()
    for ds, seed, variant, run in CELLS:
        data = load_dataset(ds, seed=seed)
        booster, _ = fit_base_model(data, cfg["base_model"])
        ctx = build_context(data, booster, cfg)
        for arm in APPROACHES:
            log0 = len(client.log)
            out = get_arm(arm, client, variant, run).run(ctx)
            i = out.info
            for s in i["sessions"]:
                u = s["usage"]
                print(f"{ds:10s} {variant:5s} {arm:10s} round {s['round']}: {s['n_calls']} calls, "
                      f"{s['n_queries']} queries, in {u['prompt_tokens']:,} (cached {u['cached_tokens']:,}), "
                      f"out {u['completion_tokens']:,} (reasoning {u['reasoning_tokens']:,}), ${s['cost_usd']:.5f}"
                      f"{'' if s['ok'] else '  FAILED: ' + str(s['error'])}")
                per[arm].append(s["cost_usd"])
            served |= {e.get("model") for e in client.log[log0:] if e.get("model")}
            print(f"  -> {len(out.candidates)} candidates, {i['llm_cache_hits']}/{i['llm_calls']} from cache, "
                  f"billed now ${i['llm_billed_usd']:.5f}")

    project({a: float(np.mean(v)) for a, v in per.items()}, cfg,
            "Projection from these sessions" + (" (mock: no reasoning tokens, a floor)" if args.dry else ""))
    if not args.dry:
        print(f"\nModel id(s) served: {sorted(served) or 'not reported'}  -> pin in configs/experiment.yaml")
        print(f"Ledger total so far: ₹{client.inner.spent_usd * cfg['llm']['budget']['inr_per_usd']:.2f}"
              if hasattr(client.inner, "spent_usd") else "")


if __name__ == "__main__":
    main()
