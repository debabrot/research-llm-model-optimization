# Tree Surgery

**Can an LLM in the optimization loop beat standard optimization when editing a trained model's parameters?**

We take a trained LightGBM regressor, freeze its tree structure, and let different methods edit its leaf values: an LLM, sparse regression, random selection, plus full refits as ceilings. Every method gets the same labelled calibration data and the same edit budget (≤ 10 leaves, ≤ 10 candidates). The full research plan, with every decision logged, is in [PLAN.md](PLAN.md).

| Phase | What | Status |
|---|---|---|
| 1 | Foundation: data, base model, leaf inspector, patcher, design matrix, tests, notebook 01 | ✅ done |
| 2 | Non-LLM arms (`original`, `random_opt`, `sparse_opt`, `refit`, `retrain`) + runner | ✅ done (gate results below) |
| 3 | LLM client (OpenRouter, cache/replay, budget guard), query tool, prompts, LLM arms (mock dry run) | ✅ done, except the real-call token check (needs your key) |
| 4 | LLM arms on DeepSeek V4.1 Flash | next |
| 5 | Corruption experiment, results notebook, blog post | – |

No LLM results yet. Phase 2 gate results (non-LLM arms) are [below](#phase-2-gate-results-non-llm-arms).

---

## Try it out

### Option 1: Colab (nothing to install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/debabrot/research-llm-model-optimization/blob/main/notebooks/01_leaf_surgery_101.ipynb)

Run all cells. The first cell installs LightGBM and clones this repo.

### Option 2: Locally

Requires Python ≥ 3.10.

```bash
git clone https://github.com/debabrot/research-llm-model-optimization.git
cd research-llm-model-optimization

# with uv (recommended)
uv sync --extra dev
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# or with plain pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest jupyter
```

Then:

```bash
# 1. Run the tests (~1 min). California Housing downloads once (~400 KB).
pytest

# 2. See the whole Phase 1 pipeline in one command (~2 s)
python scripts/quickstart.py
python scripts/quickstart.py --dataset california --seed 2

# 3. Walk through it interactively
jupyter lab notebooks/01_leaf_surgery_101.ipynb

# 4. Run the non-LLM experiment grid: 2 datasets x 5 seeds x 5 arms (~1 min)
python -m tree_surgery.runner                     # writes results/phase2/{results.csv,candidates.jsonl,summary.md}
python -m tree_surgery.runner --datasets delivery --seeds 0 1 --arms original sparse_opt

# 5. Dry-run the LLM arms with an offline mock (no key, no spend; ~3 min for the full grid)
python -m tree_surgery.runner --arms all --llm mock --datasets delivery --seeds 0
python scripts/measure_tokens.py --dry            # token/cost projection for the full grid

# 6. Phase 3 gate, real calls (needs .env, about ₹3-6, hard-capped at ₹15)
python scripts/measure_tokens.py
```

Expected output from `quickstart.py` (D1, seed 0):

```
   267 trees, 4005 leaves; test RMSE 4.239, vs truth 1.469
   ...
3. Editing T129/L7 by +1 ...
   1202 test rows in the leaf moved by 1.000000000 .. 1.000000000
   2798 other rows moved by at most 0.0e+00
4. Linearity check: max |y_base + X·d - edited prediction| = 5.0e-14
5. Rolled back; predictions identical to base: True
```

### Use the package

```python
import tree_surgery as ts

data = ts.load_dataset("delivery", seed=0)            # train / cal / val / test splits
booster, info = ts.fit_base_model(data)                # LightGBM, early-stopped on val

patcher = ts.ModelPatcher(booster)                     # frozen snapshot of the base model
table = ts.leaf_table(booster, data["cal"].X, data["cal"].y)   # one row per leaf
print("\n".join(ts.render_leaf_table(table.head())))    # what the LLM will be able to query

cand = patcher.candidate()                             # independent copy
cand.apply({(129, 7): +1.0}, max_leaves=10)            # edit leaf 7 of tree 129
cand.predict(data["val"].X)
cand.rollback()                                        # exact undo
```

---

## API keys: only for live LLM runs

Phases 1–2, the mock (`--llm mock`) and replay (`--llm replay`, the default) make **no paid calls** and need no key. Live runs read the OpenRouter settings in [`configs/experiment.yaml`](configs/experiment.yaml) (model IDs, per-session limits, prices, a ₹200 spend cap). Install the LLM extras (`uv sync --extra llm` or `pip install -e ".[llm]"`), then:

1. In OpenRouter → Settings → Keys, create a **new key just for this project** with a **credit limit of $2.10 (≈ ₹200)**. OpenRouter rejects calls past that limit, whatever the code does.
2. `cp .env.example .env` and paste the key into `OPENAI_API_KEY`. `.env` is git-ignored; never commit it or paste it into a chat.
3. In Colab, use the 🔑 Secrets panel with the same variable names instead of a `.env` file.

The client also refuses any call whose worst case (estimated input + `max_tokens` output) could cross the cap, keeping a running total in `llm_cache/spend_ledger.json` across runs. Every response is cached in `llm_cache/<model>.jsonl`, so a re-run, or a run that was interrupted and resumed, only pays for new calls. `--llm replay` reads only the cache and fails on a miss.

**Budget plan (Option A, ≈ ₹100–185):** DeepSeek V4.1 Flash only, full grid (both approaches, named + anon, both datasets, 5 seeds, 3 runs). Each run has 10 candidates in 2 rounds of 5, and each session allows ≤ 3 leaf queries of 30 rows. Claude Opus 5.5 is configured but out of this budget. See the PLAN.md changelog.

---

## What's in the package

| Module | Does |
|---|---|
| `tree_surgery/datasets.py` | D1 delivery-time generator (known truth `f_true`), D2 California Housing, seeded 50/15/15/20 splits, anon feature mapping |
| `tree_surgery/train.py` | Base LightGBM model (early stopping on val, truncated to best iteration), save/load |
| `tree_surgery/inspector.py` | Decision path → readable rule for every leaf, calibration residual stats, leaf-line rendering |
| `tree_surgery/patcher.py` | Independent edit candidates, atomic validated edits, exact rollback, save/load |
| `tree_surgery/design_matrix.py` | Sparse one-hot leaf matrix: `predict(X) == X_leaf @ leaf_values` |
| `tree_surgery/optimizers.py` | *(Phase 2)* Closed-form ridge on a leaf subset (λ on validation), OMP and Lasso-path leaf selection |
| `tree_surgery/arms.py` | *(Phase 2)* All arms behind one interface; `EditArm` enforces the budget, selects on validation, verifies `ŷ + X·d` |
| `tree_surgery/evaluation.py` | Metrics, paired differences, bootstrap CI, Wilcoxon |
| `tree_surgery/runner.py` | *(Phase 2)* Experiment grid from config → `results/<phase>/`; `--llm mock\|replay\|live` for LLM arms |
| `tree_surgery/llm_client.py` | *(Phase 3)* OpenRouter (OpenAI-compatible) and Anthropic clients, cache/replay keyed on (model, messages, params, run), budget guard, mock |
| `tree_surgery/prompts.py` | *(Phase 3)* Static prompt, history block, `query_leaves` tool, Pydantic output schemas, named/anon leaf views |
| `tree_surgery/llm_session.py` | *(Phase 3)* One session: ≤ 3 queries → JSON proposal, one retry on invalid output; scripted mock policy |

**Guarantees covered by tests:**
- Editing leaf (t, l) by δ moves exactly the samples in that leaf, by exactly δ.
- `ŷ_base + X·d` matches the edited model to within 1e-9.
- Candidates are fully independent.
- Save/load and rollback are bit-exact.
- Every rendered rule selects exactly the rows LightGBM routes to that leaf, including missing-value handling.
- *(Phase 3)* The LLM conversation is identical when validation and test labels change (feedback is calibration-only); `anon` prompts contain no real feature names, descriptions or target name, and its instructions match `named` word for word; the static prompt lists no leaves; replay reproduces a live run exactly, and a run index outside the cache fails rather than reusing another run; the budget guard refuses before calling.
- *(Phase 2)* Edit arms never exceed 10 leaves / 10 candidates (violations raise); arm outcomes don't change when test labels are replaced with NaN; the chosen candidate is the validation optimum; the patched model equals the linear prediction; OMP recovers planted leaf edits (up to leaves that are identical on calibration).

## Phase 2 gate results (non-LLM arms)

Test RMSE, mean over 5 seeds. Δ is the paired difference vs `original` with a 95% bootstrap CI (negative = better). Full tables: [`results/phase2/summary.md`](results/phase2/summary.md).

| Arm | Budget | D1 test RMSE | D1 vs truth | Δ vs original (D1) | D2 test RMSE | Δ vs original (D2) |
|---|---|---|---|---|---|---|
| `original` | – | 4.2917 | 1.4197 | – | 0.4586 | – |
| `random_opt` | ≤ 10 leaves | 4.2953 | 1.4262 | +0.09% (n.s.) | 0.4588 | +0.03% (n.s.) |
| `sparse_opt` | ≤ 10 leaves | 4.2988 | 1.4385 | **+0.16%, worse on 5/5 seeds** | 0.4585 | −0.02% (n.s.) |
| `refit` | all leaves | 4.2907 | 1.4227 | −0.02% (n.s.) | 0.4583 | −0.08% |
| `retrain` | new model | 4.2893 | 1.3664 | −0.06% (n.s.) | 0.4502 | **−1.84%, better on 5/5** |

**What this says.** The early-stopped base model leaves almost nothing for a 10-leaf edit to fix. Edits fitted on calibration always lower calibration RMSE, and validation picks the candidate that looks best, but test RMSE doesn't follow. On D1 the main numerical competitor is slightly *worse* than doing nothing. Only retraining, which also learns new structure from the extra data, clearly helps, and only on California Housing. This is the bar the LLM arms will be measured against in Phase 4.

## Phase 3: how the LLM arms work

One **run** = 10 candidates in 2 **sessions** of 5. In each session the LLM:

1. reads a fixed **static prompt**: the task, the dataset (`named`: names, units, descriptions; `anon`: `f0..fN` with ranges only), a model summary with counts only, the leaf-line format and the answer schema;
2. reads the **history**: previous candidates and their *calibration* RMSE (validation and test are never shown);
3. calls `query_leaves` up to 3 times (filter by features, rule text, tree, n_cal, residual; sort by any column; ≤ 30 rows). This is its only view of the leaves: nothing is pre-ranked for it;
4. answers with JSON, validated with Pydantic (leaves exist, ≤ 10 per candidate, no duplicates, |δ| ≤ 0.5·std(y) for `llm_direct`). An invalid answer gets one retry with the error; a second failure counts as 5 failed candidates.

`llm_select` then goes through exactly the same path as `sparse_opt` (ridge on calibration, λ and the reported candidate chosen on validation). `llm_direct` applies the LLM's own deltas.

**Gate status.** A mock dry run of the full grid (120 LLM runs, 240 sessions, 720 calls) ran with no failures ([`results/phase3_mock_dryrun/`](results/phase3_mock_dryrun/summary.md)). Measured prompt sizes project the real grid at **≈ ₹90–160 on DeepSeek Flash** (`python scripts/measure_tokens.py --dry`). The last step needs your key: `python scripts/measure_tokens.py` makes ~32 real calls (≈ ₹3–6, capped at ₹15), prints real token counts, the projected grid cost and the exact model id to pin. Those calls are cached and reused by the real run.

## D1: the delivery-time generator

Expected delivery time in minutes (never shown to any LLM):

```
f = 8 + 2.5·distance_km + 0.6·prep_time_min + 1.2·traffic_index + 0.15·rain_mm
    + 0.06·rain_mm·traffic_index          # interaction
    + 0.9·distance_km·[rush hour]          # interaction (8–10h, 17–20h)
    − 6·(1 − exp(−driver_experience_yrs/3))
    + 0.5·order_size + 5·[order_size > 5]  # threshold
    − 1.5·is_weekend
y = f + N(0, σ²),  σ = 2.47·(0.5 + distance_km/5)   # heteroscedastic, SNR ≈ 3
```

## Layout

```
PLAN.md                   research plan + changelog of every decision
configs/experiment.yaml   all fixed settings (incl. LLM budget)
tree_surgery/             the package
scripts/                  quickstart.py, measure_tokens.py (Phase 3 gate), estimate_cost.py (pre-Phase 3 model)
notebooks/01_…            leaf surgery walkthrough (Phase 1)
tests/                    pytest suite
results/phase2/           non-LLM gate run (committed)
results/phase3_mock_dryrun/  mock dry run of the full grid (machinery check, not a result)
llm_cache/                real LLM responses + spend ledger, from the Phase 3 gate onward (committed)
```
