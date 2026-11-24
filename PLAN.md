# Tree Surgery — Research Plan

**Can an LLM in the optimization loop beat standard optimization when editing a trained model's parameters?**

Status: final plan, v1 (2026-11-17). Changes after experiments start must be logged in the changelog at the bottom.

---

## 1. Question

Take a trained LightGBM model and freeze its tree structure. Can an LLM improve held-out performance by editing leaf values, beyond what a standard numerical optimizer achieves with the **same data** and the **same edit budget**?

The LLM is not an alternative to supervised learning. It still learns from labelled calibration residuals. What's being tested is whether **LLM judgment adds value on top of numerical optimization when both use the same supervised signal**.

### Hypotheses

- **H1 (alternative).** Each leaf is a readable rule (`income > 5.2 AND age <= 20`), and the LLM knows what the features mean. That may let it choose better leaves to edit than a purely numerical method, especially where calibration data is sparse or noisy.
- **H0 (null).** The LLM adds nothing over sparse regression on leaf indicators. With fixed structure and squared loss, a model's prediction is linear in its leaf values, and sparse regression already handles the "which k leaves" question well.

### Why the problem is linear (and why that matters)

For regression with squared loss:

```
ŷ(x) = Σ_t  leaf_value[t, leaf_t(x)]
```

Fix the structure, and `predict(pred_leaf=True)` gives each sample's leaf in every tree. One-hot encoding this gives a sparse design matrix `X` (n × L, where L ≈ 500–1,500 leaves). An edit is a delta vector `d`, and new predictions are `ŷ + X·d`. As a result:

- Choosing values for a given leaf subset is ridge regression, which has a closed-form solution.
- Choosing *which* k leaves is sparse regression (Lasso / OMP / greedy forward selection).
- Approach A (the LLM picks values) is roughly asking whether the LLM can approximate shrunk per-leaf mean residuals.

The non-LLM competitor is therefore strong, and the experiment is designed to give the LLM a fair chance to beat it.

---

## 2. Datasets

| # | Dataset | Rows | Why |
|---|---|---|---|
| D1 | **Semi-synthetic "delivery time"** generator (meaningful feature names: `distance_km`, `rain_mm`, `hour`, `driver_experience_yrs`, `traffic_index`, `order_size`, …) | ~20,000 | Domain-meaningful names, but a new formula the LLM can't have memorized. The true function is known, so we can measure whether edits move the model toward the truth. |
| D2 | **California Housing** (`sklearn.datasets.fetch_california_housing`) | 20,640 | Real data with named features. Probably in LLM pretraining data; the anonymization ablation controls for this. |

**D1 generator spec.** Fixed formula with main effects, at least two interactions (e.g. `rain × traffic`, `hour × distance`), one threshold nonlinearity, heteroscedastic Gaussian noise (SNR ≈ 3). Generated deterministically from a seed. The formula lives in code and in the README, never in any prompt.

**Splits.** Random 50 / 15 / 15 / 20: train / calibration / validation / test.
- **Train:** fits the base model.
- **Calibration:** the only labelled data any edit method may fit on.
- **Validation:** used for candidate selection and the ridge λ choice.
- **Test:** the final evaluation. Never visible to the LLM, any optimizer, or any selection step.

**Seeds.** 5 seeds (0–4). Each seed changes the split and the base model's seed, and for D1 also the data draw.

**Out of scope for v1:** classification, time-series or distribution-shift splits, additional datasets.

---

## 3. Base model

- LightGBM regression, `num_leaves=15`, `learning_rate=0.05`, up to **5,000** rounds (raised from 1,000 on 2026-11-23, see changelog) with **early stopping on validation** (patience 50). Other hyperparameters at their defaults, fixed in `configs/experiment.yaml`.
- Trained on **train only**.
- The base model, its predictions and its metrics are saved per (dataset, seed).

A model that is deliberately weak inflates any editing gain. Early stopping keeps the baseline honest.

---

## 4. Arms

All **edit arms** share these constraints: ≤ **10 edited leaves** per candidate and ≤ **20 candidate evaluations** per run. Each candidate starts from a fresh copy of the base model.

| Arm | Picks which leaves | Picks the values | Role |
|---|---|---|---|
| `original` | – | – | Floor |
| `random_opt` | Random 10 | Ridge on calibration | Sanity check |
| `sparse_opt` | OMP / Lasso path on `X_cal`, k = 10 | Ridge on calibration | **Main competitor** |
| `llm_select` (Approach B) | LLM | Ridge on calibration | **Main test** |
| `llm_direct` (Approach A) | LLM | LLM (bounded) | Can the LLM do the numbers too? |
| `refit` | All leaves | `Booster.refit(X_cal, y_cal, decay_rate)`; decay chosen on validation | Ceiling with structure fixed, **no sparsity budget** |
| `retrain` | – | Retrain from scratch on train + calibration, same hyperparameters + early stopping | "Just do the normal thing" ceiling, **no budget** |

`refit` and `retrain` do not satisfy the edit budget. They are reported as reference ceilings, never as competitors.

**Candidate selection (all edit arms).** Each arm produces up to 20 candidates. The one with the best **validation** RMSE is reported on **test**.

### Ablation

`llm_select` and `llm_direct` are each run with:
- **named:** real feature names
- **anon:** features renamed `f0…fN`, with the column order shuffled per seed

If `anon` ≈ `named`, the LLM isn't using semantics and behaves like a slower heuristic.

---

## 5. What the LLM sees

> **Revised 2026-11-17 (see changelog):** the full leaf table is too large to send on every call. The LLM now reaches every leaf through a query tool instead of reading one dump. The no-pre-filtering principle below still holds: the LLM chooses its own filters.

**Decision: the LLM sees all leaves (no pre-filtering).** A residual-based pre-filter would itself be a heuristic. The LLM would then only be choosing among leaves the heuristic already picked, which would confound the main comparison.

### Prompt structure

1. **Static block** (prompt-cached, identical across calls within a dataset/seed/variant):
   - Task description and constraints (k ≤ 10, delta bounds, JSON schema).
   - Dataset description: feature names, units and ranges, plus the target name and units. In the `anon` variant, only `f0…fN` with ranges.
   - Base model summary: number of trees and leaves, plus train and calibration RMSE.
   - **Leaf table**, one line per leaf:
     `T{tree}/L{leaf} | rule: distance_km > 12.3 AND rain_mm <= 4.0 | n_cal=143 | value=0.021 | mean_resid=+0.84 | std_resid=1.9`
     The rule is the full decision path, including missing-value direction. Numbers are rounded to 3 significant figures.
2. **Dynamic block:** history of previous candidates in this run (leaves chosen, deltas if Approach A, **calibration** RMSE). Validation and test scores are **never** fed back.

### Output schema (Pydantic-validated)

- **Approach B:** `{"leaves": [{"tree_id": int, "leaf_id": int, "reason": str}], "rationale": str}`. At most 10 entries, all indices must exist, no duplicates.
- **Approach A:** the same, plus `"delta": float` per leaf, with |delta| ≤ 0.5 × std(y_train).

If output is invalid, retry once with the validation error; if it is still invalid, count the call as a failed candidate. Failed candidates count toward the 20-call budget.

---

## 6. Fixed settings

| Setting | Value |
|---|---|
| Max edited leaves per candidate | 10 |
| Max candidates per run | 20 |
| Ridge λ grid (all optimizer arms) | λ ∈ {1e-3, 1e-2, 1e-1, 1, 10, 100} on the mean-squared-error objective, chosen on validation |
| Approach A delta bound | \|δ\| ≤ 0.5 × std(y_train) |
| Feedback to the LLM | Calibration metrics only |
| LLM runs per (dataset, seed, variant) | 3 (adds LLM sampling variance) |
| LLM temperature / reasoning effort | Fixed per model, recorded in config |
| `sparse_opt` candidates | 20 points along the OMP/Lasso path, each truncated to ≤ 10 leaves |
| `random_opt` candidates | 20 independent random draws of 10 leaves |

---

## 7. LLMs

| Tier | Model | Role |
|---|---|---|
| Cheap | DeepSeek Flash (pin the exact API model ID at run time; V4 Flash may have been superseded) | Full grid first; used to measure real token usage and cost |
| Strong | Claude Opus 5.5 (Fable 5.2 optional if budget allows) | Full grid, same prompts |

**Run both tiers in full, whatever the cheap tier shows.** A null result from a cheap model does not imply a null from a strong one. "Cheap ≈ sparse_opt, strong > sparse_opt" would be a meaningful finding.

**Budget estimate (before caching):** about 800 calls per model for 1 LLM run per seed (2 approaches × 20 calls × 2 variants × 5 seeds × 2 datasets). That's roughly $3 on DeepSeek Flash and roughly $110 on Opus 5.5. The 3 runs per seed would triple this. Measure on Flash first, then confirm the Opus budget. Thinking tokens bill as output.

**Client.** A provider-agnostic `LLMClient` interface, with an OpenAI-compatible implementation (DeepSeek and others) and an Anthropic adapter. It is configured through environment variables (`LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). It logs tokens, cost and latency on every call.

**Caching and replay.** Every request and response is stored in `llm_cache/*.jsonl`, keyed by a hash of (model, messages, params). Replay mode reads only from the cache and fails on a miss. This makes the published results reproducible without API keys.

---

## 8. Metrics and analysis

**Primary metric:** test RMSE.

**Secondary metrics:**
- Test MAE and R².
- Calibration and validation RMSE, and the generalization gap (test − calibration).
- Leaves edited and the fraction of parameters changed.
- LLM tokens, cost (USD) and runtime.
- **D1 only:** RMSE against the true noise-free function. This tells us whether edits move the model toward the truth.

**Analysis.**
- Paired differences per (dataset, seed), e.g. `llm_select − sparse_opt`. Report the mean and a 95% bootstrap CI over seeds × LLM runs, plus a Wilcoxon signed-rank test.
- Report every arm and every model, including negative results.

### Decision rule (pre-registered)

- **LLM adds value:** `llm_select` beats `sparse_opt` on test RMSE, with the 95% CI of the paired difference excluding 0, on **both** datasets, for a given LLM.
- **LLM is practically useful:** additionally, `llm_select` ≥ `retrain` (no worse within its CI).
- **Semantics matter:** `named` beats `anon` with a CI excluding 0.
- Anything else is reported as a negative or inconclusive result.

---

## 9. Secondary experiment: corruption repair (reported separately)

Corrupt the base model, then check whether each edit arm can find and repair the damage. Calibration is clean in every case.

- **C1, obvious:** 3 high-traffic leaves, δ = +2 × std(y).
- **C2, subtle:** 5 leaves, δ = +0.3 × std(y).
- **C3, hidden:** 3 low-traffic leaves (n_cal < 20).
- **C4, correlated:** 3 leaves in different trees covering overlapping regions.

**Metrics:** recall of the corrupted leaves among those edited (leaf-level), and the share of the corruption's RMSE damage that was recovered (function-level). Overlapping leaves make exact leaf recovery partly unidentifiable, so both are reported.

---

## 10. Deliverables

1. Repo with a small, tested package, plus three notebooks that run end-to-end in replay mode without API keys and work in Colab.
2. Committed LLM cache and per-run results CSV.
3. Blog post built from `03_results.ipynb`, covering method, results, cost and limitations, including negative results.

### Repository layout

```text
research-llm-model-optimization/
├── PLAN.md
├── README.md
├── pyproject.toml              # uv
├── requirements.txt            # Colab
├── .env.example
├── configs/
│   └── experiment.yaml
├── tree_surgery/
│   ├── __init__.py
│   ├── datasets.py             # D1 generator, D2 loader, seeded splits
│   ├── train.py                # base model with early stopping, save/load
│   ├── inspector.py            # leaf table, decision paths, residual stats
│   ├── patcher.py              # fresh copies, apply/rollback deltas, edit log
│   ├── design_matrix.py        # sparse one-hot leaf matrix from pred_leaf
│   ├── optimizers.py           # ridge on a leaf subset, OMP/Lasso selection
│   ├── llm_client.py           # provider-agnostic client, cache, replay, cost
│   ├── prompts.py              # leaf table rendering, schemas
│   ├── arms.py                 # all arms behind one interface
│   ├── corruption.py           # C1–C4
│   ├── evaluation.py           # metrics, paired stats, bootstrap CI
│   └── runner.py               # experiment grid from config
├── notebooks/
│   ├── 01_leaf_surgery_101.ipynb   # edit a leaf, watch predictions move; linearity
│   ├── 02_the_experiment.ipynb     # run the grid (replay | live)
│   └── 03_results.ipynb            # plots and tables for the blog
├── llm_cache/
├── results/
└── tests/
```

---

## 11. Build phases and acceptance criteria

**Phase 1: Foundation**
- `datasets`, `train`, `inspector`, `patcher`, `design_matrix`.
- Tests (all must pass):
  - Editing leaf (t, l) by δ changes predictions by exactly δ for samples in that leaf and by 0 for all others.
  - `ŷ_base + X·d` equals the edited model's predictions (to 1e-9).
  - Two candidates from the same base share no state.
  - Save → load round-trips edited values.
  - Rollback restores the original predictions exactly.
- Notebook 01 drafted.

**Phase 2: Non-LLM arms**
- `original`, `random_opt`, `sparse_opt`, `refit`, `retrain`; evaluation and the runner.
- **Gate:** the full grid runs for all non-LLM arms on D1 and D2 × 5 seeds. Look at the results before any LLM spend.

**Phase 3: LLM infrastructure**
- `llm_client` with cache and replay, prompts, schemas, and a mock client.
- **Gate:** a dry run with the mock client covers the full grid; token estimates come from a handful of real calls.

**Phase 4: LLM arms**
- `llm_select` and `llm_direct`, named and anon, run on DeepSeek Flash.
- Review cost, then run Opus 5.5.

**Phase 5: Analysis and write-up**
- Corruption experiment, notebook 03, README results section, blog draft.

---

## 12. Known risks and limitations

- **Small effect sizes:** with a well-tuned base model, every edit arm may land within noise of `original`. That is a valid result.
- **Memorization (D2):** partly controlled by the anon ablation; D1 is the clean test.
- **LLM nondeterminism:** handled by 3 runs per seed and cached replay.
- **Context size:** the leaf table is roughly 60–90K tokens. If the base model ends up with far more leaves than expected, cap the number of rounds in config rather than pre-filtering leaves.
- **Generalization:** findings apply to LightGBM regression with squared loss and an in-distribution calibration set. They don't automatically extend to distribution shift, classification, or neural networks.

---

## Changelog

- 2026-11-17: v1 plan finalized.
- 2026-11-17: **Phase 1 complete** (75 tests pass; notebook 01 runs end-to-end). **Finding, decision pending before Phase 2:** at §3 settings the base models are far larger than the §1/§12 estimate of 500–1,500 leaves. Measured over 5 seeds (leaf table tokens ≈ chars/4):

  | Setting | D1 leaves | D1 table | D1 test RMSE / vs truth | D2 leaves | D2 table | D2 test RMSE |
  |---|---|---|---|---|---|---|
  | lr 0.05 (plan) | 3.7–4.8K | 160–215K | 4.292 / 1.420 | 11–15K (some seeds hit the 1,000-round cap) | 455–605K | 0.458 |
  | lr 0.05, cap 100 rounds (§12 fallback) | 1.5K | ~60K | 4.535 / 2.016 | – | – | – |
  | lr 0.1 | 1.8–2.3K | 78–99K | 4.310 / 1.467 | 7.3–11K | 290–445K | 0.459 |
  | lr 0.2 | 0.7–1.5K | 32–64K | 4.361 / 1.627 | 2.2–5.9K | 88–230K | 0.465 |
  | lr 0.3 | 0.7–1.1K | 28–47K | 4.414 / 1.784 | 2.2–3.6K | 88–140K | 0.475 |

  The §12 fallback (cap rounds) weakens D1 a lot: error against the truth goes up 42%, which is exactly the inflated-baseline problem §3 warns about. No single learning rate keeps D2's full table within a practical context and cost budget. D2 numbers come from a sandbox proxy (StatLib census data via a GitHub mirror, 207 rows with missing values dropped) because sklearn's download was blocked there; re-measure on the real loader.
- 2026-11-17: **Decision: query-tool access replaces the full leaf dump (§5).** The LLM gets a `query_leaves` tool (filter by feature/rule text, n_cal, residual; sort by any column; ≤ 50 rows per call) plus a static summary. It never receives a residual-ranked shortlist it didn't ask for. Base model stays at §3 settings (lr 0.05).
- 2026-11-17: **Cost estimate (`scripts/estimate_cost.py`; assumptions until the Phase 3 gate measures real tokens).** OpenRouter list prices: Opus 5.5 $4 in / $20 out / $0.20 cache read; DeepSeek V4.1 Flash ≈ $0.05–0.14 in / ≈ $0.57 out. A query session is about 50K input (36K of it cacheable) and 7.7K output. Full grid at 1 candidate per session: Flash ≈ $6 per run, Opus ≈ $187 per run (≈ $560 for 3 runs). Available budget: ₹1,000 OpenRouter credit (≈ $10.5). **Proposal (pending confirmation):** each session proposes 5 candidates, so 20 candidates = 4 feedback rounds; Flash runs the full grid × 3 runs (≈ $4); Opus is cut to an exploratory pilot, run only after the Flash results; hard spend caps on the API key and in the client.
- 2026-11-18: **Budget locked: Option A, LLM spend ≤ ₹200 (≈ $2.10).** Changes to §6/§7: max candidates per run **20 → 10** for every edit arm (fairness preserved); LLM sessions propose 5 candidates each (2 feedback rounds), allow ≤ 3 `query_leaves` calls of ≤ 30 rows, use low reasoning effort, and cap each call at 4K output tokens. The LLM is **DeepSeek V4.1 Flash** only (`deepseek/deepseek-v4.1-flash` via OpenRouter), on the full grid: both approaches, named + anon, D1 + D2, 5 seeds, 3 runs. Estimated cost ₹100–185. **Claude Opus 5.5 moves out of v1 scope.** It is configured (`anthropic/claude-opus-5.5`) but needs a separate budget (≈ ₹4,000 for one full run at these settings). Enforcement: an OpenRouter key credit limit, a client budget guard and per-call `max_tokens`.
- 2026-11-22: **D2 re-measured on the real loader (`fetch_california_housing`), §3 settings (lr 0.05, ≤ 1,000 rounds, patience 50).** Leaf-table tokens ≈ chars/4 of `render_leaf_table`.

  | Dataset | Seed | Trees | Leaves | Table tokens |
  |---|---|---|---|---|
  | D1 | 0 | 267 | 4,005 | 169K |
  | D1 | 1 | 288 | 4,320 | 183K |
  | D1 | 2 | 298 | 4,470 | 190K |
  | D1 | 3 | 245 | 3,675 | 156K |
  | D1 | 4 | 322 | 4,830 | 205K |
  | D2 | 0 | 1,000 | 15,000 | 605K |
  | D2 | 1 | 1,000 | 15,000 | 600K |
  | D2 | 2 | 998 | 14,970 | 603K |
  | D2 | 3 | 1,000 | 15,000 | 607K |
  | D2 | 4 | 999 | 14,985 | 603K |

  D1 matches the 2026-11-17 numbers. D2 lands at the top of the proxy range (15K leaves, ~600K tokens), which confirms the query-tool decision. **Open issue, decide before Phase 2:** D2 hits the 1,000-round cap on every seed (998–1,000 trees), so early stopping never fires and the D2 base model may be under-fit. §3 relies on early stopping to keep the baseline honest. Options: raise the round cap for D2, or accept the cap and document it as a limitation.
- 2026-11-23: **Decision: round cap raised 1,000 → 5,000 for all datasets (§3).** Resolves the 2026-11-22 open issue. Applied globally so both datasets share the same hyperparameters; D1 is unaffected because it early-stops at 245–322 rounds. `retrain` uses the same cap. **LLM cost is unchanged:** the LLM never receives the full table. The static block holds only summary counts, and `query_leaves` returns a fixed ≤ 30 rows per call. With `num_leaves=15` fixed, rule length doesn't grow with the number of trees. Costs that do grow are local compute only: D2 base training, `retrain`, the leaf table build, and a wider sparse design matrix for `sparse_opt`/`refit`. **Verification pending:** rerun the leaf-count script and confirm every D2 seed stops below 5,000 rounds. If any seed hits 5,000, revisit (e.g. lr 0.1 for D2 only).
- 2026-11-23: **5,000-round cap verified.** Every D2 seed now early-stops: 1,014–1,578 trees, 15.2K–23.7K leaves (test RMSE 0.453–0.465). D1 unchanged (245–322 trees).
- 2026-11-23: **Phase 2 complete; gate run done (no LLM spend).** `optimizers.py` (closed-form ridge on a leaf subset, OMP, Lasso path), `arms.py` (`original`, `random_opt`, `sparse_opt`, `refit`, `retrain` behind one interface), paired stats in `evaluation.py`, `runner.py`. 149 tests pass (76 Phase 1 + 73 new). Full grid: `python -m tree_surgery.runner` (~1 min) → `results/phase2/`.

  **Implementation choices (within the plan, recorded here):**
  - *Candidate* = one leaf subset. Its values come from ridge on calibration with λ picked from the §6 grid on validation; λ selection doesn't count against the 10-candidate budget. The same rule will apply to `llm_select`.
  - The shared `EditArm` base enforces the budget (raises on > 10 leaves, > 10 candidates, duplicates or unknown leaves), selects on validation, and checks that the patched booster matches `ŷ + X·d` (< 1e-8). Arms receive train / cal / val only; the test split isn't in their context.
  - `sparse_opt` (default `omp`): one greedy OMP path on unit-normalised leaf columns, with candidates = supports of size 1..10. `lasso` is available as an alternative (Lasso path restricted to its sparse end).
  - `random_opt` draws from leaves with ≥ 1 calibration row (others can't be fitted).
  - `refit`: decay ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99} chosen on validation.
  - **Identifiability (relevant to §9):** leaf columns are often exact duplicates on calibration (leaves in different trees covering the same rows), and a leaf holding most of a tree's rows can be repaired through its siblings plus a shift. In a planted C1 test, OMP recovers 97–99% of the damage, but leaf-level recall is sometimes only 2/3. Both §9 metrics are needed.

  **Gate results (test RMSE, mean over 5 seeds; paired diff vs `original` with 95% bootstrap CI):**

  | Arm | D1 test RMSE | D1 vs truth | Δ vs original (D1) | D2 test RMSE | Δ vs original (D2) |
  |---|---|---|---|---|---|
  | original | 4.2917 | 1.4197 | – | 0.4586 | – |
  | random_opt | 4.2953 | 1.4262 | +0.09% [−0.0019, +0.0129] | 0.4588 | +0.03% [−0.0002, +0.0004] |
  | sparse_opt | 4.2988 | 1.4385 | **+0.16% [+0.0012, +0.0150], 0/5 wins** | 0.4585 | −0.02% [−0.0008, +0.0005] |
  | refit | 4.2907 | 1.4227 | −0.02% [−0.0024, +0.0005] | 0.4583 | −0.08% [−0.0009, −0.0000] |
  | retrain | 4.2893 | 1.3664 | −0.06% [−0.0242, +0.0195] | 0.4502 | **−1.84% [−0.0109, −0.0061], 5/5** |

  **Reading.** With an early-stopped base model, a 10-leaf edit fitted on calibration finds no generalisable signal. `sparse_opt` lowers calibration RMSE on every cell (by up to 0.10 on D1) and validation RMSE slightly, but test RMSE doesn't follow. On D1 it is significantly *worse* than doing nothing, and further from the truth. Validation selection over 10 candidates × 6 λ (≈ 3K rows) overfits. `refit` picks decay 0.99 (grid edge, i.e. "change almost nothing") on 9/10 cells. Only `retrain`, which also learns new structure from the extra 15% of data, helps clearly, and only on D2.

  **Consequence for the decision rule (§8), open, decide before Phase 4:** because `sparse_opt` ≤ `original`, "`llm_select` beats `sparse_opt`" can be met by an arm that just edits less. Candidate amendments: (a) also require `llm_select` < `original` with a CI excluding 0; (b) add the no-edit model as a validation-guarded fallback candidate in every edit arm. Not applied; the pre-registered rule is unchanged until decided.
- 2026-11-23: **Phase 3 built; gate partly done (mock dry run ✅, real-call token check pending).** `llm_client.py` (OpenRouter/OpenAI-compatible + Anthropic adapter, cache/replay, budget guard, mock), `prompts.py` (static block, history block, `query_leaves`, Pydantic schemas, named/anon leaf views), `llm_session.py` (session loop, mock policy), `llm_select` / `llm_direct` in `arms.py`, runner `--llm mock|replay|live`. 221 tests pass (72 new). Phase 2 numbers are unchanged.

  **Implementation choices (within the plan, recorded here):**
  - *Session:* static prompt (system) + history (user) → ≤ 3 `query_leaves` calls → JSON answer. Once the query limit is reached the call is made with `tool_choice="none"`. An invalid answer gets one retry with the validation error; a second failure fails the session, which counts as 5 failed candidates against the budget. API errors, budget stops and replay cache misses stop the run instead (a resumed run reuses the cache).
  - *Feedback is calibration-only.* For `llm_select`, the calibration RMSE shown in the history is from a ridge fit at the smallest grid λ (1e-3), so it carries no validation information. The reported candidate still gets its λ chosen on validation, as in every edit arm. A test checks that the whole conversation is identical when validation and test labels change.
  - *`anon`:* `f0..fN` in a per-seed shuffled order, generic target `y`, no units or descriptions; ranges and target mean/std are kept (§5). The instructions are word-for-word identical to `named`. Ranges can still hint at identity (e.g. `[0, 23]`, `[0, 1]`); that is the plan as written.
  - *Static prompt* holds counts and quantiles only, no leaf lines. It is fixed per (dataset, seed, variant, approach), so providers can cache the prefix. Size: 2.3–2.8K chars (≈ 650 tokens). One 30-row query result is ≈ 0.9–1.3K tokens.
  - *Runs:* the cache key includes the run index, so the 3 runs per cell are 3 independent samples (temperature 1.0, recorded in config), not one answer replayed. A run index missing from the cache fails in replay.
  - *Spend control:* per-model prices in config; the guard refuses any call whose worst case (estimated input + `max_tokens` output, no cache discount) could cross ₹200 total, tracked across runs in `llm_cache/spend_ledger.json`. Logged cost is OpenRouter's reported `usage.cost`.
  - *Stats:* a **seed-clustered bootstrap CI** (resample seeds, then LLM runs within a seed) is reported next to the pre-registered CI. The 3 runs of a seed share its data split and base model, so the pre-registered CI, which resamples (seed, run) pairs as if independent, comes out too narrow. The decision rule still uses the pre-registered CI until changed.

  **Gate, mock dry run:** full grid (2 datasets × 5 seeds × 2 approaches × 2 variants × 3 runs = 120 LLM runs, 240 sessions, 720 calls), 0 failed candidates, about 3 min (`results/phase3_mock_dryrun/`, labelled as not a result). **Cost projection** from measured prompt sizes (`scripts/measure_tokens.py --dry`): ≈ 11.8K input tokens per session; 4.5–9.7K output depending on reasoning length → **≈ ₹88–159 for the full grid on DeepSeek Flash**, within the ₹200 cap. **Pending:** `python scripts/measure_tokens.py` with the OpenRouter key (OpenRouter is unreachable from the build sandbox): 8 real sessions (~32 calls, ≈ ₹3–6, capped at ₹15) that are real grid cells and stay in the cache. It prints real token counts, the projected grid cost and the served model id to pin in config.

  **Evidence for the open decision-rule issue (2026-11-23 Phase 2 entry).** In the mock dry run, a scripted policy with no LLM (random leaves among those its two queries returned) "beats" `sparse_opt` on D1: `llm_select/named − sparse_opt` = −0.18%, pre-registered CI [−0.0144, −0.0025], Wilcoxon p = 0.022, 11/15 wins. Against `original` it is flat (CI [−0.0040, +0.0020]). It doesn't beat `sparse_opt` on D2, so the rule's "both datasets" condition wasn't met. For `llm_direct/named − sparse_opt` on D1, the pre-registered CI excludes 0 but the seed-clustered CI [−0.0169, +0.0001] doesn't. **Decide before the Phase 4 live run:** (a) require `llm_select` < `original` too; (b) a validation-guarded no-edit fallback for every edit arm; (c) whether the decision rule uses the seed-clustered CI.
- 2026-11-24: **Phase 3 gate, real calls (`scripts/measure_tokens.py`, DeepSeek V4.1 Flash via OpenRouter; served id `deepseek/deepseek-v4.1-flash`, already the configured id).** 8 sessions on 2 grid cells (delivery seed 0 named, california seed 0 anon), run 0. Spend ₹5.66.

  | Arm | Sessions | Calls/session | Input tokens/session | Output/session (reasoning) | USD/session | Failed |
  |---|---|---|---|---|---|---|
  | `llm_select` | 4 | 3–4 | 14.3–23.1K | 4.0–7.3K (2.2–5.7K) | 0.0062 | 0/4 |
  | `llm_direct` | 4 | 3–4 | 11.8–22.5K | 6.9–9.7K (4.6–9.2K) | 0.0087 | **3/4** |

  Projected full grid at these rates: $1.79 ≈ ₹170 (cap ₹200). Prompt-cache hits were small (0–7.7K tokens/session).

  **Finding: the `llm_direct` failures were the token cap, not the model.** Every failed session spent ≈ 2 × 4,000 output tokens, almost all reasoning, on the answer call and its retry, and returned empty content. On OpenRouter the request's `max_tokens` covers reasoning and visible output together; when reasoning uses it up, the call ends with `finish_reason="length"` and no content, and is still billed. `reasoning.effort: low` did not keep DeepSeek's reasoning short, and OpenRouter honours a direct reasoning budget (`reasoning.max_tokens`) only for some providers (not documented for DeepSeek). Left as is, the grid would mostly measure the cap for `llm_direct`.

  **Fix (applies to both LLM arms, so they stay comparable):** `max_tokens_per_call` 4,000 → **12,000** (billed only for tokens actually generated). A truncated call (`finish_reason="length"`, no content) is now detected and logged as `truncated`; the retry says the reply ran out of tokens and asks for only the JSON. It still counts as the session's one retry. The static prompt is unchanged. Cache keys include `max_tokens`, so the 8 measured sessions will not be reused; the ₹5.66 is sunk.

  **Run order for Phase 4:** the projection (≈ ₹170 + ₹5.66 spent) leaves ≈ 12% headroom. The runner now takes `--run-ids`: run index 0 over the whole grid first, then 1, then 2. If the budget guard stops a pass, the completed runs are a balanced subset (every cell has the same runs) instead of a lopsided grid. Spend after run 0 (≈ ⅓ of the grid) gives a direct re-projection before committing to runs 1–2.

  **Before Phase 4:** re-check `llm_direct` only (`python scripts/measure_tokens.py --approaches llm_direct`, ≈ ₹3–4; these sessions are grid cells at the new cap and will be reused). The decision-rule question (2026-11-23 entries) is still open.
