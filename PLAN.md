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

- LightGBM regression, `num_leaves=15`, `learning_rate=0.05`, up to 1,000 rounds with **early stopping on validation** (patience 50). Other hyperparameters at their defaults, fixed in `configs/experiment.yaml`.
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
