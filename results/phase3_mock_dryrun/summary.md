# Phase 3 gate: MOCK dry run (not an experimental result)

The `llm_*` rows come from a scripted mock policy (`tree_surgery.llm_session.make_mock_policy`), not an LLM. This run checks the machinery end to end over the full grid; its RMSE numbers say nothing about LLMs. Token counts are text-length estimates with no reasoning tokens.

5 seeds per dataset. Lower RMSE is better.

## Per-arm means

| dataset | arm | test RMSE (mean ± sd) | val RMSE | cal RMSE | gap (test−cal) | test RMSE vs truth | leaves edited | runtime s |
|---|---|---|---|---|---|---|---|---|
| delivery | original | 4.2917 ± 0.0882 | 4.3446 | 4.2181 | +0.0736 | 1.4197 | 0.0 | 0.00 |
| delivery | random_opt | 4.2953 ± 0.0854 | 4.3385 | 4.2085 | +0.0868 | 1.4262 | 10.0 | 0.13 |
| delivery | sparse_opt | 4.2988 ± 0.0950 | 4.3343 | 4.1779 | +0.1208 | 1.4385 | 4.4 | 0.14 |
| delivery | refit | 4.2907 ± 0.0887 | 4.3469 | 4.2126 | +0.0781 | 1.4227 | 4260.0 | 0.38 |
| delivery | retrain | 4.2893 ± 0.1136 | 4.3062 | 3.5902 | +0.6991 | 1.3664 | – | 0.27 |
| delivery | llm_select/named | 4.2907 ± 0.0810 | 4.3332 | 4.1972 | +0.0935 | 1.4222 | 7.5 | 0.27 |
| delivery | llm_select/anon | 4.2929 ± 0.0828 | 4.3340 | 4.1967 | +0.0962 | 1.4292 | 6.7 | 0.29 |
| delivery | llm_direct/named | 4.2910 ± 0.0821 | 4.3357 | 4.1985 | +0.0925 | 1.4228 | 5.9 | 0.13 |
| delivery | llm_direct/anon | 4.2893 ± 0.0811 | 4.3368 | 4.1998 | +0.0895 | 1.4203 | 5.7 | 0.13 |
| california | original | 0.4586 ± 0.0056 | 0.4731 | 0.4637 | -0.0050 | – | 0.0 | 0.01 |
| california | random_opt | 0.4588 ± 0.0056 | 0.4725 | 0.4629 | -0.0042 | – | 10.0 | 0.58 |
| california | sparse_opt | 0.4585 ± 0.0055 | 0.4724 | 0.4587 | -0.0002 | – | 5.2 | 0.68 |
| california | refit | 0.4583 ± 0.0053 | 0.4731 | 0.4591 | -0.0009 | – | 19689.0 | 1.53 |
| california | retrain | 0.4502 ± 0.0053 | 0.4633 | 0.2904 | +0.1598 | – | – | 0.93 |
| california | llm_select/named | 0.4588 ± 0.0055 | 0.4720 | 0.4609 | -0.0021 | – | 7.4 | 1.29 |
| california | llm_select/anon | 0.4586 ± 0.0052 | 0.4721 | 0.4606 | -0.0019 | – | 7.1 | 1.39 |
| california | llm_direct/named | 0.4586 ± 0.0052 | 0.4723 | 0.4609 | -0.0022 | – | 5.7 | 0.58 |
| california | llm_direct/anon | 0.4587 ± 0.0055 | 0.4724 | 0.4612 | -0.0025 | – | 6.1 | 0.57 |

## Paired differences in test RMSE (a − b; negative = a better)

| dataset | a − b | mean diff | rel. % | 95% bootstrap CI | CI excl. 0 | seed-clustered CI | excl. 0 | a wins | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| california | random_opt − original | +0.0001 | +0.03% | [-0.0002, +0.0004] | no | [-0.0002, +0.0004] | no | 2/5 | 0.625 |
| delivery | random_opt − original | +0.0037 | +0.09% | [-0.0019, +0.0129] | no | [-0.0019, +0.0129] | no | 3/5 | 1.000 |
| california | sparse_opt − original | -0.0001 | -0.02% | [-0.0008, +0.0005] | no | [-0.0008, +0.0005] | no | 3/5 | 1.000 |
| delivery | sparse_opt − original | +0.0071 | +0.16% | [+0.0012, +0.0150] | yes | [+0.0012, +0.0150] | yes | 0/5 | 0.062 |
| california | sparse_opt − random_opt | -0.0002 | -0.05% | [-0.0007, +0.0001] | no | [-0.0007, +0.0001] | no | 3/5 | 0.625 |
| delivery | sparse_opt − random_opt | +0.0034 | +0.08% | [-0.0108, +0.0158] | no | [-0.0108, +0.0158] | no | 1/5 | 0.438 |
| california | refit − original | -0.0004 | -0.08% | [-0.0009, -0.0000] | yes | [-0.0009, -0.0000] | yes | 4/5 | 0.188 |
| delivery | refit − original | -0.0009 | -0.02% | [-0.0024, +0.0005] | no | [-0.0024, +0.0005] | no | 3/5 | 0.312 |
| california | retrain − original | -0.0084 | -1.84% | [-0.0109, -0.0061] | yes | [-0.0109, -0.0061] | yes | 5/5 | 0.062 |
| delivery | retrain − original | -0.0023 | -0.06% | [-0.0242, +0.0195] | no | [-0.0242, +0.0195] | no | 3/5 | 1.000 |
| california | sparse_opt − retrain | +0.0083 | +1.85% | [+0.0066, +0.0102] | yes | [+0.0066, +0.0102] | yes | 0/5 | 0.062 |
| delivery | sparse_opt − retrain | +0.0094 | +0.23% | [-0.0079, +0.0261] | no | [-0.0079, +0.0261] | no | 2/5 | 0.438 |
| california | llm_select/named − sparse_opt | +0.0003 | +0.06% | [-0.0002, +0.0008] | no | [-0.0004, +0.0012] | no | 8/15 | 0.762 |
| delivery | llm_select/named − sparse_opt | -0.0080 | -0.18% | [-0.0144, -0.0025] | yes | [-0.0187, -0.0003] | yes | 11/15 | 0.022 |
| california | llm_select/named − original | +0.0002 | +0.04% | [-0.0002, +0.0006] | no | [-0.0003, +0.0007] | no | 6/15 | 0.679 |
| delivery | llm_select/named − original | -0.0010 | -0.02% | [-0.0040, +0.0020] | no | [-0.0048, +0.0030] | no | 9/15 | 0.561 |
| california | llm_select/named − retrain | +0.0086 | +1.91% | [+0.0073, +0.0101] | yes | [+0.0065, +0.0111] | yes | 0/15 | 0.000 |
| delivery | llm_select/named − retrain | +0.0014 | +0.05% | [-0.0120, +0.0145] | no | [-0.0218, +0.0234] | no | 6/15 | 0.934 |
| california | llm_select/named − llm_select/anon | +0.0002 | +0.04% | [-0.0002, +0.0005] | no | [-0.0002, +0.0007] | no | 8/15 | 0.720 |
| delivery | llm_select/named − llm_select/anon | -0.0022 | -0.05% | [-0.0054, +0.0010] | no | [-0.0067, +0.0021] | no | 9/15 | 0.229 |
| california | llm_direct/named − sparse_opt | +0.0001 | +0.02% | [-0.0004, +0.0006] | no | [-0.0006, +0.0009] | no | 8/15 | 0.847 |
| delivery | llm_direct/named − sparse_opt | -0.0078 | -0.18% | [-0.0137, -0.0019] | yes | [-0.0169, +0.0001] | no | 11/15 | 0.030 |
| california | llm_direct/named − llm_direct/anon | -0.0001 | -0.01% | [-0.0005, +0.0005] | no | [-0.0007, +0.0007] | no | 9/15 | 0.561 |
| delivery | llm_direct/named − llm_direct/anon | +0.0017 | +0.04% | [-0.0024, +0.0057] | no | [-0.0028, +0.0064] | no | 5/15 | 0.489 |

## LLM usage

| dataset | arm | runs | calls | tokens in | tokens out | cost USD | failed cands |
|---|---|---|---|---|---|---|---|
| delivery | llm_select/named | 15 | 90 | 215,697 | 16,403 | 0.0314 | 0/150 |
| delivery | llm_select/anon | 15 | 90 | 182,596 | 16,345 | 0.0281 | 0/150 |
| delivery | llm_direct/named | 15 | 90 | 217,862 | 20,774 | 0.0343 | 0/150 |
| delivery | llm_direct/anon | 15 | 90 | 184,817 | 20,495 | 0.0308 | 0/150 |
| california | llm_select/named | 15 | 90 | 205,532 | 16,987 | 0.0307 | 0/150 |
| california | llm_select/anon | 15 | 90 | 183,278 | 16,944 | 0.0285 | 0/150 |
| california | llm_direct/named | 15 | 90 | 208,221 | 20,898 | 0.0334 | 0/150 |
| california | llm_direct/anon | 15 | 90 | 185,967 | 21,096 | 0.0313 | 0/150 |

Total: 720 calls, 1,583,970 tokens in, 149,942 out, $0.2484 (billed this run: $0.0000).

## Base models

- california: 1014–1578 trees; seeds hitting the round cap: 0
- delivery: 245–322 trees; seeds hitting the round cap: 0
