# Results summary

5 seeds per dataset. Lower RMSE is better.

## Per-arm means

| dataset | arm | test RMSE (mean ± sd) | val RMSE | cal RMSE | gap (test−cal) | test RMSE vs truth | leaves edited | runtime s |
|---|---|---|---|---|---|---|---|---|
| delivery | original | 4.2917 ± 0.0882 | 4.3446 | 4.2181 | +0.0736 | 1.4197 | 0.0 | 0.00 |
| delivery | random_opt | 4.2953 ± 0.0854 | 4.3385 | 4.2085 | +0.0868 | 1.4262 | 10.0 | 0.11 |
| delivery | sparse_opt | 4.2988 ± 0.0950 | 4.3343 | 4.1779 | +0.1208 | 1.4385 | 4.4 | 0.12 |
| delivery | refit | 4.2907 ± 0.0887 | 4.3469 | 4.2126 | +0.0781 | 1.4227 | 4260.0 | 0.37 |
| delivery | retrain | 4.2893 ± 0.1136 | 4.3062 | 3.5902 | +0.6991 | 1.3664 | – | 0.27 |
| california | original | 0.4586 ± 0.0056 | 0.4731 | 0.4637 | -0.0050 | – | 0.0 | 0.01 |
| california | random_opt | 0.4588 ± 0.0056 | 0.4725 | 0.4629 | -0.0042 | – | 10.0 | 0.52 |
| california | sparse_opt | 0.4585 ± 0.0055 | 0.4724 | 0.4587 | -0.0002 | – | 5.2 | 0.61 |
| california | refit | 0.4583 ± 0.0053 | 0.4731 | 0.4591 | -0.0009 | – | 19689.0 | 1.45 |
| california | retrain | 0.4502 ± 0.0053 | 0.4633 | 0.2904 | +0.1598 | – | – | 0.91 |

## Paired differences in test RMSE (a − b; negative = a better)

| dataset | a − b | mean diff | rel. % | 95% bootstrap CI | CI excl. 0 | a wins | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| california | random_opt − original | +0.0001 | +0.03% | [-0.0002, +0.0004] | no | 2/5 | 0.625 |
| delivery | random_opt − original | +0.0037 | +0.09% | [-0.0019, +0.0129] | no | 3/5 | 1.000 |
| california | sparse_opt − original | -0.0001 | -0.02% | [-0.0008, +0.0005] | no | 3/5 | 1.000 |
| delivery | sparse_opt − original | +0.0071 | +0.16% | [+0.0012, +0.0150] | yes | 0/5 | 0.062 |
| california | sparse_opt − random_opt | -0.0002 | -0.05% | [-0.0007, +0.0001] | no | 3/5 | 0.625 |
| delivery | sparse_opt − random_opt | +0.0034 | +0.08% | [-0.0108, +0.0158] | no | 1/5 | 0.438 |
| california | refit − original | -0.0004 | -0.08% | [-0.0009, -0.0000] | yes | 4/5 | 0.188 |
| delivery | refit − original | -0.0009 | -0.02% | [-0.0024, +0.0005] | no | 3/5 | 0.312 |
| california | retrain − original | -0.0084 | -1.84% | [-0.0109, -0.0061] | yes | 5/5 | 0.062 |
| delivery | retrain − original | -0.0023 | -0.06% | [-0.0242, +0.0195] | no | 3/5 | 1.000 |
| california | sparse_opt − retrain | +0.0083 | +1.85% | [+0.0066, +0.0102] | yes | 0/5 | 0.062 |
| delivery | sparse_opt − retrain | +0.0094 | +0.23% | [-0.0079, +0.0261] | no | 2/5 | 0.438 |

## Base models

- california: 1014–1578 trees; seeds hitting the round cap: 0
- delivery: 245–322 trees; seeds hitting the round cap: 0
