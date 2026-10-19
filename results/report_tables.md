# FedKG-Maint: results summary


## C-MAPSS: reference setups vs federated (held-out factory FD004)

| setup | own_rmse | heldout_rmse | heldout_score | comm_mb | runs |
|---|---|---|---|---|---|
| local-only (FD001) | 14.20 | 48.77 | 1,018,184.38 | 0.00 | 1 |
| local-only (FD002) | 16.38 | 31.22 | 12,113.88 | 0.00 | 1 |
| local-only (FD003) | 17.13 | 54.61 | 1,936,184.00 | 0.00 | 1 |
| centralized (pooled) | 15.92 | 30.32 | 12,232.42 | 0.00 | 1 |
| fedavg | 23.02 | 32.90 | 15,715.40 | 54.38 | 1 |
| fedprox (mu=0.01) | 31.03 | 39.38 | 197,139.84 | 54.38 | 1 |

own_rmse = test sets of the training factories; heldout_* = unseen factory FD004; score = NASA scoring function (lower is better).


## C-MAPSS: significance (FedAvg vs others, held-out RMSE)

_(need >= 3 seeds; found 1 -- not enough for a significance test)_


## Privacy vs performance (DP-FedAvg, clip = 1.0, delta=1e-5)

| sigma | noise | epsilon | own_rmse | heldout_rmse |
|---|---|---|---|---|
| 0.00 | 0.00 | inf (no noise) | 31.47 | 42.03 |

noise = sigma x clip is the effective noise scale; with only 3 clients, meaningful epsilon (<10) requires noise that destroys accuracy.


## Privacy vs performance (DP-FedAvg, clip = 3.0, delta=1e-5)

| sigma | noise | epsilon | own_rmse | heldout_rmse |
|---|---|---|---|---|
| 0.00 | 0.00 | inf (no noise) | 22.77 | 33.61 |
| 0.05 | 0.15 | 8,618.4 | 28.85 | 36.75 |
| 0.20 | 0.60 | 651.8 | 84.52 | 77.99 |
| 0.50 | 1.50 | 140.7 | 331.35 | 332.05 |

noise = sigma x clip is the effective noise scale; with only 3 clients, meaningful epsilon (<10) requires noise that destroys accuracy.


## Azure failure prediction -- All features (held-out factory: model4)

| setup | own_prauc | own_f1 | heldout_prauc | heldout_f1 | comm_mb | runs |
|---|---|---|---|---|---|---|
| local-only (model1) | 0.940 ± 0.003 | 0.933 ± 0.013 | 0.859 ± 0.012 | 0.833 ± 0.033 | 0.000 ± 0.000 | 3 |
| local-only (model2) | 0.930 ± 0.002 | 0.902 ± 0.014 | 0.878 ± 0.009 | 0.854 ± 0.012 | 0.000 ± 0.000 | 3 |
| local-only (model3) | 0.929 ± 0.001 | 0.929 ± 0.003 | 0.919 ± 0.003 | 0.922 ± 0.003 | 0.000 ± 0.000 | 3 |
| centralized (pooled) | 0.930 ± 0.003 | 0.934 ± 0.009 | 0.904 ± 0.009 | 0.912 ± 0.008 | 0.000 ± 0.000 | 3 |
| fedavg | 0.927 ± 0.001 | 0.908 ± 0.003 | 0.917 ± 0.003 | 0.900 ± 0.006 | 2.458 ± 0.000 | 3 |
| fedprox (mu=0.01) | 0.929 ± 0.000 | 0.915 ± 0.003 | 0.917 ± 0.003 | 0.906 ± 0.003 | 2.458 ± 0.000 | 3 |


## Azure significance -- All features (held-out PR-AUC, FedAvg vs others)

| comparison | mean diff | p-value | significant (p<0.05) |
|---|---|---|---|
| fedavg vs centralized (pooled) | 0.0127 | 0.0677 | no |
| fedavg vs fedprox (mu=0.01) | 0.0001 | 0.8548 | no |
| fedavg vs local-only (model1) | 0.0577 | 0.0134 | yes |
| fedavg vs local-only (model2) | 0.0390 | 0.0205 | yes |
| fedavg vs local-only (model3) | -0.0024 | 0.0346 | yes |

_Paired t-test over 3 seeds; with so few seeds power is low, interpret cautiously._


## Azure failure prediction -- Sensors + age only (held-out factory: model4)

| setup | own_prauc | own_f1 | heldout_prauc | heldout_f1 | comm_mb | runs |
|---|---|---|---|---|---|---|
| local-only (model1) | 0.394 ± 0.013 | 0.437 ± 0.004 | 0.146 ± 0.007 | 0.246 ± 0.007 | 0.000 ± 0.000 | 3 |
| local-only (model2) | 0.288 ± 0.016 | 0.409 ± 0.001 | 0.151 ± 0.003 | 0.281 ± 0.002 | 0.000 ± 0.000 | 3 |
| local-only (model3) | 0.267 ± 0.018 | 0.347 ± 0.006 | 0.290 ± 0.010 | 0.373 ± 0.008 | 0.000 ± 0.000 | 3 |
| centralized (pooled) | 0.299 ± 0.015 | 0.361 ± 0.013 | 0.264 ± 0.019 | 0.316 ± 0.012 | 0.000 ± 0.000 | 3 |
| fedavg | 0.299 ± 0.008 | 0.348 ± 0.003 | 0.265 ± 0.012 | 0.360 ± 0.012 | 2.113 ± 0.000 | 3 |
| fedprox (mu=0.01) | 0.298 ± 0.006 | 0.344 ± 0.009 | 0.263 ± 0.010 | 0.357 ± 0.007 | 2.113 ± 0.000 | 3 |


## Azure significance -- Sensors + age only (held-out PR-AUC, FedAvg vs others)

| comparison | mean diff | p-value | significant (p<0.05) |
|---|---|---|---|
| fedavg vs centralized (pooled) | 0.0016 | 0.7930 | no |
| fedavg vs fedprox (mu=0.01) | 0.0021 | 0.1920 | no |
| fedavg vs local-only (model1) | 0.1192 | 0.0007 | yes |
| fedavg vs local-only (model2) | 0.1139 | 0.0045 | yes |
| fedavg vs local-only (model3) | -0.0247 | 0.0464 | yes |

_Paired t-test over 3 seeds; with so few seeds power is low, interpret cautiously._


## Azure failure prediction -- No days_since (held-out factory: model4)

| setup | own_prauc | own_f1 | heldout_prauc | heldout_f1 | comm_mb | runs |
|---|---|---|---|---|---|---|
| local-only (model1) | 0.942 | 0.940 | 0.865 | 0.876 | 0.000 | 1 |
| local-only (model2) | 0.932 | 0.921 | 0.892 | 0.900 | 0.000 | 1 |
| local-only (model3) | 0.923 | 0.925 | 0.919 | 0.925 | 0.000 | 1 |
| centralized (pooled) | 0.924 | 0.919 | 0.900 | 0.908 | 0.000 | 1 |
| fedavg | 0.927 | 0.927 | 0.914 | 0.908 | 2.305 | 1 |
| fedprox (mu=0.01) | 0.928 | 0.929 | 0.913 | 0.908 | 2.305 | 1 |


## Azure significance -- No days_since (held-out PR-AUC, FedAvg vs others)

_(need >= 3 seeds; found 1 -- not enough for a significance test)_


## Azure feature ablation: held-out PR-AUC

| setup | All features | Sensors + age only | No days_since |
|---|---|---|---|
| centralized (pooled) | 0.904 | 0.264 | 0.900 |
| fedavg | 0.917 | 0.265 | 0.914 |
| fedprox (mu=0.01) | 0.917 | 0.263 | 0.913 |
| local-only (model1) | 0.859 | 0.146 | 0.865 |
| local-only (model2) | 0.878 | 0.151 | 0.892 |
| local-only (model3) | 0.919 | 0.290 | 0.919 |

Large drops show how much the maintenance/error features drive performance (a property of the simulated dataset -- a limitation).


## Explanation grounding (held-out FD004 engines)

| mode | explanations | mentions | unsupported | pct_with_hallucination | grounded_% |
|---|---|---|---|---|---|
| template_KG | 30.00 | 7.57 | 0.00 | 0.00 | 100.00 |
| template_noKG | 30.00 | 4.00 | 0.00 | 0.00 | 100.00 |
