# FedKG-Maint: Privacy-Preserving Predictive Maintenance

Federated learning for predictive maintenance across simulated factories, with client-level
differential privacy and knowledge-graph-grounded explanations.

- **Datasets:** NASA C-MAPSS (remaining useful life) and Microsoft Azure Predictive Maintenance (24-hour failure prediction)
- **Factories:** C-MAPSS FD001-FD004 and Azure machine models model1-model4; one factory per dataset (FD004, model4) is held out and never used for training or model selection
- **Methods:** local-only and centralized references, FedAvg, FedProx, DP-FedAvg, per-factory knowledge graphs, grounded explanations with hallucination and faithfulness checks
- **Full write-up:** see the report in `report/`

## Headline results

| Experiment | Result |
|---|---|
| C-MAPSS, unseen factory FD004 (RMSE, lower is better) | FedAvg 32.9, centralized 30.3, local-only 31.2 to 54.6 |
| Azure, unseen machine model (PR-AUC, 3 seeds) | FedAvg 0.917, centralized 0.904, local-only 0.859 to 0.919 |
| Azure, sensors only (no error counts or maintenance features) | PR-AUC about 0.26 for all setups |
| Differential privacy, 3 clients | meaningful epsilon requires noise that destroys accuracy |
| Explanations (template-based) | 7.6 grounded entities with KG vs 4.0 without, no unsupported claims |

C-MAPSS results are single-seed; Azure results use 3 seeds. Limitations are discussed in the report.

## Setup

Tested with Python 3.12.

```
pip install -r requirements.txt
```

## Data (not included in this repository)

Download the datasets and place them in the project root:

- **C-MAPSS:** https://data.nasa.gov/docs/legacy/CMAPSSData.zip (dataset page: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). Unzip so that the folder `CMAPSSData/` contains `train_FD00X.txt`, `test_FD00X.txt` and `RUL_FD00X.txt`.
- **Azure Predictive Maintenance:** https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azurepredictive-maintenance. Put `PdM_telemetry.csv`, `PdM_errors.csv`, `PdM_maint.csv`, `PdM_failures.csv` and `PdM_machines.csv` in the project root.

## Run (in order)

```
# C-MAPSS pipeline
python step1_cmapss_prep.py --data_dir CMAPSSData --out_dir processed
python step2_baselines.py                      # local-only and centralized references
python step3_federated.py                      # FedAvg and FedProx (40 rounds)
python step4_privacy.py --clip 3.0 --sigmas 0 0.05 0.2 0.5 --out results/step4_privacy_clip3.csv
python step5_explain.py                        # knowledge graphs and explanations
python step8_scaling.py --central              # number of factories and client heterogeneity

# Azure pipeline
python azure_prep.py --data_dir .
python step6_azure.py --data_dir . --seeds 3
python step6_azure.py --data_dir . --seeds 3 --drop days_since error --out results/noextra.csv

# Tables, significance tests and figures
python step7_report_tables.py
```

Use `--seeds 3` on steps 2 and 3 for multi-seed C-MAPSS runs. To test an LLM with and without
knowledge-graph grounding, set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) and run
`python step5_explain.py --llm anthropic --model claude-sonnet-5-5` (this path is untested).
If you change `azure_prep.py`, delete `processed/azure_features.npz` so features are rebuilt.

## Files

| File | Purpose |
|---|---|
| `step1_cmapss_prep.py` | C-MAPSS cleaning, windows, RUL targets, local normalization |
| `common.py` | shared LSTM model, training loop, RMSE and NASA score |
| `step2_baselines.py` | local-only and centralized references |
| `step3_federated.py` | FedAvg and FedProx |
| `step4_privacy.py` | client-level differential privacy with epsilon accounting |
| `kg_explain.py`, `step5_explain.py` | knowledge graph, explanations, grounding and faithfulness checks |
| `step8_scaling.py` | effect of the number of factories and client composition on FedAvg |
| `azure_prep.py`, `step6_azure.py` | Azure features, labels and federated experiments |
| `step7_report_tables.py` | report tables, significance tests, figures |
| `results/`, `kg/` | result tables, figures, per-factory knowledge graphs |

## Notes

- The failure-mode-to-component links and maintenance actions in the knowledge graph are illustrative and have not been reviewed by a domain expert.
- Both datasets are simulated; results will not transfer directly to real factories.
- Datasets are subject to their own licenses and terms and are not redistributed here.
