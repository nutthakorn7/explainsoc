# ExplainSOC

**Leakage-Audited Explainable AI for Security Alert Triage**

Companion code for:
> N. Chalaemwongwan and T. Siritorn, "ExplainSOC: Leakage-Audited Explainable AI for Security Alert Triage," *Computers & Security* (under review).

---

## The core finding

Most XAI-for-security triage papers report inflated results because their model inputs encode **label leakage** — rule-derived fields that are deterministic transforms of the triage label. On the SALAD dataset, 5 of 9 categorical fields achieve single-field macro-F1 = 1.000. Our leakage-free model achieves macro-F1 = 0.946 — a substantially honest number that holds up in deployment.

## Repository layout

```
explainsoc/
├── leakage_audit.py       # C1: single-field target-recovery test
├── faithfulness_eval.py   # C3: deletion/insertion AUC, stability, CF validity
├── make_figures.py        # paper figures (leakage chart, AUC curves, explanation card)
├── blinding_generator.py  # produce blinded evidence cards for analyst annotation
├── compute_iaa.py         # IAA: Krippendorff α, Gwet AC1/AC2, pairwise κ
├── borderline_selector.py # select borderline alerts for annotation
└── annotation_A1.html     # browser-based annotation tool (offline, no server)
```

## Quick start

```bash
# 1. Clone and install
pip install scikit-learn scipy matplotlib numpy

# 2. Download SALAD (public dataset)
#    Place salad_full.csv in data/

# 3. Run leakage audit
python leakage_audit.py --data data/salad_full.csv

# 4. Run faithfulness evaluation (trains clean model, computes SHAP metrics)
python faithfulness_eval.py --data data/salad_full.csv --eval_n 300

# 5. Generate paper figures
python make_figures.py --data data/salad_full.csv
```

## Leakage audit results (SALAD, N=40,000)

| Feature | Type | Single-field F1 | Leaks? |
|---------|------|----------------|--------|
| attack_category | rule-derived | 1.000 | ✅ LEAK |
| severity | rule-derived | 1.000 | ✅ LEAK |
| alert_type | rule-derived | 1.000 | ✅ LEAK |
| priority_score | rule-derived | 1.000 | ✅ LEAK |
| kill_chain_phase | rule-derived | 1.000 | ✅ LEAK |
| protocol | observable | 0.703 | — |
| mitre_tactic | derived | 0.698 | — |
| difficulty_level | metadata | 0.597 | — |
| source_dataset | metadata | 0.197 | — |
| **flow features (8)** | observable | **0.922 (baseline)** | — |

## Model performance (leakage-free, GBDT n=200)

| Model | Macro-F1 | Suppress | Investigate | Escalate |
|-------|----------|----------|-------------|----------|
| Majority-class baseline | 0.198 | 0.594 | 0.000 | 0.000 |
| Stratified-random baseline | 0.331 | 0.417 | 0.359 | 0.216 |
| Leaking baseline (all features) | ≈1.000 | 1.000 | 1.000 | 1.000 |
| **ExplainSOC (observable only)** | **0.946** | **0.991** | **0.942** | **0.906** |

## Faithfulness metrics (permutation SHAP, 200 samples)

| Metric | Value |
|--------|-------|
| Faithfulness (1−deletion AUC) | 0.599 |
| Faithfulness (insertion AUC) | 0.843 |
| Stability (Kendall τ) | 0.321* |
| Counterfactual validity | **0.985** |

*Lower bound — permutation SHAP variance under n_perm=5; tree-SHAP would be exact.

## Data

- **SALAD** (public): derived from CICIDS2017 + UNSW-NB15. See `data/README.md` for download instructions.
- **Operational SOC data**: not shared (PDPA). Aggregate results and the anonymized annotation schema are in `data/operational_schema.md`.

## Citation

```bibtex
@article{chalaemwongwan2025explainsoc,
  title   = {{ExplainSOC}: Leakage-Audited Explainable {AI} for Security Alert Triage},
  author  = {Chalaemwongwan, Nutthakorn and Siritorn, Tanakorn},
  journal = {Computers \& Security},
  year    = {2025},
  note    = {Under review}
}
```

## License

MIT. The SALAD dataset has its own license; see the SALAD repository.

## Acknowledgement

Supported by the King Mongkut's Institute of Technology Ladkrabang Research Fund (New Faculty Research Grant).
