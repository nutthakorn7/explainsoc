#!/usr/bin/env python3
"""
faithfulness_eval.py — offline faithfulness metrics for ExplainSOC SHAP explanations.

Computes (no human subjects needed, fully reproducible):
  1. Deletion AUC  — mask top-k SHAP features, measure model-output drop
  2. Insertion AUC — reveal top-k SHAP features from baseline, measure output rise
  3. Stability     — Kendall τ of attributions under small Gaussian noise (100 trials)
  4. Counterfactual validity — fraction of CFs that actually flip the predicted class

Trains a clean GBDT on SALAD (leakage-free flow features only) and exports:
  - faithfulness_results.json  (all metrics, ready to paste into paper)
  - clean_model.joblib         (the model artifact for later use)

Usage:
  python3 faithfulness_eval.py --data "../JN8 SALAD_DataSet/SALAD/salad_full.csv" --cap 40000
"""
import csv, argparse, json, math, random
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import kendalltau

# Leakage-free observable feature set (from leakage_audit.py)
FLOW = ["src_port", "dst_port", "flow_duration",
        "total_fwd_packets", "total_bwd_packets",
        "flow_bytes_per_sec", "alert_count_1h", "alert_count_24h"]
TARGET = "triage_decision"

# ── data loading ──────────────────────────────────────────────────────────────
def load_csv(path, cap=40000):
    rows = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= cap: break
            rows.append(r)
    return rows

def to_X(rows, keys):
    X = []
    for r in rows:
        v = []
        for k in keys:
            try: v.append(float(r.get(k, 0) or 0))
            except ValueError: v.append(0.0)
        X.append(v)
    return np.array(X)

# ── SHAP approximation (TreeSHAP not required — use permutation for portability) ──
def permutation_shap(model, X, baseline, n_perm=20, rng=None):
    """Permutation-based Shapley values (unbiased but approximate).
    Returns attributions matrix shape (n_samples, n_features)."""
    if rng is None: rng = np.random.RandomState(42)
    n, d = X.shape
    phi = np.zeros((n, d))
    pred_cls = model.predict(X).astype(int)    # fixed predicted class per sample
    # tile baseline to (n, d) so indexing matches X
    B = np.tile(baseline[0], (n, 1))
    for _ in range(n_perm):
        perm = rng.permutation(d)
        X_prev = B.copy()
        p_prev = model.predict_proba(X_prev)
        for fi in perm:
            X_curr = X_prev.copy()
            X_curr[:, fi] = X[:, fi]
            p_curr = model.predict_proba(X_curr)
            for i in range(n):
                phi[i, fi] += p_curr[i, pred_cls[i]] - p_prev[i, pred_cls[i]]
            X_prev = X_curr
            p_prev = p_curr
    return phi / n_perm

# ── Metric 1 & 2: Deletion / Insertion AUC ───────────────────────────────────
def deletion_insertion_auc(model, X_test, phi, baseline, n_steps=8):
    """
    deletion_auc : features removed top-down → output should drop (lower = more faithful)
    insertion_auc: features revealed top-down → output should rise (higher = more faithful)
    We report faithfulness = 1 - deletion_auc and insertion_auc (both higher = better).
    """
    n, d = X_test.shape
    pred_cls = model.predict(X_test).astype(int)
    # rank features by |phi| descending
    rank = np.argsort(-np.abs(phi), axis=1)   # (n, d)

    del_probs, ins_probs = [], []
    for step in range(n_steps + 1):
        k = int(d * step / n_steps)
        X_del = X_test.copy()
        X_ins = np.tile(baseline, (n, 1))
        for i in range(n):
            top = rank[i, :k]
            X_del[i, top] = baseline[0, top]   # mask top-k → baseline
            X_ins[i, top] = X_test[i, top]     # reveal top-k
        p_del = model.predict_proba(X_del)
        p_ins = model.predict_proba(X_ins)
        del_probs.append(np.mean([p_del[i, pred_cls[i]] for i in range(n)]))
        ins_probs.append(np.mean([p_ins[i, pred_cls[i]] for i in range(n)]))

    # AUC by trapezoid
    xs = [i / n_steps for i in range(n_steps + 1)]
    del_auc = float(np.trapezoid(del_probs, xs))
    ins_auc = float(np.trapezoid(ins_probs, xs))
    return 1.0 - del_auc, ins_auc   # (faithfulness_del, faithfulness_ins)

# ── Metric 3: Stability (Kendall τ under Gaussian noise) ─────────────────────
def stability(model, X_test, phi, sigma_frac=0.01, n_trials=100, rng=None):
    """Mean Kendall τ of attribution rank-order across noise trials."""
    if rng is None: rng = np.random.RandomState(42)
    n, d = X_test.shape
    feature_range = X_test.max(0) - X_test.min(0)
    feature_range[feature_range == 0] = 1.0
    sigma = sigma_frac * feature_range

    taus = []
    for _ in range(n_trials):
        noise = rng.randn(*X_test.shape) * sigma
        X_noisy = X_test + noise
        phi_noisy = permutation_shap(model, X_noisy,
                                     np.zeros_like(X_test[:1]), n_perm=5, rng=rng)
        for i in range(n):
            tau, _ = kendalltau(phi[i], phi_noisy[i])
            if not math.isnan(tau): taus.append(tau)

    return float(np.mean(taus))

# ── Metric 4: Counterfactual validity ────────────────────────────────────────
def counterfactual_validity(model, X_test, phi, n_cf=200, rng=None):
    """
    Simple greedy CF: flip top-attributed observable feature toward the
    value that would reduce predicted-class probability most.
    Validity = fraction where class actually flips.
    """
    if rng is None: rng = np.random.RandomState(42)
    n, d = X_test.shape
    pred_cls = model.predict(X_test).astype(int)
    rank = np.argsort(-np.abs(phi), axis=1)

    valid = 0
    subset = rng.choice(n, size=min(n_cf, n), replace=False)
    for i in subset:
        X_cf = X_test[i:i+1].copy()
        for fi in rank[i]:
            # try perturbing ±10% of the feature range
            feat_range = X_test[:, fi].max() - X_test[:, fi].min()
            if feat_range == 0: continue
            best_X = None; best_prob = model.predict_proba(X_cf)[0, pred_cls[i]]
            for delta in [-0.1 * feat_range, 0.1 * feat_range,
                          -0.3 * feat_range, 0.3 * feat_range]:
                X_try = X_cf.copy(); X_try[0, fi] += delta
                p = model.predict_proba(X_try)[0, pred_cls[i]]
                if p < best_prob:
                    best_prob = p; best_X = X_try
            if best_X is not None:
                if model.predict(best_X)[0] != pred_cls[i]:
                    valid += 1; break   # CF found, class flips
    return valid / len(subset)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../JN8 SALAD_DataSet/SALAD/salad_full.csv")
    ap.add_argument("--cap", type=int, default=40000)
    ap.add_argument("--eval_n", type=int, default=300,
                    help="number of test samples to use for SHAP (slow)")
    args = ap.parse_args()

    print("Loading data...")
    rows = load_csv(args.data, args.cap)
    X = to_X(rows, FLOW)
    le = LabelEncoder()
    y = le.fit_transform([str(r.get(TARGET, "")) for r in rows])
    classes = le.classes_

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}, Classes: {list(classes)}")

    # ── Train clean model ──
    print("Training clean GBDT (leakage-free features)...")
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                     learning_rate=0.1, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    per_class = f1_score(y_te, y_pred, average=None)
    print(f"\nClean model macro-F1 = {macro_f1:.4f}")
    for cls, f in zip(classes, per_class):
        print(f"  {cls:<15} F1 = {f:.4f}")

    # ── SHAP on a subset for speed ──
    n_eval = min(args.eval_n, len(X_te))
    idx = np.random.RandomState(42).choice(len(X_te), n_eval, replace=False)
    X_sub = X_te[idx]
    baseline = np.zeros_like(X_sub[:1])   # all-zeros baseline

    print(f"\nComputing permutation SHAP on {n_eval} samples (n_perm=20)...")
    rng = np.random.RandomState(42)
    phi = permutation_shap(clf, X_sub, baseline, n_perm=20, rng=rng)

    print("Computing deletion/insertion AUC...")
    faith_del, faith_ins = deletion_insertion_auc(clf, X_sub, phi, baseline)
    print(f"  Faithfulness (1-deletion AUC) = {faith_del:.4f}")
    print(f"  Faithfulness (insertion AUC)  = {faith_ins:.4f}")

    print("Computing stability (100 noise trials on 50 samples)...")
    stab_idx = np.random.RandomState(42).choice(n_eval, size=min(50, n_eval), replace=False)
    stab = stability(clf, X_sub[stab_idx], phi[stab_idx], n_trials=100, rng=rng)
    print(f"  Stability (mean Kendall τ)    = {stab:.4f}")

    print("Computing counterfactual validity (200 samples)...")
    cf_val = counterfactual_validity(clf, X_sub, phi, n_cf=200, rng=rng)
    print(f"  Counterfactual validity       = {cf_val:.4f}")

    # ── Top feature importances ──
    print("\nMean |SHAP| per feature:")
    mean_phi = np.mean(np.abs(phi), axis=0)
    for feat, imp in sorted(zip(FLOW, mean_phi), key=lambda x: -x[1]):
        print(f"  {feat:<25} {imp:.4f}")

    results = {
        "clean_macro_f1": round(macro_f1, 4),
        "per_class_f1": {cls: round(float(f), 4) for cls, f in zip(classes, per_class)},
        "faithfulness_del": round(faith_del, 4),
        "faithfulness_ins": round(faith_ins, 4),
        "stability_tau": round(stab, 4),
        "cf_validity": round(cf_val, 4),
        "n_train": len(X_tr), "n_test": len(X_te),
        "features": FLOW, "target": TARGET,
        "model": "GradientBoostingClassifier(n_estimators=200, max_depth=5)"
    }
    with open("faithfulness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to faithfulness_results.json")
    print("\n--- PAPER TABLE VALUES ---")
    print(f"Macro-F1 (clean):               {macro_f1:.3f}")
    print(f"Faithfulness (1-deletion AUC):  {faith_del:.3f}")
    print(f"Faithfulness (insertion AUC):   {faith_ins:.3f}")
    print(f"Stability (mean Kendall τ):     {stab:.3f}")
    print(f"Counterfactual validity:        {cf_val:.3f}")

if __name__ == "__main__":
    main()
