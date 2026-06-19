#!/usr/bin/env python3
"""
make_figures.py — generate paper figures for ExplainSOC (Computers & Security).

Figure 1: Analyst explanation card (SHAP bar + counterfactual)
Figure 2: Deletion/insertion AUC curves
Figure 3: Leakage audit bar chart

All figures saved as PDF (vector) + PNG (preview).
"""
import csv, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

FLOW = ["src_port", "dst_port", "flow_duration",
        "total_fwd_packets", "total_bwd_packets",
        "flow_bytes_per_sec", "alert_count_1h", "alert_count_24h"]
TARGET = "triage_decision"
COLORS = {"escalate": "#d62728", "investigate": "#ff7f0e", "suppress": "#2ca02c"}

# ── load & train ──────────────────────────────────────────────────────────────
def load_csv(path, cap=40000):
    rows = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= cap: break
            rows.append(r)
    return rows

def to_X(rows):
    X = []
    for r in rows:
        v = []
        for k in FLOW:
            try: v.append(float(r.get(k, 0) or 0))
            except ValueError: v.append(0.0)
        X.append(v)
    return np.array(X)

# ── SHAP (permutation, fast for figure) ──────────────────────────────────────
def shap_one(model, x, baseline, pred_cls, n_perm=50, rng=None):
    """Shapley values for ONE sample x (shape d,). Returns phi (shape d,)."""
    if rng is None: rng = np.random.RandomState(42)
    d = len(x)
    phi = np.zeros(d)
    B = baseline.copy()
    for _ in range(n_perm):
        perm = rng.permutation(d)
        x_prev = B.copy()
        p_prev = model.predict_proba(x_prev.reshape(1,-1))[0, pred_cls]
        for fi in perm:
            x_curr = x_prev.copy()
            x_curr[fi] = x[fi]
            p_curr = model.predict_proba(x_curr.reshape(1,-1))[0, pred_cls]
            phi[fi] += p_curr - p_prev
            x_prev = x_curr
            p_prev = p_curr
    return phi / n_perm

# ── Figure 1: Analyst explanation card ───────────────────────────────────────
def fig_explanation_card(clf, X_te, y_te, le, rng):
    # Pick a representative "escalate" alert
    esc_idx = np.where(clf.predict(X_te) == le.transform(["escalate"])[0])[0]
    if len(esc_idx) == 0:
        esc_idx = np.arange(len(X_te))
    i = rng.choice(esc_idx[:200])
    x = X_te[i]
    pred_cls_idx = clf.predict(X_te[i:i+1]).astype(int)[0]
    pred_cls_name = le.inverse_transform([pred_cls_idx])[0]
    proba = clf.predict_proba(X_te[i:i+1])[0]

    baseline = np.zeros(len(FLOW))
    phi = shap_one(clf, x, baseline, pred_cls_idx, n_perm=80, rng=rng)

    # Counterfactual: find feature to reduce escalate prob
    cf_feat_idx = np.argmax(phi)    # most contributing feature
    feat_range = X_te[:, cf_feat_idx].max() - X_te[:, cf_feat_idx].min()
    x_cf = x.copy()
    x_cf[cf_feat_idx] = max(0.0, x[cf_feat_idx] - 0.3 * feat_range)
    cf_cls = le.inverse_transform(clf.predict(x_cf.reshape(1,-1)).astype(int))[0]

    fig = plt.figure(figsize=(10, 5))
    fig.patch.set_facecolor("#f8f9fa")
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # Left: SHAP bar
    ax1 = fig.add_subplot(gs[0, 0])
    feat_labels = ["src_port", "dst_port", "flow_dur", "fwd_pkts",
                   "bwd_pkts", "bytes/s", "alerts(1h)", "alerts(24h)"]
    order = np.argsort(np.abs(phi))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in phi[order]]
    ax1.barh(range(len(FLOW)), phi[order], color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(FLOW)))
    ax1.set_yticklabels([feat_labels[j] for j in order], fontsize=9)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("SHAP attribution $\\phi_i$", fontsize=9)
    ax1.set_title(f"SHAP: why \"{pred_cls_name.upper()}\"?", fontsize=10, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: alert card + counterfactual
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    card_color = COLORS.get(pred_cls_name, "#888888")
    rect = mpatches.FancyBboxPatch((0.02, 0.55), 0.96, 0.41,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor=card_color, facecolor=card_color + "22")
    ax2.add_patch(rect)
    ax2.text(0.50, 0.94, f"Decision: {pred_cls_name.upper()}",
        ha="center", va="top", fontsize=12, fontweight="bold", color=card_color,
        transform=ax2.transAxes)
    cls_labels = le.classes_
    for ci, (cn, cp) in enumerate(zip(cls_labels, proba)):
        bar_w = 0.55 * cp
        bar_y = 0.80 - ci * 0.10
        bar_color = COLORS.get(cn, "#888888")
        bar_rect = mpatches.FancyBboxPatch((0.10, bar_y - 0.035), bar_w, 0.065,
            boxstyle="round,pad=0.005", facecolor=bar_color + "66", edgecolor=bar_color)
        ax2.add_patch(bar_rect)
        ax2.text(0.10 + bar_w + 0.02, bar_y, f"{cn}: {cp:.2f}",
            va="center", fontsize=8, color=bar_color, fontweight="bold",
            transform=ax2.transAxes)

    # Counterfactual box
    rect_cf = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.48,
        boxstyle="round,pad=0.02", linewidth=1,
        edgecolor="#888888", facecolor="#ffffff")
    ax2.add_patch(rect_cf)
    ax2.text(0.50, 0.48, "Counterfactual explanation",
        ha="center", va="top", fontsize=9, fontweight="bold", color="#444444",
        transform=ax2.transAxes)
    feat_name = feat_labels[cf_feat_idx].replace("_", " ")
    orig_val = x[cf_feat_idx]
    new_val = x_cf[cf_feat_idx]
    pct_change = 100 * (new_val - orig_val) / (orig_val + 1e-9)
    cf_text = (f"If '{feat_name}' were\n"
               f"{new_val:,.0f} ({pct_change:+.0f}% from {orig_val:,.0f}),\n"
               f"decision would change to:\n{cf_cls.upper()}")
    ax2.text(0.50, 0.36, cf_text, ha="center", va="top", fontsize=9,
        color="#333333", transform=ax2.transAxes, linespacing=1.5)
    ax2.set_title("Analyst Explanation Card", fontsize=10, fontweight="bold",
        pad=4, color="#333333")

    plt.suptitle("ExplainSOC: Leakage-Free SHAP Explanation for SOC Alert Triage",
                 fontsize=11, y=1.01, color="#222222")
    fig.savefig("fig1_explanation_card.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("fig1_explanation_card.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved fig1_explanation_card.pdf / .png")

# ── Figure 2: Deletion / Insertion AUC curves ────────────────────────────────
def fig_deletion_insertion(clf, X_sub, phi, baseline):
    n, d = X_sub.shape
    pred_cls = clf.predict(X_sub).astype(int)
    rank = np.argsort(-np.abs(phi), axis=1)
    n_steps = 8
    del_probs, ins_probs = [], []
    xs = [i / n_steps for i in range(n_steps + 1)]
    B = np.tile(baseline[0], (n, 1))
    for step in range(n_steps + 1):
        k = int(d * step / n_steps)
        X_del = X_sub.copy(); X_ins = B.copy()
        for i in range(n):
            top = rank[i, :k]
            X_del[i, top] = B[i, top]
            X_ins[i, top] = X_sub[i, top]
        p_del = clf.predict_proba(X_del)
        p_ins = clf.predict_proba(X_ins)
        del_probs.append(np.mean([p_del[i, pred_cls[i]] for i in range(n)]))
        ins_probs.append(np.mean([p_ins[i, pred_cls[i]] for i in range(n)]))

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    pcts = [x * 100 for x in xs]
    ax.plot(pcts, del_probs, "o-", color="#d62728", label="Deletion (↓ = more faithful)", linewidth=1.8)
    ax.plot(pcts, ins_probs, "s--", color="#2ca02c", label="Insertion (↑ = more faithful)", linewidth=1.8)
    ax.fill_between(pcts, del_probs, alpha=0.08, color="#d62728")
    ax.fill_between(pcts, ins_probs, alpha=0.08, color="#2ca02c")
    ax.set_xlabel("Fraction of features masked / revealed (%)", fontsize=10)
    ax.set_ylabel("Mean predicted class probability", fontsize=10)
    ax.set_title("Deletion and Insertion Faithfulness Curves\n(ExplainSOC SHAP, 200 test alerts)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # annotate AUC values
    del_auc = float(np.trapezoid(del_probs, xs))
    ins_auc = float(np.trapezoid(ins_probs, xs))
    ax.text(0.97, 0.55, f"1-del AUC = {1-del_auc:.3f}\nins AUC = {ins_auc:.3f}",
        ha="right", va="center", fontsize=9, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa"))
    fig.tight_layout()
    fig.savefig("fig2_del_ins_auc.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("fig2_del_ins_auc.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved fig2_del_ins_auc.pdf / .png")

# ── Figure 3: Leakage audit bar chart ────────────────────────────────────────
def fig_leakage_audit():
    fields = ["attack_category", "severity", "alert_type",
              "priority_score", "kill_chain_phase",
              "mitre_tactic", "protocol", "difficulty_level", "source_dataset",
              "flow features\n(baseline, 8)"]
    f1s    = [1.000, 1.000, 1.000, 1.000, 1.000,
              0.698, 0.703, 0.597, 0.197,
              0.922]
    types  = ["rule","rule","rule","rule","rule",
              "derived","observable","metadata","metadata",
              "observable (clean)"]
    type_colors = {"rule": "#d62728", "derived": "#ff7f0e",
                   "observable": "#2ca02c", "metadata": "#9467bd",
                   "observable (clean)": "#1f77b4"}
    colors = [type_colors[t] for t in types]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(fields)), f1s, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0.95, color="#d62728", linewidth=1.2, linestyle="--",
               label="Leak threshold (0.95)")
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels(fields, fontsize=9)
    ax.set_xlabel("Single-field macro-F1 on triage_decision", fontsize=9)
    ax.set_title("Leakage Audit: Single-Field Target-Recovery Test (SALAD, N=40,000)", fontsize=10)
    ax.set_xlim(0, 1.08)
    # annotations
    for i, v in enumerate(f1s):
        label = "LEAK" if v >= 0.95 else f"{v:.3f}"
        color = "#d62728" if v >= 0.95 else "#333333"
        ax.text(v + 0.01, i, label, va="center", fontsize=8, fontweight="bold" if v >= 0.95 else "normal", color=color)
    # legend patches
    legend_patches = [mpatches.Patch(color=type_colors[t], label=t) for t in type_colors]
    legend_patches.append(mpatches.Patch(color="#d62728", label="leak threshold (0.95)", linestyle="--", fill=False))
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("fig3_leakage_audit.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("fig3_leakage_audit.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved fig3_leakage_audit.pdf / .png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../JN8 SALAD_DataSet/SALAD/salad_full.csv")
    ap.add_argument("--cap", type=int, default=40000)
    args = ap.parse_args()
    rng = np.random.RandomState(42)

    print("Loading + training clean model...")
    rows = load_csv(args.data, args.cap)
    X = to_X(rows)
    le = LabelEncoder()
    y = le.fit_transform([str(r.get(TARGET, "")) for r in rows])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    clf.fit(X_tr, y_tr)
    print(f"Model trained. Generating figures...")

    # SHAP on 200 test samples for deletion/insertion figure
    idx = rng.choice(len(X_te), 200, replace=False)
    X_sub = X_te[idx]
    baseline = np.zeros_like(X_sub[:1])
    B = np.tile(baseline[0], (len(X_sub), 1))
    pred_cls = clf.predict(X_sub).astype(int)
    phi = np.zeros((len(X_sub), len(FLOW)))
    for _ in range(20):
        perm = rng.permutation(len(FLOW))
        X_prev = B.copy()
        p_prev = clf.predict_proba(X_prev)
        for fi in perm:
            X_curr = X_prev.copy()
            X_curr[:, fi] = X_sub[:, fi]
            p_curr = clf.predict_proba(X_curr)
            for i in range(len(X_sub)):
                phi[i, fi] += p_curr[i, pred_cls[i]] - p_prev[i, pred_cls[i]]
            X_prev = X_curr; p_prev = p_curr
    phi /= 20

    fig_leakage_audit()
    fig_deletion_insertion(clf, X_sub, phi, baseline)
    fig_explanation_card(clf, X_te, y_te, le, rng)
    print("\nAll figures generated.")

if __name__ == "__main__":
    main()
