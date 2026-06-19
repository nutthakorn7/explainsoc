#!/usr/bin/env python3
"""
borderline_selector.py — P1 of the leakage-free SOC-triage study.

Defines the genuinely-ambiguous (TP/FP-overlap) sampling stratum WITHOUT touching
any label-derived field. A classifier trained on flow features ONLY scores every
alert's P(malicious); the BORDERLINE pool is where flow evidence is ambiguous:
    {P_RF(malicious) in [0.40, 0.60]}  UNION  {RF and GBT disagree at 0.5}

CRITICAL de-confound (methodology audit must-fix #6): a [0.40,0.60] band can be
dominated by the benign base rate (feature poverty) rather than real TP/FP overlap.
So we REPORT the malicious prevalence INSIDE the borderline band vs the overall base
rate. If the band is ~base-rate benign, the "ambiguity" is feature poverty, not a
genuine task -> the script FLAGS it (do not pre-register low IAA on such a stratum).

Features are observable-only (no label-derived, no source/provenance leak).
"""
import csv, argparse, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

FLOW_FEATS = ["src_port","dst_port","flow_duration","total_fwd_packets",
              "total_bwd_packets","flow_bytes_per_sec","alert_count_1h","alert_count_24h"]

def load(path, cap=60000):
    X=[]; y=[]; ids=[]
    with open(path, newline="") as f:
        for i,row in enumerate(csv.DictReader(f)):
            if i>=cap: break
            try:
                X.append([float(row.get(k,0) or 0) for k in FLOW_FEATS])
            except ValueError:
                X.append([0]*len(FLOW_FEATS))
            y.append(1 if str(row.get("is_malicious"))=="True" else 0)
            ids.append(row.get("alert_id"))
    return np.array(X), np.array(y), ids

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="../JN8 SALAD_DataSet/SALAD/salad_full.csv")
    ap.add_argument("--cap", type=int, default=60000)
    args=ap.parse_args()
    X,y,ids=load(args.data,args.cap)
    base=y.mean()
    Xtr,Xte,ytr,yte,_,ite=train_test_split(X,y,range(len(y)),test_size=0.3,random_state=42,stratify=y)
    rf=RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    gb=GradientBoostingClassifier(n_estimators=100,random_state=42).fit(Xtr,ytr)
    p_rf=rf.predict_proba(Xte)[:,1]; p_gb=gb.predict_proba(Xte)[:,1]
    disagree=(p_rf>=0.5)!=(p_gb>=0.5)
    band=(p_rf>=0.40)&(p_rf<=0.60)
    border = band | disagree
    n_te=len(yte)
    print("="*60)
    print(f"BORDERLINE SELECTOR  (n_test={n_te}, overall malicious base rate={base:.3f})")
    print("="*60)
    print(f"  RF AUC-ish (acc) = {(rf.predict(Xte)==yte).mean():.3f}   flow-only classifier")
    print(f"  in [0.40,0.60] band : {band.sum()}  ({band.mean()*100:.1f}%)")
    print(f"  RF/GBT disagree     : {disagree.sum()}  ({disagree.mean()*100:.1f}%)")
    print(f"  BORDERLINE pool     : {border.sum()}  ({border.mean()*100:.1f}%)")
    # --- DE-CONFOUND: is the borderline band genuine TP/FP overlap or benign noise? ---
    if border.sum()>0:
        prev = yte[border].mean()
        print(f"\n  malicious prevalence INSIDE borderline = {prev:.3f}   (vs base rate {base:.3f})")
        # genuine overlap -> prevalence near 0.5 and clearly different from a pure-benign or pure-attack stratum
        if abs(prev-0.5) <= 0.20 and prev > base + 0.10:
            print("  -> ✓ GENUINE OVERLAP: borderline is enriched in mixed TP/FP cases (real ambiguity).")
        elif prev <= base + 0.05:
            print("  -> ⚠ FLAG: borderline ≈ base-rate benign = FEATURE POVERTY, not real ambiguity.")
            print("     Flow features barely separate classes (matches the audit). On this fallback")
            print("     substrate, do NOT pre-register low IAA here as 'intrinsic ambiguity'.")
            print("     The REAL Cyber Defense data (richer context) is where genuine overlap lives.")
        else:
            print("  -> ~ partial: borderline somewhat enriched; inspect on the real substrate.")
    # emit the borderline alert_ids for the S1 sampling stratum
    bids=[ids[ite[i]] for i in range(n_te) if border[i]]
    open("_borderline_ids.txt","w").write("\n".join(map(str,bids)))
    print(f"\n  wrote {len(bids)} borderline alert_ids -> _borderline_ids.txt (S1 stratum)")

if __name__=="__main__":
    main()
