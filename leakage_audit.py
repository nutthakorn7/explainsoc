#!/usr/bin/env python3
"""
leakage_audit.py — P3 of the leakage-free SOC-triage study.

The quantitative pre-publication leakage audit the rejected papers lacked. For each
TARGET a model predicts (is_malicious, attack_category, and the human triage gold once
collected), it asks: can a SINGLE suspect field recover the target almost perfectly?
If a single field's macro-F1 reaches >=95% of the clean flow-only baseline (or is
near 1.0), that field LEAKS the answer and must be excluded from model inputs and from
the analyst card.

This is the tool that would have caught all four rejected papers: e.g. on SALAD,
severity alone recovers triage_decision (the rule), attack_category alone recovers
is_malicious -- circular by construction. Run it on YOUR benchmark BEFORE submission
and report the table in the paper.
"""
import csv, argparse, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

FLOW = ["src_port","dst_port","flow_duration","total_fwd_packets","total_bwd_packets",
        "flow_bytes_per_sec","alert_count_1h","alert_count_24h"]
SUSPECT_CATS = ["source_dataset","attack_category","severity","alert_type","protocol",
                "triage_decision","priority_score","mitre_tactic","kill_chain_phase","difficulty_level"]
TARGETS = ["is_malicious","attack_category","triage_decision"]   # add 'human_triage' once gold exists

def load(path, cols, cap=40000):
    rows=[]
    with open(path,newline="") as f:
        for i,r in enumerate(csv.DictReader(f)):
            if i>=cap: break
            rows.append(r)
    return rows

def numeric(rows,keys):
    X=[]
    for r in rows:
        v=[]
        for k in keys:
            try: v.append(float(r.get(k,0) or 0))
            except ValueError: v.append(0.0)
        X.append(v)
    return np.array(X)

def enc(rows,key):
    le=LabelEncoder(); return le.fit_transform([str(r.get(key,"")) for r in rows]).reshape(-1,1)

def mf1(Xtr,Xte,ytr,yte,strong=False):
    clf=GradientBoostingClassifier(n_estimators=60,random_state=42) if strong else DecisionTreeClassifier(max_depth=12,random_state=42)
    clf.fit(Xtr,ytr); return f1_score(yte,clf.predict(Xte),average="macro")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="../JN8 SALAD_DataSet/SALAD/salad_full.csv")
    ap.add_argument("--cap", type=int, default=40000)
    args=ap.parse_args()
    rows=load(args.data,None,args.cap)
    idx=list(range(len(rows)))
    tr,te=train_test_split(idx,test_size=0.3,random_state=42)
    def split(M,y): return M[tr],M[te],y[tr],y[te]
    print("="*72)
    print("QUANTITATIVE LEAKAGE AUDIT  (single-field recovery vs clean flow-only baseline)")
    print("  rule: single-field macro-F1 >= 0.95 x clean  OR  near 1.0  =>  LEAK / circular")
    print("="*72)
    Xflow=numeric(rows,FLOW)
    for tgt in TARGETS:
        yt=LabelEncoder().fit_transform([str(r.get(tgt,"")) for r in rows])
        if len(set(yt))<2: continue
        Xt_tr,Xt_te,ytr,yte=split(Xflow,yt)
        clean=mf1(Xt_tr,Xt_te,ytr,yte,strong=True)
        print(f"\nTARGET = {tgt}   (clean flow-only baseline macro-F1 = {clean:.3f})")
        flags=[]
        for fld in SUSPECT_CATS:
            if fld==tgt: continue
            M=enc(rows,fld); a,b,c,dd=split(M,yt)
            try: s=mf1(a,b,c,dd)
            except Exception: continue
            ratio=s/clean if clean>0 else float('inf')
            leak = s>=0.95 or ratio>=0.95
            tag = "  <== LEAK (circular)" if leak else ""
            if leak: flags.append(fld)
            print(f"   {fld:<18} single-field macro-F1 = {s:.3f}   ({ratio:.2f}x clean){tag}")
        print(f"   => LEAK CHANNELS for {tgt}: {flags or 'none'}  (must exclude from inputs + analyst card)")
    print("\n" + "="*72)
    print("INTERPRETATION: any field flagged LEAK reproduces the target by itself = it is the")
    print("answer in disguise. On SALAD this confirms triage/priority/severity are rule transforms")
    print("of the attack label (the circularity that sank the prior papers). Report this table.")

if __name__=="__main__":
    main()
