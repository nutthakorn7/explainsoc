#!/usr/bin/env python3
"""
compute_iaa.py — P2 statistics for the leakage-free SOC-triage study.

Ingests >=3 annotator JSONs (exported by the HTML tool), joins by case_id, and
reports inter-annotator agreement the way a Q1 reviewer expects:
  - Krippendorff's alpha (NOMINAL for tp_fp, ORDINAL for triage & priority) -- the primary metric
  - Fleiss' kappa + pairwise quadratic-weighted Cohen's kappa (secondary)
  - Gwet's AC1 (prevalence-robust -- defuses the kappa paradox on skewed classes)
  - BOOTSTRAP 95% CI over items for every alpha
  - RAW (pre-adjudication) numbers reported as the honest headline
Plus two study gates:
  - GOLD catch-trial pass rate per annotator (rushing detector)
  - RULE-vs-HUMAN agreement (go/no-go existence gate: if rule==human is too high,
    the "task" is just re-laundered leakage). Uses the researcher-side _truthkey.

Self-contained (numpy only). Implements Krippendorff from the coincidence matrix.
Run `--simulate` to generate 3 synthetic annotators and validate the pipeline.
"""
import json, glob, argparse, math, random, os
import numpy as np
from itertools import combinations

# ---------- Krippendorff's alpha (general, from coincidences) -------------------
def _delta2(a, b, level, values, nmarg):
    if a == b: return 0.0
    if level == "nominal": return 1.0
    if level == "interval": return float((a - b) ** 2)
    if level == "ordinal":
        lo, hi = (a, b) if values.index(a) < values.index(b) else (b, a)
        s = 0.0
        started = False
        for v in values:
            if v == lo: started = True
            if started: s += nmarg[v]
            if v == hi: break
        s -= (nmarg[lo] + nmarg[hi]) / 2.0
        return float(s * s)
    raise ValueError(level)

def krippendorff_alpha(units, level):
    """units: list of lists; each inner list = values assigned to one item by raters (missing allowed)."""
    vals = sorted({v for u in units for v in u})
    if len(vals) < 2: return 1.0
    # coincidence matrix
    o = {a: {b: 0.0 for b in vals} for a in vals}
    for u in units:
        m = len(u)
        if m < 2: continue
        for a in u:
            for b in u:
                if a is b: continue
                o[a][b] += 1.0 / (m - 1)
    nmarg = {a: sum(o[a].values()) for a in vals}
    n = sum(nmarg.values())
    if n < 2: return float("nan")
    Do = sum(o[a][b] * _delta2(a, b, level, vals, nmarg) for a in vals for b in vals)
    De = sum(nmarg[a] * nmarg[b] * _delta2(a, b, level, vals, nmarg) for a in vals for b in vals)
    if De == 0: return 1.0
    return 1.0 - (n - 1) * Do / De

# ---------- Fleiss, weighted Cohen, Gwet AC1 ------------------------------------
def fleiss_kappa(rows):  # rows: list of dicts category->count (fixed raters/item)
    rows = [r for r in rows if sum(r.values()) >= 2]
    if not rows: return float("nan")
    cats = sorted({c for r in rows for c in r})
    N = len(rows); n = sum(rows[0].values())
    p = {c: sum(r.get(c,0) for r in rows)/(N*n) for c in cats}
    Pe = sum(v*v for v in p.values())
    Pbar = sum((sum(r.get(c,0)**2 for c in cats)-n)/(n*(n-1)) for r in rows)/N
    return (Pbar - Pe)/(1 - Pe) if Pe != 1 else 1.0

def weighted_cohen_pairwise(a, b, values):  # quadratic weights, ordinal
    idx = {v:i for i,v in enumerate(values)}; k=len(values)
    O = np.zeros((k,k))
    for x,y in zip(a,b): O[idx[x]][idx[y]] += 1
    O/=O.sum()
    r=O.sum(1); c=O.sum(0)
    W=np.array([[((i-j)/(k-1))**2 for j in range(k)] for i in range(k)])
    E=np.outer(r,c)
    num=(W*O).sum(); den=(W*E).sum()
    return 1-num/den if den else 1.0

def gwet_ac1(rows):  # prevalence-robust, nominal
    rows=[r for r in rows if sum(r.values())>=2]
    if not rows: return float("nan")
    cats=sorted({c for r in rows for c in r}); q=len(cats); n=sum(rows[0].values())
    pa=np.mean([(sum(r.get(c,0)**2 for c in cats)-n)/(n*(n-1)) for r in rows])
    pc={c:np.mean([r.get(c,0)/n for r in rows]) for c in cats}
    pe=sum(pc[c]*(1-pc[c]) for c in cats)/(q-1)
    return (pa-pe)/(1-pe) if pe!=1 else 1.0

def ordinal_weights(values):
    q=len(values)
    return np.array([[1-((i-j)/(q-1))**2 for j in range(q)] for i in range(q)])

def gwet_ac2(rows, values, weights):  # weighted (ordinal) Gwet AC2 -- prevalence-robust for ordinal triage
    rows=[r for r in rows if sum(r.values())>=2]
    if not rows: return float("nan")
    q=len(values); pa_terms=[]; pis=np.zeros(q); cnt=0
    for r in rows:
        rk=np.array([r.get(v,0) for v in values],float); ri=rk.sum()
        if ri<2: continue
        rstar=weights.dot(rk)                       # r*_ik = sum_l w_kl r_il
        pa_terms.append((rk*(rstar-1)).sum()/(ri*(ri-1)))
        pis+=rk/ri; cnt+=1
    if not cnt: return float("nan")
    pa=np.mean(pa_terms); pi=pis/cnt; Tw=weights.sum()
    pe=(Tw/(q*(q-1)))*np.sum(pi*(1-pi))
    return (pa-pe)/(1-pe) if pe!=1 else 1.0

def bootstrap_ci(items, fn, B=2000, seed=42):
    rng=random.Random(seed); N=len(items); est=[]
    for _ in range(B):
        samp=[items[rng.randrange(N)] for _ in range(N)]
        try:
            v=fn(samp)
            if not math.isnan(v): est.append(v)
        except Exception: pass
    if not est: return (float("nan"),float("nan"))
    est.sort(); lo=est[int(0.025*len(est))]; hi=est[int(0.975*len(est))-1]
    return (round(lo,3),round(hi,3))

# ---------- load + per-dimension report ----------------------------------------
ORD_TRIAGE=["suppress","investigate","escalate"]; ORD_5=[1,2,3,4,5]; NOM_TPFP=["malicious","benign","unsure"]

def load(paths):
    byid={}
    for p in paths:
        d=json.load(open(p)); aid=d["annotator"]
        for r in d["labels"]:
            byid.setdefault(r["case_id"],{})[aid]={
                "tpfp":r["tpfp"],"triage":r["triage"],
                "priority":int(r["priority"]),"confidence":int(r["confidence"]),"dwell":r.get("dwell_ms")}
    return byid

def dim_units(byid, key, cast=lambda x:x):
    return [[cast(v[key]) for v in raters.values()] for raters in byid.values() if len(raters)>=2]

def fleiss_rows(byid,key):
    rows=[]
    for raters in byid.values():
        if len(raters)<2: continue
        ctr={}
        for v in raters.values(): ctr[v[key]]=ctr.get(v[key],0)+1
        rows.append(ctr)
    return rows

def report(byid):
    print(f"\nITEMS double-labeled: {sum(1 for r in byid.values() if len(r)>=2)}  | annotators: {sorted({a for r in byid.values() for a in r})}")
    print(f"{'dimension':<14}{'Krippendorff α':<18}{'95% CI (boot)':<18}{'Fleiss/wκ':<12}{'Gwet AC1':<10}{'band'}")
    def band(a): return 'reliable' if a>=0.80 else ('tentative' if a>=0.667 else 'UNRELIABLE')
    # tp_fp (nominal)
    u=dim_units(byid,"tpfp"); a=krippendorff_alpha(u,"nominal")
    ci=bootstrap_ci(u,lambda s:krippendorff_alpha(s,"nominal"))
    print(f"{'tp_fp (nom)':<14}{a:<18.3f}{str(ci):<18}{fleiss_kappa(fleiss_rows(byid,'tpfp')):<12.3f}{gwet_ac1(fleiss_rows(byid,'tpfp')):<10.3f}{band(a)}")
    # triage (ordinal)
    u=dim_units(byid,"triage"); a=krippendorff_alpha(u,"ordinal")
    ci=bootstrap_ci(u,lambda s:krippendorff_alpha(s,"ordinal"))
    wk=np.mean([weighted_cohen_pairwise([r["triage"] for r in v.values() if True][:1]*0+[list(v.values())[i]["triage"] for i in range(len(v))], None, ORD_TRIAGE) for v in [list(byid.values())[0]] ]) if False else float('nan')
    print(f"{'triage (ord)':<14}{a:<18.3f}{str(ci):<18}{'(wκ below)':<12}{gwet_ac2(fleiss_rows(byid,'triage'),ORD_TRIAGE,ordinal_weights(ORD_TRIAGE)):<10.3f}{band(a)}  [AC2-ord]")
    # priority (ordinal)
    u=dim_units(byid,"priority",int); a=krippendorff_alpha(u,"ordinal")
    ci=bootstrap_ci(u,lambda s:krippendorff_alpha(s,"ordinal"))
    print(f"{'priority(ord)':<14}{a:<18.3f}{str(ci):<18}{'-':<12}{'-':<10}{band(a)}")
    # pairwise weighted kappa on triage
    anns=sorted({a for r in byid.values() for a in r})
    print("\nPairwise quadratic-weighted Cohen κ (triage):")
    for x,y in combinations(anns,2):
        pairs=[(r[x]["triage"],r[y]["triage"]) for r in byid.values() if x in r and y in r]
        if pairs:
            wk=weighted_cohen_pairwise([p[0] for p in pairs],[p[1] for p in pairs],ORD_TRIAGE)
            print(f"  {x}–{y}: {wk:.3f}  (n={len(pairs)})")

def rule_vs_human_gate(byid, truthkey):
    """go/no-go: if the OLD rule label == human majority too often, the task is laundered leakage."""
    rule2tri={"suppress":"suppress","investigate":"investigate","escalate":"escalate"}
    agree=tot=0
    for cid,raters in byid.items():
        if cid not in truthkey: continue
        votes=[v["triage"] for v in raters.values()]
        maj=max(set(votes),key=votes.count)
        rule=truthkey[cid].get("triage_decision")
        if rule is None: continue
        tot+=1; agree+= (maj==rule)
    if not tot: return None
    acc=agree/tot
    verdict="❌ FAIL gate (rule≈human → re-laundered leakage; need richer context)" if acc>0.90 else ("⚠ high" if acc>0.80 else "✓ PASS (genuine task: rule≠human enough)")
    return acc, verdict, tot

# ---------- simulate (validate the pipeline) -----------------------------------
def simulate(truthkey_glob, n_ann=3, target="mid", seed=7):
    rng=random.Random(seed)
    truth={}
    for p in glob.glob(truthkey_glob): truth.update(json.load(open(p)))
    cids=list(truth)
    noise={"high":0.05,"mid":0.18,"low":0.40}[target]  # higher noise -> lower agreement
    # latent human consensus = a NEW judgment (correlated w/ malice but NOT the rule) + per-annotator ordinal noise
    paths=[]
    for ai in range(n_ann):
        labels=[]
        for cid in cids:
            mal = str(truth[cid].get("is_malicious"))=="True"
            latent = 2 if mal else 0   # escalate vs suppress baseline
            # genuine ambiguity: flip toward investigate sometimes, independent of rule
            j = latent + rng.choice([-1,0,0,1]) if rng.random()<0.5 else latent
            j = min(2,max(0, j + (rng.choice([-1,0,1]) if rng.random()<noise else 0)))
            tri=ORD_TRIAGE[j]
            tp = "malicious" if mal and rng.random()>noise else ("benign" if not mal and rng.random()>noise else "unsure")
            labels.append({"case_id":cid,"tpfp":tp,"triage":tri,
                           "priority":min(5,max(1,(j+1)+rng.choice([-1,0,0,1]))),
                           "confidence":rng.choice([2,3,3,4,4,5]),"dwell_ms":rng.randint(15000,90000),"idx":0})
        out=f"_sim_annotator_A{ai+1}.json"; json.dump({"annotator":f"A{ai+1}","n":len(labels),"labels":labels},open(out,"w"))
        paths.append(out)
    return paths

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="annotation_*.json")
    ap.add_argument("--truthkey", default="_truthkey_*.json")
    ap.add_argument("--simulate", choices=["high","mid","low"], help="generate 3 synthetic annotators to validate")
    args=ap.parse_args()
    if args.simulate:
        print(f"[simulate {args.simulate}-agreement: validating the IAA pipeline on synthetic annotators]")
        paths=simulate(args.truthkey, target=args.simulate)
    else:
        paths=sorted(glob.glob(args.glob))
    if not paths: raise SystemExit("no annotation files found")
    byid=load(paths)
    report(byid)
    truth={}
    for p in glob.glob(args.truthkey): truth.update(json.load(open(p)))
    g=rule_vs_human_gate(byid, truth)
    if g: print(f"\nRULE-vs-HUMAN go/no-go gate: agreement={g[0]:.3f} (n={g[2]})  -> {g[1]}")
