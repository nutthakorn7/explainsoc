#!/usr/bin/env python3
"""
blinding_generator.py — P0 of the leakage-free SOC-triage study.

Takes a raw alert record (SALAD schema now; real Cyber Defense SOC export later)
and renders a NEUTRAL analyst card that shows ONLY observable evidence a T1
analyst sees, with every label-derived and provenance field removed. The goal:
a human (or a model) MUST NOT be able to recover the verdict or the source
dataset from the card.

De-leak rules (grounded in the methodology audit of SALAD/UNSW-CICIDS):
  HIDE  (label-derived / verdict):  severity, confidence, alert_type,
        mitre_tactic, mitre_technique, kill_chain_phase, attack_category,
        is_malicious, triage_decision, priority_score, difficulty_level,
        alert_description, correlation_key
  HIDE  (provenance leaks source -> attack mix):  source_dataset, timestamp
  TRANSFORM (audit found these SHOWN fields still leak):
        - protocol  : non-tcp => 100% UNSW + 80.6% malicious  -> collapse to {tcp, other}
        - dst_ip/src_ip : first octet leaks source (192->CICIDS, 10->UNSW) -> fresh random per alert
        - network_segment : derived from dst-IP 2nd octet           -> DROP
  KEEP (observable telemetry):  src_port, dst_port, flow_duration,
        total_fwd_packets, total_bwd_packets, flow_bytes_per_sec,
        alert_count_1h, alert_count_24h, is_repeated_target
        (alert_count_* are ~0 and is_repeated_target ~99.9% True in SALAD ->
         flagged DEGENERATE; kept for the real-data version where they carry signal)
"""
import csv, json, hashlib, random, argparse, sys

HIDE = {
    "severity","confidence","alert_type","mitre_tactic","mitre_technique",
    "kill_chain_phase","attack_category","is_malicious","triage_decision",
    "priority_score","difficulty_level","alert_description","correlation_key",
    "source_dataset","timestamp","network_segment",
}
KEEP_NUMERIC = [
    "src_port","dst_port","flow_duration","total_fwd_packets",
    "total_bwd_packets","flow_bytes_per_sec","alert_count_1h","alert_count_24h",
]
DEGENERATE = {"alert_count_1h","alert_count_24h","is_repeated_target"}  # near-constant in SALAD v1

def _rand_ip(seed_key, salt):
    """Deterministic-but-source-independent fake IP (stable per alert via salt, but
    octet distribution carries NO source signal)."""
    h = hashlib.sha256(f"{salt}|{seed_key}".encode()).digest()
    # avoid 10.x / 192.x patterns that leak source: map first octet into 11..223 excl. 10,192,172
    bad = {10,127,172,192}
    o1 = 11 + (h[0] % 200)
    if o1 in bad: o1 = (o1 + 7)
    return f"{o1}.{h[1]}.{h[2]}.{h[3]%254+1}"

def neutralize(row, salt="STUDY_SALT_2026"):
    """Return (card_fields dict, observable-only) from a raw alert row."""
    aid = row.get("alert_id","NA")
    proto = (row.get("protocol","") or "").strip().lower()
    card = {
        "case_id": hashlib.sha256(f"{salt}|{aid}".encode()).hexdigest()[:12],
        "src_ip": _rand_ip(aid+"|s", salt),
        "dst_ip": _rand_ip(aid+"|d", salt),
        "protocol": "tcp" if proto == "tcp" else "other",   # collapse: kills source/verdict proxy
    }
    for k in KEEP_NUMERIC:
        v = row.get(k, "")
        card[k] = v
    card["is_repeated_target"] = row.get("is_repeated_target","")
    # degeneracy note for transparency (real-data version will differ)
    card["_degenerate_fields_note"] = [k for k in DEGENERATE if str(row.get(k,"")).strip() in ("0","0.0","")] or "n/a"
    return card

def render_text(card):
    """Neutral SIEM-style card text -- NO verdict words, NO attack names."""
    L = []
    L.append(f"=== Network Flow Alert  [case {card['case_id']}] ===")
    L.append(f"  {card['src_ip']}:{card.get('src_port','?')}  ->  {card['dst_ip']}:{card.get('dst_port','?')}  ({card['protocol']})")
    L.append(f"  duration={card.get('flow_duration','?')}s  fwd_pkts={card.get('total_fwd_packets','?')}  bwd_pkts={card.get('total_bwd_packets','?')}  bytes/s={card.get('flow_bytes_per_sec','?')}")
    L.append(f"  recent alerts from src: 1h={card.get('alert_count_1h','?')}  24h={card.get('alert_count_24h','?')}  repeated_target={card.get('is_repeated_target','?')}")
    L.append( "  -> Your call: malicious / benign / unsure | triage: suppress/investigate/escalate | priority 1-5 | confidence 1-5")
    return "\n".join(L)

def leak_self_check(rows, n=4000):
    """Sanity: can a strong feature recover the HIDDEN verdict/source from the SHOWN card?
    If recovery is high, the card still leaks. (Quick chi-style proxy: does protocol or any
    shown field perfectly separate is_malicious / source_dataset?)"""
    from collections import Counter, defaultdict
    shown_proto = defaultdict(Counter)   # card protocol -> is_malicious
    shown_proto_src = defaultdict(Counter)
    for r in rows[:n]:
        c = neutralize(r)
        shown_proto[c["protocol"]][r.get("is_malicious","?")] += 1
        shown_proto_src[c["protocol"]][r.get("source_dataset","?")] += 1
    def purity(d):
        worst = 0.0
        for k,ctr in d.items():
            tot=sum(ctr.values()); worst=max(worst, max(ctr.values())/tot if tot else 0)
        return worst
    return {"protocol->malicious worst-purity": round(purity(shown_proto),3),
            "protocol->source worst-purity":    round(purity(shown_proto_src),3),
            "note": "want these near base-rate (~0.79 mal / ~0.91 src), NOT ~1.0 (which = leak)"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../JN8 SALAD_DataSet/SALAD/salad_full.csv")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()
    rows=[]
    with open(args.data, newline="") as f:
        r=csv.DictReader(f)
        for i,row in enumerate(r):
            rows.append(row)
            if i>=8000: break
    # show diverse samples (1 benign, 1 attack, 1 random)
    benign=[x for x in rows if x.get("is_malicious")=="False"][:1]
    attack=[x for x in rows if x.get("is_malicious")=="True"][:1]
    picks = benign+attack+rows[100:100+args.samples]
    print("####### SAMPLE NEUTRAL CARDS (what the analyst sees) #######\n")
    for row in picks[:args.samples+2]:
        hidden = {k:row.get(k) for k in ("attack_category","severity","triage_decision","is_malicious","source_dataset")}
        print(render_text(neutralize(row)))
        print(f"   [HIDDEN from analyst -> {hidden}]\n")
    print("####### LEAK SELF-CHECK #######")
    print(json.dumps(leak_self_check(rows), indent=2, ensure_ascii=False))
