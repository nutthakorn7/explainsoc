# Data

## SALAD (public substrate)

SALAD is derived from CICIDS2017 and UNSW-NB15. Download from:
- TODO: add arXiv/Zenodo DOI once published

Place the downloaded files here:
```
data/
├── salad_full.csv     (2,778,424 rows, 29 features)
├── salad_train.csv
├── salad_test.csv
└── salad_val.csv
```

## Operational SOC data (not shared)

The operational alerts from Cyber Defense Co., Ltd. are not shared due to PDPA.
The anonymized annotation schema is in `operational_schema.md`.

## operational_schema.md

The annotation interface (annotation_A1.html) presents analysts with evidence cards
containing these fields (no rule-derived fields included):

- `case_id`: anonymized alert identifier
- `src_port`, `dst_port`: source and destination ports
- `flow_duration`: flow duration in seconds
- `flow_bytes_per_sec`: bytes per second
- `total_fwd_packets`, `total_bwd_packets`: packet counts
- `alert_count_1h`, `alert_count_24h`: temporal context

Analysts label each alert with:
- `tpfp`: malicious / benign / unsure
- `triage`: suppress / investigate / escalate
- `priority`: 1–5
- `confidence`: 1–5
