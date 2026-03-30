# mideast Sentinel-1/2 Time-Series Pipeline

This project downloads **Sentinel-2 (optical)** and **Sentinel-1 (SAR)** imagery for Middle East AOIs, then performs:
- feature extraction,
- unsupervised classification,
- baseline vs recent change detection,
- export of PNG visualizations and summary tables.

## What this does

For each AOI in `config.yaml`:
1. Query Planetary Computer STAC (`sentinel-2-l2a`, `sentinel-1-rtc`)
2. Build median composites for baseline and recent windows
3. Compute features:
   - S2: NDVI, NBR, NDBI
   - S1: VV(dB), VH(dB), VV-VH ratio(dB)
4. Run KMeans classification (recent composite)
5. Compute change metrics (`delta_ndvi`, `delta_nbr`, `delta_vv_db`)
6. Save maps and summary stats in `outputs/`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python mideast_sentinel_analysis.py
```

Outputs:
- `outputs/*_s2_baseline.png`
- `outputs/*_s2_recent.png`
- `outputs/*_s1_baseline.png`
- `outputs/*_s1_recent.png`
- `outputs/*_change.png`
- `outputs/summary.csv`
- `outputs/summary.json`

## AOIs and dates

Edit `config.yaml`:
- `aois`: hotspot definitions
- `baseline` and `recent`: time windows
- `cloud_cover_max`
- `max_items_per_window`

## Notes on war/event matching

This pipeline is geospatial/remote-sensing based. For matching to current war reports:
- ingest trusted event feeds manually (e.g., timestamped incident tables),
- map those events to AOIs and dates,
- compare with change metrics in `summary.csv`.

## Caution

Change signals are not proof of a specific military event by themselves.
Always validate against independent reporting and expert review.
