from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
import yaml

from mideast_sentinel_analysis import run

st.set_page_config(page_title="Mideast Sentinel Lab", page_icon="🛰️", layout="wide")
st.title("🛰️ Mideast Sentinel Lab")
st.caption("Sentinel-1/2 classification + change detection for configured AOIs")

CFG_PATH = Path("config.yaml")


def load_cfg() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_cfg()

with st.sidebar:
    st.header("Run settings")
    baseline_start = st.date_input("Baseline start", value=pd.to_datetime(cfg["baseline"]["start"]).date())
    baseline_end = st.date_input("Baseline end", value=pd.to_datetime(cfg["baseline"]["end"]).date())
    recent_start = st.date_input("Recent start", value=pd.to_datetime(cfg["recent"]["start"]).date())
    recent_end = st.date_input("Recent end", value=pd.to_datetime(cfg["recent"]["end"]).date())
    cloud_cover_max = st.slider("Max cloud cover (S2)", 0, 100, int(cfg.get("cloud_cover_max", 30)))
    max_items = st.slider("Max scenes per window", 2, 20, int(cfg.get("max_items_per_window", 8)))

    run_btn = st.button("Run analysis", type="primary")

# AOI map
rows = []
for a in cfg["aois"]:
    lon, lat = a["center"]
    rows.append({"name": a["name"], "lon": lon, "lat": lat})

df_aoi = pd.DataFrame(rows)

st.subheader("AOI overview")
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v11",
        initial_view_state=pdk.ViewState(
            latitude=float(df_aoi["lat"].mean()), longitude=float(df_aoi["lon"].mean()), zoom=4
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df_aoi,
                get_position="[lon, lat]",
                get_radius=20000,
                get_fill_color=[255, 140, 0, 180],
                pickable=True,
            )
        ],
        tooltip={"text": "{name}\n({lat}, {lon})"},
    )
)

if run_btn:
    cfg_run = dict(cfg)
    cfg_run["baseline"] = {"start": str(baseline_start), "end": str(baseline_end)}
    cfg_run["recent"] = {"start": str(recent_start), "end": str(recent_end)}
    cfg_run["cloud_cover_max"] = int(cloud_cover_max)
    cfg_run["max_items_per_window"] = int(max_items)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(cfg_run, tf)
        run_cfg = tf.name

    with st.spinner("Running Sentinel analysis... this may take a while."):
        run(run_cfg)

    st.success("Analysis completed.")

out_dir = Path(cfg.get("out_dir", "outputs"))
summary_csv = out_dir / "summary.csv"

st.subheader("Results")
if summary_csv.exists():
    df = pd.read_csv(summary_csv)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No summary yet. Click 'Run analysis'.")

st.subheader("Generated images")
images = sorted(out_dir.glob("*.png")) if out_dir.exists() else []
if not images:
    st.info("No images yet. Run analysis first.")
else:
    cols = st.columns(2)
    for i, img in enumerate(images):
        with cols[i % 2]:
            st.image(str(img), caption=img.name, use_container_width=True)
