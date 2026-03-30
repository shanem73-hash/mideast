from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import xarray as xr
import yaml
from odc.stac import load
from pystac_client import Client
from sklearn.cluster import KMeans

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass
class AOI:
    name: str
    center: Tuple[float, float]
    half_size_deg: float

    @property
    def bbox(self) -> List[float]:
        lon, lat = self.center
        h = self.half_size_deg
        return [lon - h, lat - h, lon + h, lat + h]


def read_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def search_items(collection: str, bbox: List[float], start: str, end: str, max_items: int, cloud_cover_max: int | None = None):
    catalog = Client.open(PC_STAC)
    query = {}
    if cloud_cover_max is not None:
        query["eo:cloud_cover"] = {"lt": cloud_cover_max}

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query=query if query else None,
    )
    items = list(search.get_items())[:max_items]
    return [planetary_computer.sign(item) for item in items]


def load_s2_composite(items, bbox: List[float]) -> xr.Dataset | None:
    if not items:
        return None
    ds = load(items, bands=["B04", "B08", "B12"], bbox=bbox, crs="EPSG:4326", resolution=0.0002)
    if ds.time.size == 0:
        return None
    return ds.median(dim="time", skipna=True)


def load_s1_composite(items, bbox: List[float]) -> xr.Dataset | None:
    if not items:
        return None
    ds = load(items, bands=["vv", "vh"], bbox=bbox, crs="EPSG:4326", resolution=0.0002)
    if ds.time.size == 0:
        return None
    return ds.median(dim="time", skipna=True)


def safe_div(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    return xr.where(np.abs(b) < 1e-8, np.nan, a / b)


def s2_features(comp: xr.Dataset) -> Dict[str, xr.DataArray]:
    red = comp["B04"].astype("float32")
    nir = comp["B08"].astype("float32")
    swir = comp["B12"].astype("float32")
    ndvi = safe_div(nir - red, nir + red)
    nbr = safe_div(nir - swir, nir + swir)
    ndbi = safe_div(swir - nir, swir + nir)
    return {"ndvi": ndvi, "nbr": nbr, "ndbi": ndbi}


def s1_features(comp: xr.Dataset) -> Dict[str, xr.DataArray]:
    vv = comp["vv"].astype("float32")
    vh = comp["vh"].astype("float32")
    vv_db = 10.0 * np.log10(vv.clip(min=1e-6))
    vh_db = 10.0 * np.log10(vh.clip(min=1e-6))
    ratio_db = vv_db - vh_db
    return {"vv_db": vv_db, "vh_db": vh_db, "ratio_db": ratio_db}


def classify_kmeans(feature_arrays: List[xr.DataArray], n_clusters: int = 4) -> xr.DataArray:
    stacked = np.stack([fa.values for fa in feature_arrays], axis=-1)
    h, w, c = stacked.shape
    X = stacked.reshape(-1, c)
    valid = np.isfinite(X).all(axis=1)

    labels = np.full((X.shape[0],), -1, dtype=np.int16)
    if valid.sum() > n_clusters * 20:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels_valid = km.fit_predict(X[valid])
        labels[valid] = labels_valid

    arr = labels.reshape(h, w)
    return xr.DataArray(arr, coords=feature_arrays[0].coords, dims=feature_arrays[0].dims, name="class")


def diff_stats(a: xr.DataArray, b: xr.DataArray) -> dict:
    d = (b - a).values
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"mean": None, "std": None, "p95": None, "p99": None}
    return {
        "mean": float(np.mean(d)),
        "std": float(np.std(d)),
        "p95": float(np.percentile(d, 95)),
        "p99": float(np.percentile(d, 99)),
    }


def export_map_points(out_csv: Path, aoi_name: str, metric_name: str, da: xr.DataArray, stride: int = 6):
    if "x" not in da.coords or "y" not in da.coords:
        return
    arr = da.values
    ys = da["y"].values
    xs = da["x"].values

    rows = []
    for iy in range(0, arr.shape[0], stride):
        for ix in range(0, arr.shape[1], stride):
            v = arr[iy, ix]
            if not np.isfinite(v):
                continue
            rows.append(
                {
                    "aoi": aoi_name,
                    "metric": metric_name,
                    "lon": float(xs[ix]),
                    "lat": float(ys[iy]),
                    "value": float(v),
                }
            )

    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)


def plot_panel(out_png: Path, title: str, layers: Dict[str, xr.DataArray]):
    n = len(layers)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
    if n == 1:
        axs = [axs]
    for ax, (k, da) in zip(axs, layers.items()):
        im = ax.imshow(da.values, cmap="viridis")
        ax.set_title(k)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(title)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run(config_path: str = "config.yaml"):
    cfg = read_config(config_path)
    out_dir = Path(cfg.get("out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = cfg["baseline"]
    recent = cfg["recent"]
    c = cfg["collections"]
    cloud_max = int(cfg.get("cloud_cover_max", 30))
    max_items = int(cfg.get("max_items_per_window", 8))

    records = []

    for a in cfg["aois"]:
        aoi = AOI(a["name"], tuple(a["center"]), float(a["half_size_deg"]))
        bbox = aoi.bbox

        s2_b = search_items(c["sentinel2"], bbox, baseline["start"], baseline["end"], max_items, cloud_max)
        s2_r = search_items(c["sentinel2"], bbox, recent["start"], recent["end"], max_items, cloud_max)
        s1_b = search_items(c["sentinel1"], bbox, baseline["start"], baseline["end"], max_items)
        s1_r = search_items(c["sentinel1"], bbox, recent["start"], recent["end"], max_items)

        s2b = load_s2_composite(s2_b, bbox)
        s2r = load_s2_composite(s2_r, bbox)
        s1b = load_s1_composite(s1_b, bbox)
        s1r = load_s1_composite(s1_r, bbox)

        if any(x is None for x in [s2b, s2r, s1b, s1r]):
            records.append({"aoi": aoi.name, "status": "missing_data"})
            continue

        f2b, f2r = s2_features(s2b), s2_features(s2r)
        f1b, f1r = s1_features(s1b), s1_features(s1r)

        class_map = classify_kmeans([f2r["ndvi"], f2r["nbr"], f2r["ndbi"], f1r["vv_db"], f1r["ratio_db"]], n_clusters=5)

        delta_ndvi = f2r["ndvi"] - f2b["ndvi"]
        delta_nbr = f2r["nbr"] - f2b["nbr"]
        delta_vv = f1r["vv_db"] - f1b["vv_db"]

        plot_panel(out_dir / f"{aoi.name}_s2_baseline.png", f"{aoi.name} Sentinel-2 baseline", f2b)
        plot_panel(out_dir / f"{aoi.name}_s2_recent.png", f"{aoi.name} Sentinel-2 recent", f2r)
        plot_panel(out_dir / f"{aoi.name}_s1_baseline.png", f"{aoi.name} Sentinel-1 baseline", f1b)
        plot_panel(out_dir / f"{aoi.name}_s1_recent.png", f"{aoi.name} Sentinel-1 recent", f1r)
        plot_panel(
            out_dir / f"{aoi.name}_change.png",
            f"{aoi.name} change metrics",
            {"delta_ndvi": delta_ndvi, "delta_nbr": delta_nbr, "delta_vv_db": delta_vv, "class": class_map},
        )

        map_dir = out_dir / "map_points"
        map_dir.mkdir(parents=True, exist_ok=True)
        export_map_points(map_dir / f"{aoi.name}_delta_ndvi.csv", aoi.name, "delta_ndvi", delta_ndvi)
        export_map_points(map_dir / f"{aoi.name}_delta_nbr.csv", aoi.name, "delta_nbr", delta_nbr)
        export_map_points(map_dir / f"{aoi.name}_delta_vv_db.csv", aoi.name, "delta_vv_db", delta_vv)

        rec = {
            "aoi": aoi.name,
            "status": "ok",
            "items_s2_baseline": len(s2_b),
            "items_s2_recent": len(s2_r),
            "items_s1_baseline": len(s1_b),
            "items_s1_recent": len(s1_r),
            "delta_ndvi": diff_stats(f2b["ndvi"], f2r["ndvi"]),
            "delta_nbr": diff_stats(f2b["nbr"], f2r["nbr"]),
            "delta_vv_db": diff_stats(f1b["vv_db"], f1r["vv_db"]),
        }
        records.append(rec)

    summary = pd.DataFrame(records)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Done. Wrote results to: {out_dir.resolve()}")


if __name__ == "__main__":
    run("config.yaml")
