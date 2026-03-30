from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import DBSCAN

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class Hotspot:
    name: str
    lon: float
    lat: float
    weight: int


def _gdelt_query(mode: str, query: str, maxrecords: int = 250) -> dict:
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "maxrecords": str(maxrecords),
    }
    r = requests.get(GDELT_DOC_API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_news_points(days_back: int = 14) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back)
    # Focus terms can be tuned.
    q = f'(iran OR israel) AND (missile OR strike OR explosion OR drone OR conflict) AND sourcecountry:IR'

    # GDELT DOC article mode with social shares; we then use timelinegeo as geohints.
    _ = _gdelt_query("artlist", q, maxrecords=250)
    geo = _gdelt_query("timelinegeo", q, maxrecords=250)

    rows = []
    for g in geo.get("timeline", []):
        date = g.get("date")
        for s in g.get("series", []):
            name = s.get("name", "")
            value = int(s.get("value", 0))
            # name usually: "lat,lon"
            if not name or "," not in name:
                continue
            try:
                lat_s, lon_s = name.split(",", 1)
                lat = float(lat_s)
                lon = float(lon_s)
            except Exception:
                continue
            # loose regional filter for Middle East/Iran neighborhood
            if not (20 <= lat <= 45 and 35 <= lon <= 65):
                continue
            rows.append({"date": date, "lat": lat, "lon": lon, "weight": value})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # keep recent-ish only if date parseable
    def _ok(d):
        try:
            dt = datetime.strptime(d, "%Y%m%d").replace(tzinfo=timezone.utc)
            return dt >= since
        except Exception:
            return True

    df = df[df["date"].apply(_ok)].reset_index(drop=True)
    return df


def cluster_hotspots(df: pd.DataFrame, top_k: int = 6) -> List[Hotspot]:
    if df.empty:
        return []

    X = df[["lon", "lat"]].values
    # approx ~20km scale in deg
    db = DBSCAN(eps=0.20, min_samples=2)
    labels = db.fit_predict(X)
    df = df.copy()
    df["cluster"] = labels

    out: List[Hotspot] = []
    for c in sorted([x for x in df["cluster"].unique() if x >= 0]):
        d = df[df["cluster"] == c]
        w = int(d["weight"].sum())
        lon = float(np.average(d["lon"], weights=d["weight"].clip(lower=1)))
        lat = float(np.average(d["lat"], weights=d["weight"].clip(lower=1)))
        out.append(Hotspot(name=f"news_hotspot_{c}", lon=lon, lat=lat, weight=w))

    out.sort(key=lambda h: h.weight, reverse=True)
    return out[:top_k]


def auto_hotspots(days_back: int = 14, top_k: int = 6) -> List[Hotspot]:
    df = fetch_news_points(days_back=days_back)
    return cluster_hotspots(df, top_k=top_k)
