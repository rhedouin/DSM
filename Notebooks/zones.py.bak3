"""
zones.py
========
Catalogue of geographic zones for DSM/DTM dataset collection.

Each zone is a ~10 km × 10 km bounding box (box_size = 0.05°, i.e. ±0.05° from
centre) selected to maximise diversity across:
  - Terrain type   : flat plain / rolling hills / Alpine valley / dense forest
  - Land use       : urban / periurban / agricultural / forest
  - Climate        : Atlantic / Continental / Mediterranean
  - Elevation band : 0–100 m / 100–400 m / 400–2000 m

All 5 zones are confirmed to have Lidar HD IGN coverage as of 2024–2025.

Usage
-----
    from zones import ZONES, ZONES_BY_ID, get_bbox

    for zone in ZONES:
        bbox = get_bbox(zone)
        ...

Geographic train / validation / test split
------------------------------------------
    TRAIN : caen, grenoble, landes
    VAL   : paris_sud
    TEST  : toulouse

The val/test zones are geographically distant from train zones to prevent
spatial leakage.
"""

from __future__ import annotations
from typing import TypedDict


class ZoneSpec(TypedDict):
    id: str          # Short slug — used as folder name and file prefix
    name: str        # Human-readable label
    lon: float       # Centre longitude (WGS84 decimal degrees)
    lat: float       # Centre latitude  (WGS84 decimal degrees)
    box_size: float  # Half-width of bounding box in degrees (default 0.05)
    terrain: str     # Brief terrain description
    landuse: str     # Dominant land use
    split: str       # "train", "val", or "test"
    notes: str       # Special remarks


# ---------------------------------------------------------------------------
# Zone catalogue
# ---------------------------------------------------------------------------
# Bounding box = [lon - box_size, lat - box_size, lon + box_size, lat + box_size]
# At 49°N: 0.05° lon ≈ 3.3 km, 0.05° lat ≈ 5.5 km  →  box ~6.6 × 11 km
# At 43°N: 0.05° lon ≈ 3.7 km                        →  box ~7.4 × 11 km
# ---------------------------------------------------------------------------
ZONES: list[ZoneSpec] = [
    dict(
        id       = "caen",
        name     = "Caen (Normandie)",
        lon      = -0.37,
        lat      =  49.18,
        box_size =  0.05,
        terrain  = "rolling bocage hills, 10–120 m",
        landuse  = "periurban / farmland / hedgerow",
        split    = "train",
        notes    = "Baseline zone — Lidar HD acq. 2023-02-15 confirmed",
    ),
    dict(
        id       = "grenoble",
        name     = "Grenoble (Isère, French Alps)",
        lon      =  5.72,
        lat      = 45.17,
        box_size =  0.05,
        terrain  = "steep Alpine valley, 200–800 m within bbox",
        landuse  = "dense urban core + forested slopes",
        split    = "train",
        notes    = "Extreme relief — tests model on high-gradient terrain",
    ),
    dict(
        id       = "landes",
        name     = "Landes de Gascogne (Gironde)",
        lon      = -0.85,
        lat      =  44.30,
        box_size =  0.05,
        terrain  = "flat coastal plain, 20–60 m",
        landuse  = "dense planted maritime pine forest",
        split    = "train",
        notes    = "Dense canopy creates large DSM–DTM gap; key test case for "
                   "forest canopy removal",
    ),
    dict(
        id       = "paris_sud",
        name     = "Paris-Sud / Essonne",
        lon      =  2.30,
        lat      = 48.75,
        box_size =  0.05,
        terrain  = "flat to gently undulating plateau, 50–100 m",
        landuse  = "suburban residential / parks / farmland",
        split    = "val",
        notes    = "High building-density heterogeneity; validation zone",
    ),
    dict(
        id       = "toulouse",
        name     = "Toulouse (Haute-Garonne)",
        lon      =  1.45,
        lat      = 43.60,
        box_size =  0.05,
        terrain  = "Garonne alluvial plain, 130–160 m",
        landuse  = "urban / periurban / floodplain agriculture",
        split    = "test",
        notes    = "Mediterranean–Atlantic transition, flat — held-out test zone",
    ),
]

# Lookup dict  {zone_id: ZoneSpec}
ZONES_BY_ID: dict[str, ZoneSpec] = {z["id"]: z for z in ZONES}

# Pre-defined splits
TRAIN_ZONES = [z["id"] for z in ZONES if z["split"] == "train"]
VAL_ZONES   = [z["id"] for z in ZONES if z["split"] == "val"]
TEST_ZONES  = [z["id"] for z in ZONES if z["split"] == "test"]


def get_bbox(zone: ZoneSpec) -> list[float]:
    """Return [lon_min, lat_min, lon_max, lat_max] for a zone."""
    hs = zone["box_size"]
    return [
        zone["lon"] - hs,
        zone["lat"] - hs,
        zone["lon"] + hs,
        zone["lat"] + hs,
    ]


if __name__ == "__main__":
    print(f"{'ID':15s} {'Name':35s} {'Split':6s} {'Terrain'}")
    print("-" * 90)
    for z in ZONES:
        bb = get_bbox(z)
        print(f"{z['id']:15s} {z['name']:35s} {z['split']:6s} {z['terrain']}")
    print(f"\nTrain : {TRAIN_ZONES}")
    print(f"Val   : {VAL_ZONES}")
    print(f"Test  : {TEST_ZONES}")
