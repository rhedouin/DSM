"""
0_download_data.py
==================
Downloads all raw input data and the Lidar HD DTM target for every zone
defined in zones.py (or a user-specified subset).

For each zone the following files are saved under  data/zones/{zone_id}/:
  sentinel2.tif    -- bands B02, B03, B04, B08 + NDVI  (10 m, via AWS STAC)
  copdem30.tif     -- Copernicus DEM 30 m               (via AWS STAC)
  tcd.tif          -- Tree Cover Density  (WMS RGBA render, ~10 m)
  imd.tif          -- Imperviousness Degree (WMS RGBA render, ~10 m)
  dtm_lidar.tif    -- High-resolution Lidar HD DTM ~50 cm (WMS, TARGET)

Temporal alignment strategy
---------------------------
  HRL_VINTAGE_YEAR = 2015  -- TCD/IMD vintage closest to COP-DEM era (~2011-2015)
  S2_ANCHOR_YEAR   = 2017  -- Sentinel-2 L2A systematically available on AWS from 2017

Usage
-----
  python 0_download_data.py                         # download all zones
  python 0_download_data.py --zones caen grenoble   # specific zones only
  python 0_download_data.py --force                 # re-download existing files
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
import rasterio
from pystac_client import Client
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

# Allow anonymous access to public S3 COGs
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

# ---------------------------------------------------------------------------
# 1.  Global parameters (same for all zones)
# ---------------------------------------------------------------------------

# Temporal alignment
# Available IGN Geoplateforme TCD/IMD vintages: 2006, 2009, 2012, 2015, 2018
# COP-DEM covers ~2011-2015, so 2015 is the best temporal match.
HRL_VINTAGE_YEAR      = 2015
S2_ANCHOR_YEAR        = 2017   # Sentinel-2 L2A available on AWS from ~March 2017
S2_SEARCH_WINDOW_DAYS = 365    # search +/- N days around 1 July of S2_ANCHOR_YEAR
MAX_CLOUD_COVER       = 20     # %

# WMS / STAC endpoints
STAC_AWS = "https://earth-search.aws.element84.com/v1"
WMS_IGN  = "https://data.geopf.fr/wms-r/wms"

# WMS layer names
_HRL_SUFFIX = f"CLC{str(HRL_VINTAGE_YEAR)[2:]}"   # e.g. 2015 -> "CLC15"
LAYER_TCD   = f"LANDCOVER.HR.TCD.{_HRL_SUFFIX}"
LAYER_IMD   = f"LANDCOVER.HR.IMD.{_HRL_SUFFIX}"
LAYER_DTM   = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"

# Target WMS resolution (metres/pixel)
WMS_RESOLUTION = 10

# Sentinel-2 band assets  (STAC asset key -> band name)
S2_BANDS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"}


# ---------------------------------------------------------------------------
# 2.  Low-level helpers
# ---------------------------------------------------------------------------

def _wms_pixel_size(bbox: list[float], lat: float) -> tuple[int, int]:
    """Estimate WMS request size (width_px, height_px) for a WGS84 bbox."""
    width_km  = (bbox[2] - bbox[0]) * 111.0 * math.cos(math.radians(lat))
    height_km = (bbox[3] - bbox[1]) * 111.0
    return int(width_km * 1000 / WMS_RESOLUTION), int(height_km * 1000 / WMS_RESOLUTION)


def download_wms(layer: str, out_path: Path,
                 bbox: list[float], width_px: int, height_px: int) -> "Path | None":
    """
    Download one layer from IGN Geoplateforme WMS and save as GeoTIFF.

    Uses WMS 1.3.0 with EPSG:4326 (BBOX order: lat_min, lon_min, lat_max, lon_max).
    Returns the output Path on success, or None if the request fails.
    """
    wms_bbox = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    print(f"  WMS {layer}  ({width_px}x{height_px} px) ...", end=" ", flush=True)

    r = requests.get(WMS_IGN, params={
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": layer, "CRS": "EPSG:4326", "BBOX": wms_bbox,
        "WIDTH": str(width_px), "HEIGHT": str(height_px),
        "FORMAT": "image/geotiff", "STYLES": "",
    }, timeout=120)

    if r.status_code != 200 or r.content[:4] not in [b"II*\x00", b"MM\x00*"]:
        print(f"FAILED (status={r.status_code}): {r.text[:120]}")
        return None

    out_path.write_bytes(r.content)
    with rasterio.open(out_path) as src:
        print(f"OK  {src.width}x{src.height} px, {src.count} band(s)")
    return out_path


def read_cog_window(href: str, bbox_wgs84: list[float]) -> "tuple[np.ndarray, dict]":
    """
    Read a spatial window from a remote Cloud-Optimised GeoTIFF.

    Parameters
    ----------
    href       : HTTPS or S3 URL of the COG
    bbox_wgs84 : [lon_min, lat_min, lon_max, lat_max]

    Returns
    -------
    (data, profile) where data is float32 2-D and profile is ready for
    writing a single-band GeoTIFF.
    """
    with rasterio.open(href) as src:
        bounds  = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
        window  = from_bounds(*bounds, transform=src.transform)
        data    = src.read(1, window=window).astype("float32")
        profile = src.profile.copy()
        profile.update(
            width     = data.shape[1],
            height    = data.shape[0],
            transform = src.window_transform(window),
            driver    = "GTiff",
            dtype     = "float32",
            compress  = "deflate",
        )
    return data, profile


def save_tif(path: Path, data: np.ndarray, profile: dict,
             band_names: "list[str] | None" = None) -> None:
    """Write a (bands, H, W) or (H, W) array to a GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis]
    profile.update(count=data.shape[0])
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if band_names:
            for i, name in enumerate(band_names, 1):
                dst.set_band_description(i, name)
    print(f"  Saved {path.name}  {data.shape}")


# ---------------------------------------------------------------------------
# 3.  Lidar HD date detection (informational)
# ---------------------------------------------------------------------------

def get_lidar_date(bbox_wgs84: list[float]) -> "datetime | None":
    """
    Attempt to find the Lidar HD acquisition date for the ROI.

    Tries three sources in order:
      1. IGN STAC catalog
      2. WMS GetFeatureInfo on the DTM layer
      3. IGN WFS -- layer IGNF_LIDAR-HD_METADONNEE:metadata

    Returns a timezone-aware datetime (UTC), or None if detection fails.
    """
    print("  [date] Trying IGN STAC ...", end=" ", flush=True)
    try:
        stac_ign = Client.open("https://data.geopf.fr/stac/")
        for coll in ["LIDAR-HD", "lidar-hd", "MNT-LIDAR-HD"]:
            try:
                items = list(stac_ign.search(
                    collections=[coll], bbox=bbox_wgs84, limit=5).items())
                if items:
                    dt = items[0].datetime
                    if dt:
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        print(f"found via STAC ({coll}): {dt.date()}")
                        return dt
            except Exception:
                continue
    except Exception as e:
        print(f"unavailable ({e})")

    print("  [date] Trying WMS GetFeatureInfo ...", end=" ", flush=True)
    try:
        lon_c = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
        lat_c = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
        cx    = int((lon_c - bbox_wgs84[0]) / (bbox_wgs84[2] - bbox_wgs84[0]) * 100)
        cy    = int((bbox_wgs84[3] - lat_c) / (bbox_wgs84[3] - bbox_wgs84[1]) * 100)
        wms_bb = f"{bbox_wgs84[1]},{bbox_wgs84[0]},{bbox_wgs84[3]},{bbox_wgs84[2]}"
        r = requests.get(WMS_IGN, params={
            "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
            "LAYERS": LAYER_DTM, "QUERY_LAYERS": LAYER_DTM,
            "CRS": "EPSG:4326", "BBOX": wms_bb,
            "WIDTH": "100", "HEIGHT": "100", "I": str(cx), "J": str(cy),
            "INFO_FORMAT": "application/json",
        }, timeout=30)
        if r.status_code == 200 and "application/json" in r.headers.get("Content-Type", ""):
            for feat in r.json().get("features", []):
                for k, v in feat.get("properties", {}).items():
                    if "date" in k.lower() and v:
                        try:
                            dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                            print(f"found via GetFeatureInfo ({k}): {dt.date()}")
                            return dt
                        except ValueError:
                            pass
    except Exception as e:
        print(f"failed ({e})")

    print("  [date] Trying IGN WFS metadata ...", end=" ", flush=True)
    try:
        wfs_bbox = (f"{bbox_wgs84[1]},{bbox_wgs84[0]},{bbox_wgs84[3]},{bbox_wgs84[2]},"
                    "urn:ogc:def:crs:EPSG::4326")
        r = requests.get("https://data.geopf.fr/wfs/ows", params={
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": "IGNF_LIDAR-HD_METADONNEE:metadata",
            "BBOX": wfs_bbox, "OUTPUTFORMAT": "application/json", "COUNT": "10",
        }, timeout=30)
        if r.status_code == 200 and "json" in r.headers.get("Content-Type", ""):
            feats = r.json().get("features", [])
            if feats:
                dates = []
                for feat in feats:
                    props = feat.get("properties", {})
                    for key in ("date_debut_acquisition", "date_fin_acquisition"):
                        v = props.get(key)
                        if v:
                            try:
                                dates.append(
                                    datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                                )
                            except (ValueError, TypeError):
                                pass
                if dates:
                    dates.sort()
                    median_dt = dates[len(dates) // 2]
                    print(f"found ({len(feats)} tile(s)): {median_dt.date()}")
                    return median_dt
    except Exception as e:
        print(f"failed ({e})")

    print("  [date] Not found -- will show unknown in summary.")
    return None


def extract_date_from_tif(path: Path) -> "datetime | None":
    """Try to extract acquisition date from GeoTIFF GDAL metadata (fallback)."""
    try:
        with rasterio.open(path) as src:
            tags     = src.tags()
            date_kws = ("datetime", "date", "acqui", "time")
            date_fmts = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                         "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
            for k, v in tags.items():
                if any(kw in k.lower() for kw in date_kws) and v:
                    for fmt in date_fmts:
                        try:
                            dt = datetime.strptime(str(v)[:19], fmt)
                            return dt.replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 4.  Per-zone download orchestrator
# ---------------------------------------------------------------------------

def download_zone(zone: dict, data_root: Path, force: bool = False) -> dict:
    """
    Download all raw data for one zone.

    Saves to  data_root/zones/{zone_id}/  with fixed filenames:
      sentinel2.tif  copdem30.tif  tcd.tif  imd.tif  dtm_lidar.tif

    Parameters
    ----------
    zone      : ZoneSpec dict from zones.py
    data_root : repository  data/  directory
    force     : if False, skip files that already exist

    Returns
    -------
    dict mapping layer name -> local Path (or None on failure)
    """
    zone_id  = zone["id"]
    zone_dir = data_root / "zones" / zone_id
    zone_dir.mkdir(parents=True, exist_ok=True)

    hs   = zone["box_size"]
    bbox = [zone["lon"] - hs, zone["lat"] - hs,
            zone["lon"] + hs, zone["lat"] + hs]
    width_px, height_px = _wms_pixel_size(bbox, zone["lat"])

    print(f"\n{'=' * 64}")
    print(f"  ZONE : {zone['name']}  [{zone_id}]  ({zone['split']})")
    print(f"  Bbox : {[round(v, 4) for v in bbox]}")
    print(f"  WMS  : {width_px} x {height_px} px @ {WMS_RESOLUTION} m/px")
    print(f"{'=' * 64}")

    paths: dict = {}

    # A -- Lidar HD date detection (informational only)
    print("\n[A] Lidar HD acquisition date detection")
    lidar_date = get_lidar_date(bbox)

    # B -- DTM Lidar HD (TARGET)
    print("\n[B] DTM Lidar HD (target)")
    dtm_path = zone_dir / "dtm_lidar.tif"
    if dtm_path.exists() and not force:
        print("  Already exists -- skipping")
    else:
        dtm_path = download_wms(LAYER_DTM, dtm_path, bbox, width_px, height_px)

    if lidar_date is None and dtm_path and dtm_path.exists():
        lidar_date = extract_date_from_tif(dtm_path)
        if lidar_date:
            print(f"  [date] Extracted from GeoTIFF: {lidar_date.date()}")

    paths["dtm_lidar"] = dtm_path

    # C -- Sentinel-2 L2A  (AWS STAC COG)
    print(f"\n[C] Sentinel-2 L2A  (anchor {S2_ANCHOR_YEAR}, HRL vintage {HRL_VINTAGE_YEAR})")
    s2_path = zone_dir / "sentinel2.tif"

    if s2_path.exists() and not force:
        print("  Already exists -- skipping")
        paths["sentinel2"] = s2_path
    else:
        catalog   = Client.open(STAC_AWS)
        s2_anchor = datetime(S2_ANCHOR_YEAR, 7, 1, tzinfo=timezone.utc)
        s2_start  = (s2_anchor - timedelta(days=S2_SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
        s2_end    = (s2_anchor + timedelta(days=S2_SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
        print(f"  Window: {s2_start}/{s2_end}  (cloud < {MAX_CLOUD_COVER}%)")

        search = catalog.search(
            collections=["sentinel-2-l2a"], bbox=bbox,
            datetime=f"{s2_start}/{s2_end}",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}, limit=50,
        )
        items = list(search.items())
        print(f"  Found: {len(items)} images")

        if not items:
            exp_days  = max(S2_SEARCH_WINDOW_DAYS * 2, 730)
            exp_start = (s2_anchor - timedelta(days=exp_days)).strftime("%Y-%m-%d")
            exp_end   = (s2_anchor + timedelta(days=exp_days)).strftime("%Y-%m-%d")
            print(f"  Expanding to +/-{exp_days} d ...")
            items = list(catalog.search(
                collections=["sentinel-2-l2a"], bbox=bbox,
                datetime=f"{exp_start}/{exp_end}",
                query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}, limit=50,
            ).items())

        if not items:
            print("  ERROR: no Sentinel-2 image found")
            paths["sentinel2"] = None
        else:
            _anchor_naive = s2_anchor.replace(tzinfo=None)
            items.sort(key=lambda it: (
                abs((it.datetime.replace(tzinfo=None) - _anchor_naive).days)
                if it.datetime else 999999,
                it.properties.get("eo:cloud_cover", 100),
            ))
            item      = items[0]
            cloud_pct = item.properties.get("eo:cloud_cover", "?")
            s2_dt     = item.datetime
            dt_naive  = s2_dt.replace(tzinfo=None) if s2_dt else None
            delta_d   = abs((dt_naive - _anchor_naive).days) if dt_naive else "?"
            print(f"  Best: {item.id}  date={s2_dt}  cloud={cloud_pct}%  dt={delta_d}d")

            band_arrays = {}
            ref_profile = None
            ok = True
            for asset_key, band_name in S2_BANDS.items():
                asset = item.assets.get(asset_key)
                if asset is None:
                    print(f"  WARNING: asset {asset_key!r} missing")
                    ok = False
                    break
                print(f"  {band_name} ...", end=" ", flush=True)
                data, profile = read_cog_window(asset.href, bbox)
                data = data / 10000.0
                band_arrays[band_name] = data
                if ref_profile is None:
                    ref_profile = profile
                print(f"{data.shape}  [{data.min():.3f}, {data.max():.3f}]")

            if ok and ref_profile is not None:
                nir  = band_arrays["B08"]
                red  = band_arrays["B04"]
                ndvi = (nir - red) / (nir + red + 1e-8)
                band_arrays["NDVI"] = ndvi
                band_order = ["B02", "B03", "B04", "B08", "NDVI"]
                stack = np.stack([band_arrays[b] for b in band_order], axis=0)
                ref_profile.update(count=len(band_order))
                with rasterio.open(s2_path, "w", **ref_profile) as dst:
                    dst.write(stack)
                    for i, name in enumerate(band_order, 1):
                        dst.set_band_description(i, name)
                print(f"  Saved {s2_path.name}  {stack.shape}  CRS={ref_profile['crs']}")
                paths["sentinel2"] = s2_path
            else:
                paths["sentinel2"] = None

    # D -- Copernicus DEM 30 m  (AWS STAC COG)
    print("\n[D] Copernicus DEM 30 m")
    dem_path = zone_dir / "copdem30.tif"
    if dem_path.exists() and not force:
        print("  Already exists -- skipping")
        paths["copdem30"] = dem_path
    else:
        catalog   = Client.open(STAC_AWS)
        dem_items = list(catalog.search(
            collections=["cop-dem-glo-30"], bbox=bbox, limit=10).items())
        print(f"  DEM tiles: {len(dem_items)}")
        if not dem_items:
            print("  ERROR: no COP-DEM tile found")
            paths["copdem30"] = None
        else:
            dem_href = dem_items[0].assets["data"].href
            dem_data, dem_profile = read_cog_window(dem_href, bbox)
            save_tif(dem_path, dem_data, dem_profile, band_names=["COP-DEM-30m"])
            print(f"  Elevation: [{dem_data.min():.1f}, {dem_data.max():.1f}] m")
            paths["copdem30"] = dem_path

    # E -- TCD (WMS RGBA)
    # Note: WMS returns a colourised visualisation. Greyscale luminance used as proxy.
    print(f"\n[E] TCD  ({LAYER_TCD})")
    tcd_path = zone_dir / "tcd.tif"
    if tcd_path.exists() and not force:
        print("  Already exists -- skipping")
        paths["tcd"] = tcd_path
    else:
        paths["tcd"] = download_wms(LAYER_TCD, tcd_path, bbox, width_px, height_px)

    # F -- IMD (WMS RGBA)
    print(f"\n[F] IMD  ({LAYER_IMD})")
    imd_path = zone_dir / "imd.tif"
    if imd_path.exists() and not force:
        print("  Already exists -- skipping")
        paths["imd"] = imd_path
    else:
        paths["imd"] = download_wms(LAYER_IMD, imd_path, bbox, width_px, height_px)

    # G -- Zone summary
    print(f"\n{'=' * 64}")
    print(f"  Zone summary: {zone['name']}  [{zone_id}]")
    print(f"{'=' * 64}")
    for lname, p in paths.items():
        if p and p.exists():
            size_kb = p.stat().st_size / 1024
            with rasterio.open(p) as src:
                status = f"{src.width}x{src.height} px, {src.count} band(s),  {size_kb:.0f} KB"
        else:
            status = "MISSING"
        print(f"  {lname:15s}  {status}")

    lidar_str = lidar_date.date().isoformat() if lidar_date else "not detected"
    print(f"\n  Lidar acq. date  : {lidar_str}  (informational)")
    print(f"  TCD/IMD vintage  : {HRL_VINTAGE_YEAR}  ({_HRL_SUFFIX})")
    print(f"  S2 anchor year   : {S2_ANCHOR_YEAR}")
    return paths


# ---------------------------------------------------------------------------
# 5.  Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download DSM/DTM training data for multiple geographic zones."
    )
    parser.add_argument(
        "--zones", nargs="*", metavar="ZONE_ID",
        help="Zone IDs to process (default: all).  E.g. --zones caen grenoble",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist.",
    )
    return parser.parse_args()


def main():
    from zones import ZONES, ZONES_BY_ID

    args = parse_args()
    if args.zones:
        unknown = set(args.zones) - set(ZONES_BY_ID.keys())
        if unknown:
            print(f"ERROR: unknown zone ID(s): {sorted(unknown)}")
            print(f"Available: {list(ZONES_BY_ID.keys())}")
            sys.exit(1)
        selected = [ZONES_BY_ID[zid] for zid in args.zones]
    else:
        selected = ZONES

    data_root = Path(__file__).resolve().parent.parent / "data"
    print(f"\nDownloading {len(selected)} zone(s) into {data_root}/")
    print(f"  HRL vintage : {HRL_VINTAGE_YEAR}  ({_HRL_SUFFIX})")
    print(f"  S2 anchor   : {S2_ANCHOR_YEAR}  +/-{S2_SEARCH_WINDOW_DAYS} d")
    print(f"  Max clouds  : {MAX_CLOUD_COVER}%")

    results = {}
    for zone in selected:
        results[zone["id"]] = download_zone(zone, data_root, force=args.force)

    # Global summary table
    layers = ["sentinel2", "copdem30", "tcd", "imd", "dtm_lidar"]
    print(f"\n{'=' * 64}")
    print("  GLOBAL DOWNLOAD SUMMARY")
    print(f"{'=' * 64}")
    print(f"  {'Zone':15s}" + "".join(f"  {l:12s}" for l in layers))
    print("  " + "-" * 80)
    for zone_id, paths in results.items():
        row = f"  {zone_id:15s}"
        for layer in layers:
            p  = paths.get(layer)
            ok = p is not None and Path(p).exists()
            row += f"  {'OK':12s}" if ok else f"  {'MISSING':12s}"
        print(row)
    print()


if __name__ == "__main__":
    main()
