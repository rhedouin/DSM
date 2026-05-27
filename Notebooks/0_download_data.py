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
from collections import defaultdict

import rasterio.merge
from pystac_client import Client
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject as rio_reproject, Resampling as RioResampling

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
MAX_CLOUD_COVER       = 10     # % -- STAC item-level filter (tile average)

# SCL classes considered cloudy/contaminated for per-pixel masking
# 3=cloud shadow  8=cloud medium  9=cloud high  10=thin cirrus
SCL_CLOUD_CLASSES = [3, 8, 9, 10]

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


def _select_best_s2_items(items: list, anchor_dt) -> list:
    """
    From a list of STAC items, select the best item *per MGRS tile*.
    Best = closest to anchor date, then lowest cloud cover.
    Returns a list with one item per unique tile.
    """
    anchor_naive = anchor_dt.replace(tzinfo=None)
    by_tile: dict[str, list] = defaultdict(list)
    for it in items:
        tile = it.properties.get("s2:mgrs_tile", it.id)
        by_tile[tile].append(it)
    selected = []
    for tile_items in by_tile.values():
        tile_items.sort(key=lambda it: (
            abs((it.datetime.replace(tzinfo=None) - anchor_naive).days)
            if it.datetime else 999_999,
            it.properties.get("eo:cloud_cover", 100),
        ))
        selected.append(tile_items[0])
    return selected


def _mosaic_s2_band(hrefs: list[str], bbox_wgs84: list[float],
                    ref_profile: "dict | None" = None,
                    resampling: RioResampling = RioResampling.bilinear,
                    ) -> "tuple[np.ndarray, dict] | tuple[None, None]":
    """
    Read one S2 band from multiple COG tiles and mosaic them.

    If ref_profile is supplied, the mosaic is reprojected to that exact
    pixel grid so all bands share the same transform/CRS.

    Use resampling=RioResampling.nearest for categorical layers (e.g. SCL).

    Returns (data_float32_2d, profile) or (None, None) on failure.
    """
    datasets: list[rasterio.DatasetReader] = []
    memfiles: list[rasterio.MemoryFile]    = []
    try:
        for href in hrefs:
            try:
                data, prof = read_cog_window(href, bbox_wgs84)
                prof = prof.copy()
                prof.update(count=1, dtype="float32", nodata=0.0)
                mf = rasterio.MemoryFile()
                with rasterio.open(mf, "w", **prof) as ds:
                    ds.write(data.astype("float32")[np.newaxis])
                memfiles.append(mf)
                datasets.append(rasterio.open(mf))
            except Exception as exc:
                print(f"    WARNING: could not read tile ({exc})")

        if not datasets:
            return None, None

        if len(datasets) == 1 and ref_profile is None:
            arr = datasets[0].read(1).astype("float32")
            out_profile = datasets[0].profile.copy()
            return arr, out_profile

        # Reproject all tiles to the CRS of the first tile before merging.
        # Zones that straddle two UTM zones produce tiles in different CRS
        # (e.g. EPSG:32632 and EPSG:32633); rasterio.merge raises an error
        # rather than reprojecting automatically.
        target_crs = datasets[0].crs
        aligned_datasets: list[rasterio.DatasetReader] = []
        aligned_memfiles: list[rasterio.MemoryFile] = []
        for ds in datasets:
            if ds.crs == target_crs:
                aligned_datasets.append(ds)
            else:
                from rasterio.warp import calculate_default_transform
                dst_transform, dst_w, dst_h = calculate_default_transform(
                    ds.crs, target_crs, ds.width, ds.height, *ds.bounds
                )
                arr_src = ds.read(1).astype("float32")
                arr_dst = np.zeros((dst_h, dst_w), dtype="float32")
                rio_reproject(
                    source=arr_src, destination=arr_dst,
                    src_transform=ds.transform, src_crs=ds.crs,
                    dst_transform=dst_transform, dst_crs=target_crs,
                    resampling=resampling,
                    src_nodata=0.0, dst_nodata=0.0,
                )
                rep_profile = ds.profile.copy()
                rep_profile.update(
                    crs=target_crs, transform=dst_transform,
                    width=dst_w, height=dst_h,
                    dtype="float32", nodata=0.0, count=1,
                )
                amf = rasterio.MemoryFile()
                with rasterio.open(amf, "w", **rep_profile) as out_ds:
                    out_ds.write(arr_dst[np.newaxis])
                aligned_memfiles.append(amf)
                aligned_datasets.append(rasterio.open(amf))
        datasets = aligned_datasets
        memfiles.extend(aligned_memfiles)

        # Mosaic all tiles (now all in the same CRS)
        mosaic, tr = rasterio.merge.merge(datasets, nodata=0.0, method="first")
        out_profile = datasets[0].profile.copy()
        out_profile.update(
            width=mosaic.shape[-1], height=mosaic.shape[-2], transform=tr
        )

        # Align to reference grid so all bands have identical pixel grids
        if ref_profile is not None:
            aligned = np.zeros(
                (ref_profile["height"], ref_profile["width"]), dtype="float32"
            )
            rio_reproject(
                source        = mosaic[0].astype("float32"),
                destination   = aligned,
                src_transform = tr,
                src_crs       = out_profile["crs"],
                dst_transform = ref_profile["transform"],
                dst_crs       = ref_profile["crs"],
                resampling    = resampling,
                src_nodata    = 0.0,
                dst_nodata    = 0.0,
            )
            return aligned, ref_profile

        return mosaic[0].astype("float32"), out_profile

    finally:
        for ds in datasets:
            try: ds.close()
            except Exception: pass
        for mf in memfiles:
            try: mf.close()
            except Exception: pass


def _mosaic_cop_dem(items: list, bbox_wgs84: list[float]
                   ) -> "tuple[np.ndarray, dict] | tuple[None, None]":
    """
    Read and mosaic all COP-DEM 30 m COG tiles covering a bounding box.

    COP-DEM tiles are 1°×1° so a zone centred near a tile corner can span
    up to 4 tiles.  We read each tile with boundless=True so that every tile
    returns an array of exactly the bbox size, padded with NODATA_F outside
    the tile's own extent.  The merge then picks the first valid value at
    each pixel, giving full seamless coverage.

    Using boundless=False (default) clips the array to the tile extent but
    keeps the transform pointing at the full bbox origin — which places tile
    fragments at the wrong position in the mosaic.
    """
    NODATA_F = -32768.0
    datasets: list[rasterio.DatasetReader] = []
    memfiles: list[rasterio.MemoryFile]    = []
    try:
        for item in items:
            href = item.assets["data"].href
            try:
                with rasterio.open(href) as src:
                    bounds = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
                    window = from_bounds(*bounds, transform=src.transform)
                    # boundless=True: areas outside this tile are padded with
                    # NODATA_F instead of being clipped, so the returned array
                    # covers exactly the full bbox and its transform is correct.
                    data = src.read(
                        1, window=window,
                        boundless=True, fill_value=NODATA_F,
                    ).astype("float32")
                    nd = src.nodata
                    if nd is not None and float(nd) != NODATA_F:
                        data[data == float(nd)] = NODATA_F
                    prof = src.profile.copy()
                    prof.update(
                        width     = data.shape[1],
                        height    = data.shape[0],
                        transform = src.window_transform(window),
                        driver    = "GTiff",
                        dtype     = "float32",
                        compress  = "deflate",
                        nodata    = NODATA_F,
                        count     = 1,
                    )
                mf = rasterio.MemoryFile()
                with rasterio.open(mf, "w", **prof) as ds:
                    ds.write(data[np.newaxis])
                memfiles.append(mf)
                datasets.append(rasterio.open(mf))
            except Exception as exc:
                print(f"    WARNING: could not read DEM tile {item.id} ({exc})")

        if not datasets:
            return None, None

        if len(datasets) == 1:
            arr  = datasets[0].read(1).astype("float32")
            prof = datasets[0].profile.copy()
            return arr, prof

        # All COP-DEM tiles share EPSG:4326 — no CRS conflicts
        mosaic, tr = rasterio.merge.merge(datasets, nodata=NODATA_F, method="first")
        prof = datasets[0].profile.copy()
        prof.update(
            width     = mosaic.shape[-1],
            height    = mosaic.shape[-2],
            transform = tr,
            nodata    = NODATA_F,
        )
        return mosaic[0].astype("float32"), prof

    finally:
        for ds in datasets:
            try: ds.close()
            except Exception: pass
        for mf in memfiles:
            try: mf.close()
            except Exception: pass


def _cop_dem_zero_frac(path: Path) -> float:
    """Fraction of pixels with value exactly 0.0 in the COP-DEM raster."""
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")
        return float((data == 0.0).sum()) / max(data.size, 1)
    except Exception:
        return 0.0


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

def download_zone(zone: dict, data_root: Path, force: bool = False,
                  max_cloud_cover: int = MAX_CLOUD_COVER) -> dict:
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
        print(f"  Window: {s2_start}/{s2_end}  (cloud < {max_cloud_cover}%)")

        search = catalog.search(
            collections=["sentinel-2-l2a"], bbox=bbox,
            datetime=f"{s2_start}/{s2_end}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}}, limit=50,
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
                query={"eo:cloud_cover": {"lt": max_cloud_cover}}, limit=50,
            ).items())

        if not items:
            print("  ERROR: no Sentinel-2 image found")
            paths["sentinel2"] = None
        else:
            # Pick best item *per MGRS tile* so zones crossing tile boundaries
            # get full coverage via mosaicking.
            best_items = _select_best_s2_items(items, s2_anchor)
            tiles_info = ", ".join(
                f"{it.properties.get('s2:mgrs_tile','?')} "
                f"({it.properties.get('eo:cloud_cover','?')}%)"
                for it in best_items
            )
            print(f"  Selected {len(best_items)} tile(s): {tiles_info}")

            band_arrays: dict[str, np.ndarray] = {}
            ref_profile: "dict | None" = None
            ok = True
            for asset_key, band_name in S2_BANDS.items():
                hrefs = [
                    it.assets[asset_key].href
                    for it in best_items
                    if asset_key in it.assets
                ]
                if not hrefs:
                    print(f"  WARNING: asset {asset_key!r} not found in any tile")
                    ok = False
                    break
                print(f"  {band_name} ({len(hrefs)} tile(s)) ...", end=" ", flush=True)
                data, profile = _mosaic_s2_band(hrefs, bbox, ref_profile)
                if data is None:
                    print("FAILED")
                    ok = False
                    break
                data = data / 10000.0
                band_arrays[band_name] = data
                if ref_profile is None:
                    ref_profile = profile
                print(f"{data.shape}  [{data.min():.3f}, {data.max():.3f}]")

            # -- SCL per-pixel cloud mask ---------------------------------
            if ok and ref_profile is not None:
                scl_hrefs = [
                    it.assets["scl"].href
                    for it in best_items if "scl" in it.assets
                ]
                if scl_hrefs:
                    print(f"  SCL mask ({len(scl_hrefs)} tile(s)) ...",
                          end=" ", flush=True)
                    scl_data, _ = _mosaic_s2_band(
                        scl_hrefs, bbox, ref_profile,
                        resampling=RioResampling.nearest,
                    )
                    if scl_data is not None:
                        cloud_mask = np.isin(
                            np.round(scl_data).astype(np.int32),
                            SCL_CLOUD_CLASSES,
                        )
                        pct = 100.0 * cloud_mask.sum() / cloud_mask.size
                        print(f"{scl_data.shape}  "
                              f"cloud/shadow = {pct:.1f}% of pixels")
                        if pct > 30:
                            print(f"  WARNING: {pct:.1f}% cloud/shadow pixels -- "
                                  "consider re-downloading this zone")
                        for bname in band_arrays:
                            band_arrays[bname][cloud_mask] = 0.0  # nodata = 0
                    else:
                        print("FAILED -- no per-pixel cloud masking")
                else:
                    print("  WARNING: SCL asset missing -- "
                          "no per-pixel cloud masking")

            if ok and ref_profile is not None:
                nir  = band_arrays["B08"]
                red  = band_arrays["B04"]
                ndvi = (nir - red) / (nir + red + 1e-8)
                band_arrays["NDVI"] = ndvi
                band_order = ["B02", "B03", "B04", "B08", "NDVI"]
                stack = np.stack([band_arrays[b] for b in band_order], axis=0)
                out_profile = ref_profile.copy()
                out_profile.update(count=len(band_order), dtype="float32",
                                   nodata=0.0, compress="deflate")
                with rasterio.open(s2_path, "w", **out_profile) as dst:
                    dst.write(stack)
                    for i, name in enumerate(band_order, 1):
                        dst.set_band_description(i, name)
                n_nodata = int((stack[0] == 0).sum())
                coverage = 100 * (1 - n_nodata / stack[0].size)
                print(f"  Saved {s2_path.name}  {stack.shape}  "
                      f"CRS={out_profile['crs']}  coverage={coverage:.1f}%")
                if coverage < 50:
                    print(f"  WARNING: coverage only {coverage:.1f}% -- "
                          "consider re-downloading or checking zone bbox")
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
        print(f"  DEM tiles found: {len(dem_items)}")
        if not dem_items:
            print("  ERROR: no COP-DEM tile found")
            paths["copdem30"] = None
        else:
            dem_data, dem_profile = _mosaic_cop_dem(dem_items, bbox)
            if dem_data is None:
                print("  ERROR: could not read any DEM tile")
                paths["copdem30"] = None
            else:
                save_tif(dem_path, dem_data, dem_profile, band_names=["COP-DEM-30m"])
                valid = dem_data[dem_data > -32000]
                elev_str = f"[{valid.min():.1f}, {valid.max():.1f}] m" if valid.size else "no valid pixels"
                print(f"  Elevation: {elev_str}  ({len(dem_items)} tile(s) mosaicked)")
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
    parser.add_argument(
        "--max-cloud", type=int, default=MAX_CLOUD_COVER, metavar="PCT",
        help=f"Maximum tile-level cloud cover %% for S2 search (default: {MAX_CLOUD_COVER}).",
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
    print(f"  Max clouds  : {args.max_cloud}%  (tile-level)  "
          f"+SCL per-pixel mask classes {SCL_CLOUD_CLASSES}")

    results = {}
    for zone in selected:
        results[zone["id"]] = download_zone(
            zone, data_root, force=args.force,
            max_cloud_cover=args.max_cloud,
        )

    # Global summary table
    layers = ["sentinel2", "copdem30", "tcd", "imd", "dtm_lidar"]
    print(f"\n{'=' * 64}")
    print("  GLOBAL DOWNLOAD SUMMARY")
    print(f"{'=' * 64}")
    print(f"  {'Zone':15s}" + "".join(f"  {l:12s}" for l in layers))
    print("  " + "-" * 80)
    cop_dem_warnings: list[tuple[str, float]] = []
    for zone_id, paths in results.items():
        row = f"  {zone_id:15s}"
        for layer in layers:
            p  = paths.get(layer)
            ok = p is not None and Path(p).exists()
            if ok and layer == "copdem30":
                frac = _cop_dem_zero_frac(Path(p))
                if frac > 0.05:
                    cop_dem_warnings.append((zone_id, frac))
                    row += f"  {'⚠ WARN':12s}"
                else:
                    row += f"  {'OK':12s}"
            else:
                row += f"  {'OK':12s}" if ok else f"  {'MISSING':12s}"
        print(row)
    print()
    if cop_dem_warnings:
        print("  " + "!" * 60)
        print("  !! COP-DEM COVERAGE WARNING                               !!")
        print("  " + "!" * 60)
        for zone_id, frac in cop_dem_warnings:
            print(f"  !!  {zone_id:20s}  {frac*100:.1f}% pixels == 0  "
                  "→ re-download with --force  !!")
        print("  " + "!" * 60)
        print()


if __name__ == "__main__":
    main()
