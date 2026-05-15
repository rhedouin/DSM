"""
0_download_data.py
==================
Downloads all input data and the target for the DSM model:

INPUTS (predictors):
  1. Sentinel-2 L2A  — bands B02, B03, B04, B08 + NDVI  (10 m, via AWS STAC)
  2. Copernicus DSM  — global DEM at 30 m                 (via AWS STAC)
  3. TCD             — Tree Cover Density (20 m, via IGN Geoplateforme WMS)
  4. IMD             — Imperviousness Degree (20 m, via IGN Geoplateforme WMS)

TARGET:
  5. DTM Lidar HD    — high-resolution DTM (~50 cm, via IGN Geoplateforme WMS)

All outputs are cropped to the bounding box and saved individually.
Alignment to a common grid is done in 1_prepare.py.
"""

import os
import sys
import math
import numpy as np
import requests
import rasterio
from datetime import datetime, timezone, timedelta
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pystac_client import Client
from pathlib import Path

# Allow access to public COGs on S3 without AWS credentials
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

# =============================================================================
# 1. Parameters
# =============================================================================
ZONE = "Caen"
LON, LAT = -0.37, 49.18            # approximate center
BOX_SIZE = 0.05                     # degrees (~5 km)
# Temporal alignment anchor: all predictors (TCD, IMD, Sentinel-2) are aligned to
# HRL_VINTAGE_YEAR. COP-DEM covers ~2011-2015, so 2015 is the best match.
# Available Geoplateforme TCD/IMD vintages: 2006, 2009, 2012, 2015, 2018.
HRL_VINTAGE_YEAR = 2015
S2_ANCHOR_YEAR = 2017               # Sentinel-2 search year (L2A available on AWS from ~2017)
S2_SEARCH_WINDOW_DAYS = 365         # Sentinel-2: search ±N days around Jul 1 of S2_ANCHOR_YEAR
MAX_CLOUD_COVER = 20                 # %

# OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

# Service URLs
STAC_AWS = "https://earth-search.aws.element84.com/v1"
WMS_IGN  = "https://data.geopf.fr/wms-r/wms"

# WMS layers (IGN / Geoplateforme) — TCD/IMD derived from HRL_VINTAGE_YEAR
_hrl_suffix = f"CLC{str(HRL_VINTAGE_YEAR)[2:]}"          # e.g. 2015 → "CLC15"
LAYER_TCD = f"LANDCOVER.HR.TCD.{_hrl_suffix}"
LAYER_IMD = f"LANDCOVER.HR.IMD.{_hrl_suffix}"
LAYER_DTM = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"     # Lidar HD DTM

# Target WMS resolution (~10 m/pixel)
WMS_RESOLUTION = 10  # meters

# Sentinel-2 bands
S2_BANDS = {
    "blue":  "B02",
    "green": "B03",
    "red":   "B04",
    "nir":   "B08",
}

# =============================================================================
# 2. Bounding box and dimensions
# =============================================================================
bbox = [LON - BOX_SIZE, LAT - BOX_SIZE, LON + BOX_SIZE, LAT + BOX_SIZE]
print(f"Zone: {ZONE}")
print(f"Bbox (WGS84): {bbox}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estimate pixel dimensions for WMS requests
width_km = (bbox[2] - bbox[0]) * 111.0 * math.cos(math.radians(LAT))
height_km = (bbox[3] - bbox[1]) * 111.0
width_px = int(width_km * 1000 / WMS_RESOLUTION)
height_px = int(height_km * 1000 / WMS_RESOLUTION)
print(f"Estimated WMS dimensions: {width_px} x {height_px} px (~{WMS_RESOLUTION} m)")


# =============================================================================
# 3. Utility functions
# =============================================================================
def download_wms(layer, filename, width=None, height=None):
    """Download a WMS layer from IGN Geoplateforme as GeoTIFF."""
    w = width or width_px
    h = height or height_px
    # WMS 1.3.0 + EPSG:4326 → BBOX = lat_min, lon_min, lat_max, lon_max
    wms_bbox = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"

    print(f"\n--- {filename} ({layer}) ---")
    print(f"  WMS request: {w}x{h} pixels")

    r = requests.get(WMS_IGN, params={
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "CRS": "EPSG:4326",
        "BBOX": wms_bbox,
        "WIDTH": str(w),
        "HEIGHT": str(h),
        "FORMAT": "image/geotiff",
        "STYLES": "",
    }, timeout=120)

    if r.status_code != 200 or r.content[:4] not in [b'II*\x00', b'MM\x00*']:
        print(f"  ERROR: status={r.status_code}")
        print(f"  Response: {r.text[:300]}")
        return None

    path = os.path.join(OUTPUT_DIR, f"{filename}_{ZONE.lower()}.tif")
    with open(path, "wb") as f:
        f.write(r.content)

    with rasterio.open(path) as src:
        data = src.read(1)
        print(f"  Saved: {path}")
        print(f"  Shape: {data.shape}, CRS: {src.crs}")
        print(f"  min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}")

    return path


def read_cog_window(href, bbox_wgs84):
    """Read a window from a remote COG (Cloud Optimized GeoTIFF)."""
    with rasterio.open(href) as src:
        src_bounds = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
        window = from_bounds(*src_bounds, transform=src.transform)
        data = src.read(1, window=window).astype("float32")
        profile = src.profile.copy()
        profile.update(
            width=data.shape[1],
            height=data.shape[0],
            transform=src.window_transform(window),
            driver="GTiff",
            dtype="float32",
            compress="deflate",
        )
        return data, profile


def save_tif(path, data, profile, description=None):
    """Save a numpy array as a GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis, :]
    profile.update(count=data.shape[0])
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if description and data.shape[0] == 1:
            dst.set_band_description(1, description)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {path} ({size_kb:.0f} KB)")


def get_lidar_date(bbox_wgs84):
    """
    Try to find the Lidar HD acquisition date for the ROI.
    Attempts: (1) IGN STAC catalog, (2) WMS GetFeatureInfo at bbox centre.
    Returns a timezone-aware datetime, or None if not found.
    """
    # Attempt 1: IGN STAC
    print("  Trying IGN STAC...")
    try:
        stac_ign = Client.open("https://data.geopf.fr/stac/")
        for coll in ["LIDAR-HD", "lidar-hd", "MNT-LIDAR-HD"]:
            try:
                results = stac_ign.search(collections=[coll], bbox=bbox_wgs84, limit=5)
                items_stac = list(results.items())
                if items_stac:
                    dt = items_stac[0].datetime
                    if dt:
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        print(f"  Found via STAC (collection={coll}): {dt.date()}")
                        return dt
            except Exception:
                continue
    except Exception as e:
        print(f"  IGN STAC unavailable: {e}")

    # Attempt 2: WMS GetFeatureInfo at bbox centre
    print("  Trying WMS GetFeatureInfo...")
    try:
        lon_c = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
        lat_c = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
        cx = int((lon_c - bbox_wgs84[0]) / (bbox_wgs84[2] - bbox_wgs84[0]) * 100)
        cy = int((bbox_wgs84[3] - lat_c) / (bbox_wgs84[3] - bbox_wgs84[1]) * 100)
        wms_bbox_gfi = f"{bbox_wgs84[1]},{bbox_wgs84[0]},{bbox_wgs84[3]},{bbox_wgs84[2]}"
        r = requests.get(WMS_IGN, params={
            "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
            "LAYERS": LAYER_DTM, "QUERY_LAYERS": LAYER_DTM,
            "CRS": "EPSG:4326", "BBOX": wms_bbox_gfi,
            "WIDTH": "100", "HEIGHT": "100",
            "I": str(cx), "J": str(cy),
            "INFO_FORMAT": "application/json",
        }, timeout=30)
        if r.status_code == 200 and "application/json" in r.headers.get("Content-Type", ""):
            info = r.json()
            for feat in info.get("features", []):
                for k, v in feat.get("properties", {}).items():
                    if "date" in k.lower() and v:
                        try:
                            dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                            print(f"  Found via GetFeatureInfo ({k}): {dt.date()}")
                            return dt
                        except ValueError:
                            pass
    except Exception as e:
        print(f"  GetFeatureInfo failed: {e}")

    # Attempt 3: IGN WFS — IGNF_LIDAR-HD_METADONNEE:metadata (tile-level acquisition dates)
    print("  Trying IGN WFS Lidar HD metadata...")
    try:
        wfs_bbox = (f"{bbox_wgs84[1]},{bbox_wgs84[0]},{bbox_wgs84[3]},{bbox_wgs84[2]},"
                    "urn:ogc:def:crs:EPSG::4326")
        r = requests.get("https://data.geopf.fr/wfs/ows", params={
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": "IGNF_LIDAR-HD_METADONNEE:metadata",
            "BBOX": wfs_bbox,
            "OUTPUTFORMAT": "application/json",
            "COUNT": "10",
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
                                dates.append(datetime.fromisoformat(str(v).replace("Z", "+00:00")))
                            except (ValueError, TypeError):
                                pass
                if dates:
                    dates.sort()
                    median_dt = dates[len(dates) // 2]
                    print(f"  Found via WFS Lidar metadata ({len(feats)} tile(s)): {median_dt.date()}")
                    return median_dt
    except Exception as e:
        print(f"  WFS failed: {e}")

    print("  Pre-download detection failed — will retry from GeoTIFF metadata after DTM download.")
    return None


def select_hrl_vintage(lidar_year):
    """
    Return the closest available TCD/IMD HRL vintage and its layer suffix.
    Available Geoplateforme vintages: 2006, 2009, 2012, 2015, 2018.
    Example: lidar_year=2021 → (2018, 'CLC18').
    """
    vintages = [2006, 2009, 2012, 2015, 2018]
    closest = min(vintages, key=lambda y: abs(y - lidar_year))
    suffix = f"CLC{str(closest)[2:]}"  # 2018 → "CLC18", 2015 → "CLC15"
    return closest, suffix


def extract_date_from_tif(path):
    """
    Try to extract an acquisition date from GeoTIFF GDAL metadata tags.
    Checks all available tag namespaces for any key containing 'date', 'time', or 'acqui'.
    Returns a timezone-aware datetime, or None if not found.
    """
    try:
        with rasterio.open(path) as src:
            all_tags = {}
            all_tags.update(src.tags())
            for ns in ("GDAL_METADATA", "IMAGE_STRUCTURE", "xml:GDAL_METADATA"):
                try:
                    all_tags.update(src.tags(ns=ns))
                except Exception:
                    pass
            date_kws = ("datetime", "date", "acqui", "time", "created")
            date_fmts = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                         "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y")
            for k, v in all_tags.items():
                if any(kw in k.lower() for kw in date_kws) and v:
                    for fmt in date_fmts:
                        try:
                            dt = datetime.strptime(str(v)[:19], fmt)
                            print(f"  Found in GeoTIFF tag '{k}': {dt.date()}")
                            return dt.replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
    except Exception as e:
        print(f"  GeoTIFF metadata read failed: {e}")
    return None


# =============================================================================
# 3b. Lidar HD acquisition date — informational only
# =============================================================================
print("\n" + "=" * 60)
print("LIDAR DATE DETECTION (informational)")
print("=" * 60)

print(f"Predictor anchor     : HRL vintage {HRL_VINTAGE_YEAR} ({_hrl_suffix})")
print(f"  TCD layer          : {LAYER_TCD}")
print(f"  IMD layer          : {LAYER_IMD}")
lidar_date = get_lidar_date(bbox)
if lidar_date is not None:
    print(f"  Lidar HD acq. date : {lidar_date.date()} (target only, not used as predictor anchor)")
else:
    print("  Lidar HD acq. date : not detected")


# =============================================================================
# 3c. DTM Lidar HD download (first, so GeoTIFF metadata can refine the date)
# =============================================================================
print("\n" + "=" * 60)
print("DTM LIDAR HD — TARGET")
print("=" * 60)

dtm_path = download_wms(LAYER_DTM, "dtm_lidar")

# If Lidar date still unknown, try GeoTIFF metadata (informational only)
if lidar_date is None and dtm_path:
    lidar_date = extract_date_from_tif(dtm_path)
    if lidar_date is not None:
        print(f"  Lidar date from GeoTIFF : {lidar_date.date()}")


# =============================================================================
# 4. Sentinel-2 L2A (via AWS STAC — COGs)
# =============================================================================
print("\n" + "=" * 60)
print("SENTINEL-2 L2A")
print("=" * 60)

catalog = Client.open(STAC_AWS)

# Search window centred on Jul 1 of S2_ANCHOR_YEAR
# Note: Sentinel-2 L2A is not available on AWS Earth Search before ~March 2017.
s2_anchor = datetime(S2_ANCHOR_YEAR, 7, 1, tzinfo=timezone.utc)
s2_start = (s2_anchor - timedelta(days=S2_SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
s2_end   = (s2_anchor + timedelta(days=S2_SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
s2_dates = f"{s2_start}/{s2_end}"
print(f"Sentinel-2 window: \u00b1{S2_SEARCH_WINDOW_DAYS} d around {s2_anchor.date()} (S2 anchor {S2_ANCHOR_YEAR}, HRL anchor {HRL_VINTAGE_YEAR}) \u2192 {s2_dates}")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=s2_dates,
    query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
    limit=50,
)
items = list(search.items())
print(f"Images found: {len(items)}")

# Expand search window if no result
if not items:
    exp_days = max(S2_SEARCH_WINDOW_DAYS * 2, 730)
    s2_start = (s2_anchor - timedelta(days=exp_days)).strftime("%Y-%m-%d")
    s2_end   = (s2_anchor + timedelta(days=exp_days)).strftime("%Y-%m-%d")
    print(f"  No results — expanding to \u00b1{exp_days} days...")
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox,
        datetime=f"{s2_start}/{s2_end}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        limit=50,
    )
    items = list(search.items())

if not items:
    print("ERROR: no Sentinel-2 image found.")
    sys.exit(1)

# Sort: temporal proximity to HRL anchor (primary), cloud cover (secondary)
_anchor_naive = s2_anchor.replace(tzinfo=None)
def _s2_sort_key(it):
    dt = it.datetime
    if dt is None:
        return (999999, 100)
    return (abs((dt.replace(tzinfo=None) - _anchor_naive).days), it.properties.get("eo:cloud_cover", 100))
items.sort(key=_s2_sort_key)

item = items[0]
cloud_cover = item.properties.get("eo:cloud_cover", "?")
print(f"Selected image: {item.id}")
print(f"  Date: {item.datetime}, Cloud cover: {cloud_cover}%")
s2_dt = item.datetime
if s2_dt is None:
    s2_dt = s2_anchor
elif s2_dt.tzinfo is None:
    s2_dt = s2_dt.replace(tzinfo=timezone.utc)
print(f"  \u0394t from HRL anchor ({HRL_VINTAGE_YEAR}-07-01): {abs((s2_dt.replace(tzinfo=None) - _anchor_naive).days)} days")

band_arrays = {}
ref_profile = None

for asset_key, band_name in S2_BANDS.items():
    asset = item.assets.get(asset_key)
    if not asset:
        print(f"  WARNING: asset '{asset_key}' not found")
        continue

    print(f"\n  Reading {band_name} ({asset_key})...")
    data, profile = read_cog_window(asset.href, bbox)

    if ref_profile is None:
        ref_profile = profile

    data = data / 10000.0  # DN → reflectance
    band_arrays[band_name] = data
    print(f"    Shape: {data.shape}, min={data.min():.4f}, max={data.max():.4f}")

# NDVI
nir = band_arrays["B08"]
red = band_arrays["B04"]
ndvi = (nir - red) / (nir + red + 1e-8)
band_arrays["NDVI"] = ndvi
print(f"\n  NDVI: min={ndvi.min():.4f}, max={ndvi.max():.4f}")

# Stack and save
band_order = ["B02", "B03", "B04", "B08", "NDVI"]
stack = np.stack([band_arrays[b] for b in band_order], axis=0)
s2_path = os.path.join(OUTPUT_DIR, f"sentinel2_{ZONE.lower()}.tif")
ref_profile.update(count=len(band_order))
with rasterio.open(s2_path, "w", **ref_profile) as dst:
    dst.write(stack)
    for i, name in enumerate(band_order):
        dst.set_band_description(i + 1, name)
print(f"\n  Sentinel-2 → {s2_path}")
print(f"  {stack.shape[0]} bands: {band_order}")
print(f"  CRS: {ref_profile['crs']}, Size: {stack.shape[1]}x{stack.shape[2]}")


# =============================================================================
# 5. Copernicus DEM 30 m (via AWS STAC)
# =============================================================================
print("\n" + "=" * 60)
print("COPERNICUS DEM 30 m")
print("=" * 60)

dem_search = catalog.search(
    collections=["cop-dem-glo-30"],
    bbox=bbox,
    limit=10,
)
dem_items = list(dem_search.items())
print(f"DEM tiles found: {len(dem_items)}")

if not dem_items:
    print("ERROR: no DEM tile found.")
    sys.exit(1)

dem_item = dem_items[0]
print(f"Tile: {dem_item.id}")
dem_href = dem_item.assets["data"].href
print(f"  URL: {dem_href[:100]}...")

dem_data, dem_profile = read_cog_window(dem_href, bbox)
dem_path = os.path.join(OUTPUT_DIR, f"copdem30_{ZONE.lower()}.tif")
save_tif(dem_path, dem_data, dem_profile, description="COP-DEM-30m")
print(f"  Shape: {dem_data.shape}, min={dem_data.min():.2f} m, max={dem_data.max():.2f} m")


# =============================================================================
# 6. TCD — Tree Cover Density (via WMS Geoplateforme)
# =============================================================================
print("\n" + "=" * 60)
print("TCD — TREE COVER DENSITY")
print("=" * 60)

tcd_path = download_wms(LAYER_TCD, "tcd")


# =============================================================================
# 7. IMD — Imperviousness Degree (via WMS Geoplateforme)
# =============================================================================
print("\n" + "=" * 60)
print("IMD — IMPERVIOUSNESS DEGREE")
print("=" * 60)

imd_path = download_wms(LAYER_IMD, "imd")


# =============================================================================
# 8. Summary  (DTM Lidar HD was downloaded in section 3c)
# =============================================================================
print("\n" + "=" * 60)
print("DOWNLOADED FILES SUMMARY")
print("=" * 60)

files = {
    "Sentinel-2 (B02,B03,B04,B08,NDVI)": s2_path,
    "Copernicus DEM 30 m":               dem_path,
    "TCD (tree cover density)":          tcd_path,
    "IMD (imperviousness)":              imd_path,
    "DTM Lidar HD (target)":             dtm_path,
}

for name, path in files.items():
    if path and os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        with rasterio.open(path) as src:
            print(f"  {name:45s} → {src.width}x{src.height}, {src.count} band(s), {size_kb:.0f} KB")
    else:
        print(f"  {name:45s} → MISSING")

print(f"\nOutput directory: {OUTPUT_DIR}")

# --- Temporal coherence ---
print("\n" + "=" * 60)
print("TEMPORAL COHERENCE SUMMARY")
print("=" * 60)

_delta_s2 = abs((s2_dt.replace(tzinfo=None) - _anchor_naive).days)
print(f"  {'TCD / IMD anchor (Copernicus HRL)':45s} → vintage {HRL_VINTAGE_YEAR}  [{_hrl_suffix}]")
print(f"  {'Sentinel-2 L2A':45s} → {s2_dt.date()}  (\u0394t = {_delta_s2} day{'s' if _delta_s2 != 1 else ''} from anchor)")
print(f"  {'Copernicus DEM 30m':45s} → ~2011\u20132015 (static product, matches anchor era)")
print(f"  {'DTM Lidar HD (target)':45s} → {lidar_date.date() if lidar_date else 'acquisition date unknown'}")

print("\nDone!")
