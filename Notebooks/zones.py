"""
zones.py
========
Catalogue of geographic zones for DSM/DTM dataset collection.

Each zone is a ~20 km bounding box (box_size = 0.10°, i.e. ±0.10° from centre)
selected to maximise diversity across:
  - Terrain type  : flat plain / bocage / Alpine / forest / plateau / coastal cliff
  - Land use      : urban / periurban / agricultural / forest / maquis / scrubland
  - Elevation     : 0–100 m / 100–400 m / 400–2000 m
  - Climate       : Atlantic / Continental / Mediterranean / Alpine / Sub-Alpine
  - Region        : all major French physiographic regions represented

At 10 m resolution (Sentinel-2 reference grid) each 0.2° × 0.2° bbox yields
approximately 1400 × 2200 px, giving ~180 stride-128 patches per zone.

Geographic train / validation / test split
------------------------------------------
  TRAIN (28) : caen, grenoble, landes, cantal_plateau, bretagne_brest,
               champagne_plaine, pyrenees_ariege, alsace_nord, perigord_noir,
               morvan_foret, ardeche_plateau, var_maures, maine_sarthe,
               normandie_seine, foret_fontainebleau, jura_plateau, vercors_plateau,
               limousin_correze, beauce_loiret, haute_saone_foret, drome_baronnies,
               berry_sancerre, tarn_millau, landes_biscarrosse, bocage_virois,
               alsace_vignoble, pays_de_caux, auvergne_puy_dome
  VAL   ( 6) : paris_sud, anjou_loire, herault_garrigue,
               lyon_est, strasbourg_ried, bordeaux_medoc
  TEST  ( 6) : toulouse, nice_arriere, lorraine_moselle,
               rennes_periurban, dijon_plateau, picardie_somme

Val/test zones are geographically distant from all training zones to
prevent spatial leakage.  Lidar HD IGN coverage confirmed or expected for
all zones as of 2024–2025.
"""

from __future__ import annotations
from typing import TypedDict


class ZoneSpec(TypedDict):
    id: str          # Short slug — used as folder name and file prefix
    name: str        # Human-readable label
    lon: float       # Centre longitude (WGS84 decimal degrees)
    lat: float       # Centre latitude  (WGS84 decimal degrees)
    box_size: float  # Half-width of bounding box in degrees (default 0.10)
    terrain: str     # Brief terrain description
    landuse: str     # Dominant land use
    split: str       # "train", "val", or "test"
    notes: str       # Special remarks


# ---------------------------------------------------------------------------
# Zone catalogue
# ---------------------------------------------------------------------------
# Bounding box = [lon - box_size, lat - box_size, lon + box_size, lat + box_size]
# ---------------------------------------------------------------------------

ZONES: list[ZoneSpec] = [

    # ── TRAIN (28 zones) ───────────────────────────────────────────────────

    dict(
        id       = "caen",
        name     = "Caen (Normandie)",
        lon      = -0.37, lat = 49.18, box_size = 0.10,
        terrain  = "rolling bocage hills, 10–120 m",
        landuse  = "periurban / farmland / hedgerow",
        split    = "train",
        notes    = "Baseline zone — mixed periurban bocage, Lidar HD 2023",
    ),
    dict(
        id       = "grenoble",
        name     = "Grenoble (Isère, French Alps)",
        lon      = 5.72, lat = 45.17, box_size = 0.10,
        terrain  = "steep Alpine valley, 200–1000 m",
        landuse  = "dense urban core + forested Alpine slopes",
        split    = "train",
        notes    = "Extreme relief — tests model on high-gradient Alpine terrain",
    ),
    dict(
        id       = "landes",
        name     = "Landes de Gascogne (Gironde)",
        lon      = -0.85, lat = 44.30, box_size = 0.10,
        terrain  = "flat coastal plain, 20–60 m",
        landuse  = "dense planted maritime pine forest",
        split    = "train",
        notes    = "Large DSM-DTM gap from dense pine canopy — key forest test case",
    ),
    dict(
        id       = "cantal_plateau",
        name     = "Cantal (Massif Central)",
        lon      = 2.88, lat = 45.08, box_size = 0.10,
        terrain  = "volcanic plateau, 900–1300 m",
        landuse  = "highland pasture / heathland",
        split    = "train",
        notes    = "High-altitude open grassland, minimal canopy — volcanic erosion surface",
    ),
    dict(
        id       = "bretagne_brest",
        name     = "Finistère (Brest hinterland)",
        lon      = -4.48, lat = 48.40, box_size = 0.10,
        terrain  = "Atlantic bocage, 30–200 m",
        landuse  = "bocage farmland / hedgerows / mixed agriculture",
        split    = "train",
        notes    = "Armorican massif Atlantic bocage, wetter than Caen",
    ),
    dict(
        id       = "champagne_plaine",
        name     = "Champagne (east of Reims)",
        lon      = 4.15, lat = 48.93, box_size = 0.10,
        terrain  = "open chalk plain, 80–150 m",
        landuse  = "large-scale arable farming (wheat, beet, barley)",
        split    = "train",
        notes    = "Very flat Champagne crayeuse — tests model near-zero relief",
    ),
    dict(
        id       = "pyrenees_ariege",
        name     = "Pyrénées (Ariège foothills)",
        lon      = 1.60, lat = 42.96, box_size = 0.10,
        terrain  = "Pyrenean ridge and foothills, 400–1400 m",
        landuse  = "mixed deciduous/coniferous forest + subalpine pasture",
        split    = "train",
        notes    = "High-gradient SW mountain zone, different climate from Grenoble",
    ),
    dict(
        id       = "alsace_nord",
        name     = "Alsace (north Bas-Rhin)",
        lon      = 7.55, lat = 48.77, box_size = 0.10,
        terrain  = "flat Rhine alluvial plain, 120–160 m",
        landuse  = "arable fields / periurban / light industry",
        split    = "train",
        notes    = "Flat Rhine floodplain — very low relief, high canal density",
    ),
    dict(
        id       = "perigord_noir",
        name     = "Périgord Noir (Sarlat area)",
        lon      = 1.21, lat = 44.87, box_size = 0.10,
        terrain  = "rolling limestone Périgord hills, 100–280 m",
        landuse  = "sessile oak forest + mixed cereal / sunflower agriculture",
        split    = "train",
        notes    = "Karst limestone relief, dense mixed forest and cropland mosaic",
    ),
    dict(
        id       = "morvan_foret",
        name     = "Morvan (Bourgogne)",
        lon      = 4.07, lat = 47.12, box_size = 0.10,
        terrain  = "rounded forested crystalline uplands, 500–900 m",
        landuse  = "dense deciduous and coniferous forest",
        split    = "train",
        notes    = "Humid Morvan massif — dense forest, rounded summit relief",
    ),
    dict(
        id       = "ardeche_plateau",
        name     = "Ardèche (central plateau)",
        lon      = 4.13, lat = 44.68, box_size = 0.10,
        terrain  = "basalt and limestone plateau with gorges, 300–800 m",
        landuse  = "scrubland + sparse oak forest + dryland agriculture",
        split    = "train",
        notes    = "Volcanic Ardèche plateau — Mediterranean-Atlantic transition",
    ),
    dict(
        id       = "var_maures",
        name     = "Massif des Maures (Var)",
        lon      = 6.30, lat = 43.47, box_size = 0.10,
        terrain  = "Mediterranean siliceous coastal hills, 100–500 m",
        landuse  = "cork oak and holm oak forest (maquis)",
        split    = "train",
        notes    = "Dense Mediterranean maquis on siliceous substrate — fire-prone",
    ),
    dict(
        id       = "maine_sarthe",
        name     = "Maine (Sarthe)",
        lon      = 0.13, lat = 47.98, box_size = 0.10,
        terrain  = "gentle Armorican fringe hills, 100–230 m",
        landuse  = "mixed deciduous forest + bocage farmland",
        split    = "train",
        notes    = "Armorican massif western fringe — mixed forest and bocage",
    ),
    dict(
        id       = "normandie_seine",
        name     = "Seine valley (Rouen, Normandie)",
        lon      = 1.08, lat = 49.42, box_size = 0.10,
        terrain  = "Seine meander valley + chalk plateau, 5–160 m",
        landuse  = "floodplain meadow / suburban / chalk arable plateau",
        split    = "train",
        notes    = "Major river incised into chalk — strong valley/plateau contrast",
    ),
    dict(
        id       = "foret_fontainebleau",
        name     = "Forêt de Fontainebleau (Seine-et-Marne)",
        lon      = 2.68, lat = 48.38, box_size = 0.10,
        terrain  = "sandy Brie plateau with sandstone buttes, 60–150 m",
        landuse  = "dense broadleaf/mixed forest on sand / periurban fringe",
        split    = "train",
        notes    = "Iconic forest on Fontainebleau sandstone — low relief, complex canopy",
    ),
    dict(
        id       = "jura_plateau",
        name     = "Jura (plateau central, Doubs)",
        lon      = 5.82, lat = 46.93, box_size = 0.10,
        terrain  = "Jura anticlinal limestone ridges, 400–1000 m",
        landuse  = "montane forest (spruce, beech) + upland grassland",
        split    = "train",
        notes    = "Classic Jura fold ridges — regular alternation of forested crests and meadow valleys",
    ),
    dict(
        id       = "vercors_plateau",
        name     = "Vercors (plateau, Drôme/Isère)",
        lon      = 5.37, lat = 44.90, box_size = 0.10,
        terrain  = "karstic limestone plateau with cliff edges, 900–1700 m",
        landuse  = "beech and fir forest + subalpine meadow + karst cirques",
        split    = "train",
        notes    = "High pre-Alpine plateau bounded by 500 m limestone cliffs",
    ),
    dict(
        id       = "limousin_correze",
        name     = "Corrèze (Limousin plateau)",
        lon      = 1.87, lat = 45.38, box_size = 0.10,
        terrain  = "rounded granite uplands, 400–700 m",
        landuse  = "mixed bocage + deciduous forest + river valleys",
        split    = "train",
        notes    = "Humid Limousin massif — granitic bocage with intricate stream network",
    ),
    dict(
        id       = "beauce_loiret",
        name     = "Beauce (Loiret / Eure-et-Loir)",
        lon      = 1.57, lat = 48.15, box_size = 0.10,
        terrain  = "extremely flat cereal plain, 100–160 m",
        landuse  = "intensive wheat, rapeseed and beet farming",
        split    = "train",
        notes    = "Flattest zone in dataset — tests model at near-zero residual",
    ),
    dict(
        id       = "haute_saone_foret",
        name     = "Haute-Saône (Vosges du Sud)",
        lon      = 6.27, lat = 47.72, box_size = 0.10,
        terrain  = "Vosges crystalline foothills, 300–700 m",
        landuse  = "dense coniferous and mixed forest + rural meadow",
        split    = "train",
        notes    = "Southern Vosges slopes — dense fir/spruce canopy, moderate relief",
    ),
    dict(
        id       = "drome_baronnies",
        name     = "Baronnies Provençales (Drôme)",
        lon      = 5.18, lat = 44.48, box_size = 0.10,
        terrain  = "limestone synclines and ridges, 400–1200 m",
        landuse  = "lavender / olive / sparse oak scrubland / dryland cereal",
        split    = "train",
        notes    = "Pre-Alpine Provençal ranges — strongly folded limestone, semi-arid",
    ),
    dict(
        id       = "berry_sancerre",
        name     = "Berry (Loire hills, Sancerre)",
        lon      = 2.85, lat = 47.33, box_size = 0.10,
        terrain  = "gentle limestone hills above Loire, 150–350 m",
        landuse  = "vineyards / chalk grassland / mixed cereal",
        split    = "train",
        notes    = "Berry/Nivernais limestone hills — wide terraced Loire valley in frame",
    ),
    dict(
        id       = "tarn_millau",
        name     = "Gorges du Tarn (Lozère/Aveyron)",
        lon      = 3.08, lat = 44.10, box_size = 0.10,
        terrain  = "deeply incised limestone gorge, 300–1000 m",
        landuse  = "cliffside scrubland + mixed gorge forest + causses plateau",
        split    = "train",
        notes    = "Extreme canyon relief on Grands Causses — 600 m gorge walls",
    ),
    dict(
        id       = "landes_biscarrosse",
        name     = "Landes côte (Biscarrosse)",
        lon      = -1.17, lat = 44.65, box_size = 0.10,
        terrain  = "flat coastal dune and lake system, 0–30 m",
        landuse  = "coastal pine forest / dune / coastal lagoon",
        split    = "train",
        notes    = "Atlantic coastal dune forest — different from interior Landes zone",
    ),
    dict(
        id       = "bocage_virois",
        name     = "Bocage virois (Calvados/Orne)",
        lon      = -0.82, lat = 48.65, box_size = 0.10,
        terrain  = "dense bocage hills on Armorican crystalline, 150–300 m",
        landuse  = "hedgerow bocage / dairy farming / small woodlands",
        split    = "train",
        notes    = "Classic Norman bocage south of Caen — denser hedges than Caen zone",
    ),
    dict(
        id       = "alsace_vignoble",
        name     = "Alsace (Route des Vins, Haut-Rhin)",
        lon      = 7.27, lat = 48.10, box_size = 0.10,
        terrain  = "Vosges foothills vineyard slopes, 200–700 m",
        landuse  = "vineyard terraces + mixed Vosges forest above + Rhine plain below",
        split    = "train",
        notes    = "Steep vineyard slopes with abrupt contrast plain/crest",
    ),
    dict(
        id       = "pays_de_caux",
        name     = "Pays de Caux (Seine-Maritime)",
        lon      = 0.37, lat = 49.73, box_size = 0.10,
        terrain  = "chalk plateau with coastal cliffs and dry valleys, 0–130 m",
        landuse  = "arable chalk plateau / coastal scrub / sea-cliff grassland",
        split    = "train",
        notes    = "Caux chalk downs — abrupt 80 m sea cliffs (falaises) into Channel",
    ),
    dict(
        id       = "auvergne_puy_dome",
        name     = "Chaîne des Puys (Puy-de-Dôme)",
        lon      = 2.97, lat = 45.75, box_size = 0.10,
        terrain  = "volcanic cinder cones and lava flows, 400–1465 m",
        landuse  = "grassland / beech forest / crater lakes",
        split    = "train",
        notes    = "UNESCO World Heritage volcanic chain — unique conical micro-topography",
    ),

    # ── VAL (6 zones) ──────────────────────────────────────────────────────

    dict(
        id       = "paris_sud",
        name     = "Paris-Sud (Essonne)",
        lon      = 2.30, lat = 48.75, box_size = 0.10,
        terrain  = "flat to gently undulating plateau, 50–100 m",
        landuse  = "suburban residential / parks / farmland",
        split    = "val",
        notes    = "High building-density heterogeneity — primary validation zone",
    ),
    dict(
        id       = "anjou_loire",
        name     = "Anjou (Loire valley)",
        lon      = -0.53, lat = 47.47, box_size = 0.10,
        terrain  = "Loire floodplain and tuffeau terrace, 20–80 m",
        landuse  = "market gardening / vineyards / wetland / poplar plantations",
        split    = "val",
        notes    = "Flat Loire valley with alluvial terraces and tuffeau cliff benches",
    ),
    dict(
        id       = "herault_garrigue",
        name     = "Hérault (garrigue N. Montpellier)",
        lon      = 3.88, lat = 43.65, box_size = 0.10,
        terrain  = "Mediterranean limestone garrigue, 80–350 m",
        landuse  = "garrigue scrubland + vineyards + olive groves",
        split    = "val",
        notes    = "Dry Mediterranean karst with open scrub north of Montpellier",
    ),
    dict(
        id       = "lyon_est",
        name     = "Lyon Est (Rhône, Ain)",
        lon      = 5.08, lat = 45.65, box_size = 0.10,
        terrain  = "Rhône-Saône confluence terrace, 160–400 m",
        landuse  = "suburban / industrial / mixed periurban / forested plateau",
        split    = "val",
        notes    = "Dense periurban area east of Lyon — large industrial and residential mix",
    ),
    dict(
        id       = "strasbourg_ried",
        name     = "Strasbourg (Ried alsacien, Bas-Rhin)",
        lon      = 7.63, lat = 48.52, box_size = 0.10,
        terrain  = "Rhine alluvial plain and Ried wetland, 130–200 m",
        landuse  = "urban fringe / Rhine floodplain forest / wetland meadow",
        split    = "val",
        notes    = "Rhine floodplain with Ried riparian forest — very flat, high urban density",
    ),
    dict(
        id       = "bordeaux_medoc",
        name     = "Médoc (Gironde estuary)",
        lon      = -0.73, lat = 45.17, box_size = 0.10,
        terrain  = "flat estuary peninsula, 0–30 m",
        landuse  = "world-class vineyards / estuary marsh / pine forest strip",
        split    = "val",
        notes    = "Médoc peninsula — extreme flatness, vineyard canopy, tidal influence",
    ),

    # ── TEST (6 zones) ─────────────────────────────────────────────────────

    dict(
        id       = "toulouse",
        name     = "Toulouse (Haute-Garonne)",
        lon      = 1.45, lat = 43.60, box_size = 0.10,
        terrain  = "Garonne alluvial plain, 130–160 m",
        landuse  = "urban / periurban / floodplain agriculture",
        split    = "test",
        notes    = "Mediterranean-Atlantic city on flat alluvial plain — held-out test",
    ),
    dict(
        id       = "nice_arriere",
        name     = "Nice arrière-pays (Alpes-Mar.)",
        lon      = 7.08, lat = 43.75, box_size = 0.10,
        terrain  = "steep coastal limestone hills, 200–900 m",
        landuse  = "Mediterranean maquis + perched villages + olive groves",
        split    = "test",
        notes    = "Steep pre-Alpine limestone ridges behind the Côte d'Azur",
    ),
    dict(
        id       = "lorraine_moselle",
        name     = "Lorraine (Moselle plateau)",
        lon      = 6.35, lat = 49.30, box_size = 0.10,
        terrain  = "plateau and Moselle valley, 200–400 m",
        landuse  = "mixed forest + agricultural plateau + periurban",
        split    = "test",
        notes    = "Lorraine escarpment — forested plateau with incised river valley",
    ),
    dict(
        id       = "rennes_periurban",
        name     = "Rennes (Ille-et-Vilaine)",
        lon      = -1.68, lat = 48.10, box_size = 0.10,
        terrain  = "flat to gently rolling Breton sedimentary basin, 30–100 m",
        landuse  = "suburban / bocage farmland / periurban",
        split    = "test",
        notes    = "Fast-growing Breton city — medium-density suburban sprawl on low relief",
    ),
    dict(
        id       = "dijon_plateau",
        name     = "Côte-d'Or (Dijon plateau)",
        lon      = 4.93, lat = 47.22, box_size = 0.10,
        terrain  = "Côte escarpment and Saône plain, 180–500 m",
        landuse  = "world-class vineyards on escarpment / mixed crops / periurban",
        split    = "test",
        notes    = "Côte de Nuits/Beaune — sharp limestone cuesta above flat Saône plain",
    ),
    dict(
        id       = "picardie_somme",
        name     = "Picardie (Somme valley)",
        lon      = 2.78, lat = 49.92, box_size = 0.10,
        terrain  = "chalk downs and Somme valley, 20–160 m",
        landuse  = "large-scale arable / river marsh / periurban",
        split    = "test",
        notes    = "Picardie chalk plateau cut by broad Somme valley — low but variable relief",
    ),
]

# ---------------------------------------------------------------------------
# Convenience lookups
# ---------------------------------------------------------------------------

ZONES_BY_ID: dict[str, ZoneSpec] = {z["id"]: z for z in ZONES}

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
    print(f"{'ID':25s} {'Name':45s} {'Split':6s} {'Terrain'}")
    print("-" * 120)
    for z in ZONES:
        print(f"{z['id']:25s} {z['name']:45s} {z['split']:6s} {z['terrain']}")
    print(f"\nTrain ({len(TRAIN_ZONES)}): {TRAIN_ZONES}")
    print(f"Val   ({len(VAL_ZONES)}):  {VAL_ZONES}")
    print(f"Test  ({len(TEST_ZONES)}):  {TEST_ZONES}")
    print(f"\nTotal: {len(ZONES)} zones")
