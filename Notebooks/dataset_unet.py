"""
dataset_unet.py
===============
Re-export shim so that  ``from dataset_unet import ...``  works from
4_train_unet.py and 2_baseline.py.

Python module names cannot start with a digit, so 3_dataset_unet.py is
loaded dynamically via importlib and its public symbols are re-exported here.
"""
import importlib.util
from pathlib import Path

_path = Path(__file__).with_name("3_dataset_unet.py")
_spec = importlib.util.spec_from_file_location("_dsm_dataset_impl", _path)
_m    = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

CHANNEL_SPEC                = _m.CHANNEL_SPEC
CHANNEL_NAMES               = _m.CHANNEL_NAMES
N_CHANNELS                  = _m.N_CHANNELS
NormStats                   = _m.NormStats
ZoneData                    = _m.ZoneData
load_zone                   = _m.load_zone
compute_normalization_stats = _m.compute_normalization_stats
load_normalization_stats    = _m.load_normalization_stats
DSMPatchDataset             = _m.DSMPatchDataset
