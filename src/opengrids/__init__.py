"""opengrids: model-coherent comparable-sales grids for any AVM.

Give the engine your sales, your model's per-sale SHAP contributions and a
search config. For each subject it finds comps (relaxing filters until the
local fit has enough support), derives per-unit adjustments that agree with
the model's own opinion, and returns the full adjustment grid.
"""
from .check import CheckReport, DataCheckError, Issue
from .config import DEFAULT_CONFIG, normalize_config
from .dataset import Dataset
from .engine import CompGridEngine
from .errors import GridConfigError, OpenGridsError, SubjectNotFoundError
from .fields import CostOfSale, FieldMeta, SpatialLevel
from .grid import CompGrid

__all__ = [
    "CompGridEngine", "CompGrid", "Dataset", "FieldMeta", "SpatialLevel", "CostOfSale",
    "DEFAULT_CONFIG", "CheckReport", "DataCheckError", "Issue", "normalize_config", "GridConfigError", "OpenGridsError", "SubjectNotFoundError",
]
__version__ = "0.1.0"
