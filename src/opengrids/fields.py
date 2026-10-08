"""Per-field metadata and dataset-level settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Impact = Literal["positive", "negative", "mixed"]


@dataclass
class FieldMeta:
    """What the engine needs to know about one column.

    Attributes:
        label: Human-readable name used in grid output.
        category: Free-form grouping such as ``"land"`` or ``"improvement"``.
        impact: Expected direction of the field's effect on value. When it is
            ``"positive"`` or ``"negative"``, a computed rate with the opposite
            sign is clamped to 0 (the sign clamp) unless an override rate is set.
        override_rate: Fixed per-unit dollar rate that replaces the computed
            one. For categorical fields, use ``variant_overrides`` instead.
        variant_overrides: ``{variant: rate}`` overrides for a categorical field.
        is_categorical: Adjust by variant (``subject_rate - comp_rate``) rather
            than by a per-unit slope.
        is_boolean: Display hint only; the math treats booleans as 0/1.
        is_integer: Display hint (whole numbers). ``None`` means infer it
            from the sales column's dtype.
    """

    label: Optional[str] = None
    category: str = "unknown"
    impact: Impact = "mixed"
    override_rate: Optional[float] = None
    variant_overrides: dict = field(default_factory=dict)
    is_categorical: bool = False
    is_boolean: bool = False
    is_integer: Optional[bool] = None

    def as_dict(self, name: str) -> dict:
        return {
            "label": self.label or name,
            "category": self.category or "unknown",
            "impact": self.impact or "mixed",
            "override_rate": self.override_rate,
            "is_categorical": bool(self.is_categorical),
            "is_boolean": bool(self.is_boolean),
        }


@dataclass
class SpatialLevel:
    """One rung of the spatial ladder, ordered from finest to coarsest.

    ``clean_to_next`` means every member of this level nests entirely inside
    a single member of the next coarser level. On a clean step, peel-back can
    replace this level with the coarser one. On an unclean step, it adds the
    coarser level alongside this one, so a comp that already qualified can't
    fall out of the pool.
    """

    field: str
    clean_to_next: bool = False


@dataclass
class CostOfSale:
    """Cost-of-sale (level-of-assessment) handling.

    Attributes:
        pct: Percent deducted, e.g. ``15.0`` for Florida's 15%.
        source: ``"computed"`` takes the time-adjusted price times
            ``(100 - pct) / 100``. ``"field"`` reads ``field``, a column that
            already has the deduction applied.
        field: Column name, used only when ``source == "field"``.
        model_trained_on_cost_of_sale: Set this to True when the model's
            target already had cost of sale deducted. Otherwise the SHAP
            contributions, intercept and prediction are scaled by the same
            factor so they stay on the same basis as the sale prices.
    """

    pct: float = 0.0
    source: Literal["computed", "field"] = "computed"
    field: Optional[str] = None
    model_trained_on_cost_of_sale: bool = False

    @property
    def factor(self) -> float:
        return (100.0 - float(self.pct or 0.0)) / 100.0

    @property
    def model_shrink_factor(self) -> float:
        return 1.0 if self.model_trained_on_cost_of_sale else self.factor
