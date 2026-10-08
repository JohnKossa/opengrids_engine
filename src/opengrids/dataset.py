"""The in-memory dataset a :class:`~opengrids.CompGridEngine` works against."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from os import PathLike
from typing import Any, Optional, Union

import pandas as pd

from ._util import canon, py
from .errors import SubjectNotFoundError
from .fields import CostOfSale, FieldMeta, SpatialLevel

PathOrFrame = Union[str, PathLike, pd.DataFrame]

#: Contribution-table columns that describe the decomposition itself rather
#: than a model feature. These are never adjustment fields.
BOOKKEEPING_COLUMNS = ("key", "key_sale", "base_value", "contribution_sum", "prediction", "check_delta")


class Dataset:
    """Sales, SHAP contributions and (optionally) the universe, held in memory.

    The sales-side tables are scanned on every grid, so they're kept as
    DataFrames. The universe is used only to look up a subject by key: pass
    it if that's convenient, or pass subjects to
    :meth:`CompGridEngine.grid` directly as dicts or Series. For a large
    jurisdiction, keep the universe out of memory and stream subject rows
    from Parquet or SQL in chunks.

    Args:
        sales: One row per sale. It needs the key, sale-key, sale-price,
            time-adjusted-price and sale-date columns, plus every field used
            in filters, ranking or adjustments.
        contributions: One row per sale with the model's SHAP contribution
            for each feature, in dollars, keyed by the same ``key`` and
            ``key_sale``. It should also carry ``base_value`` (the SHAP
            expected value) and ``prediction``. Any model works; the engine
            never calls the model itself.
        universe: Optional. One row per parcel: the features plus
            ``prediction``. Used only by :meth:`subject`.
        fields: ``{column: FieldMeta}``. Fields not listed get defaults
            (continuous, impact ``"mixed"``).
        spatial_ladder: Spatial membership fields ordered finest to coarsest,
            each a ``SpatialLevel`` or a bare column name (unclean step).
        cost_of_sale: Optional :class:`CostOfSale`. ``None`` turns it off.
        valuation_date: Carried through to grid output for provenance.
        Column-name arguments: map your schema onto the engine's roles. The
            defaults follow OpenAVMKit naming.
    """

    def __init__(
        self,
        sales: pd.DataFrame,
        contributions: pd.DataFrame,
        universe: Optional[pd.DataFrame] = None,
        *,
        fields: Optional[Mapping[str, FieldMeta]] = None,
        spatial_ladder: Optional[Sequence[Union[SpatialLevel, str]]] = None,
        cost_of_sale: Optional[CostOfSale] = None,
        valuation_date: Optional[Union[str, date, datetime]] = None,
        key: str = "key",
        sale_key: str = "key_sale",
        lat_field: str = "latitude",
        lng_field: str = "longitude",
        sale_price_field: str = "sale_price",
        sale_price_time_adj_field: str = "sale_price_time_adj",
        sale_date_field: str = "sale_date",
        prediction_field: str = "prediction",
        base_value_field: str = "base_value",
    ):
        self.key = key
        self.sale_key = sale_key
        self.lat_field = lat_field
        self.lng_field = lng_field
        self.sale_price_field = sale_price_field
        self.sale_price_time_adj_field = sale_price_time_adj_field
        self.sale_date_field = sale_date_field
        self.prediction_field = prediction_field
        self.base_value_field = base_value_field

        missing = [c for c in (key, sale_price_time_adj_field, sale_date_field) if c not in sales.columns]
        if missing:
            raise ValueError(f"sales is missing required column(s): {missing}")
        if key not in contributions.columns:
            raise ValueError(f"contributions is missing the key column {key!r}")

        # Keys are compared as strings everywhere: parcel IDs often carry leading zeros.
        self.sales = sales.reset_index(drop=True).copy()
        self.sales[key] = self.sales[key].astype(str)
        self.contributions = contributions.reset_index(drop=True).copy()
        self.contributions[key] = self.contributions[key].astype(str)

        self._universe = None
        if universe is not None:
            if key not in universe.columns:
                raise ValueError(f"universe is missing the key column {key!r}")
            u = universe.copy()
            u[key] = u[key].astype(str)
            self._universe = u.drop_duplicates(subset=[key], keep="first").set_index(key, drop=False)

        self.fields: dict[str, FieldMeta] = dict(fields or {})
        self.spatial_ladder: list[SpatialLevel] = [
            lvl if isinstance(lvl, SpatialLevel) else SpatialLevel(str(lvl)) for lvl in (spatial_ladder or [])
        ]
        self.cost_of_sale = cost_of_sale
        self.valuation_date = valuation_date

        self._canon_cache: dict[str, pd.Series] = {}
        self._display_rows: Optional[dict[str, int]] = None

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_files(
        cls,
        sales: PathOrFrame,
        contributions: PathOrFrame,
        universe: Optional[PathOrFrame] = None,
        *,
        universe_columns: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> "Dataset":
        """Build from CSV or Parquet paths (or DataFrames) in any combination.

        ``universe_columns`` limits what's read from a large universe file.
        Key columns are read as strings so leading zeros survive.
        """
        key = kwargs.get("key", "key")
        sale_key = kwargs.get("sale_key", "key_sale")

        def _read(src: PathOrFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
            if isinstance(src, pd.DataFrame):
                return src if columns is None else src[list(columns)]
            path = str(src)
            if path.lower().endswith((".parquet", ".pq")):
                return pd.read_parquet(path, columns=list(columns) if columns else None)
            return pd.read_csv(path, dtype={key: str, sale_key: str}, usecols=columns, low_memory=False)

        uni_cols = None
        if universe_columns is not None:
            uni_cols = list(dict.fromkeys([key, *universe_columns]))
        return cls(
            _read(sales),
            _read(contributions),
            _read(universe, uni_cols) if universe is not None else None,
            **kwargs,
        )

    @classmethod
    def from_model(
        cls,
        model: Any,
        sales: pd.DataFrame,
        universe: Optional[pd.DataFrame] = None,
        *,
        features: Optional[Sequence[str]] = None,
        method: str = "auto",
        target: Any = "auto",
        background: Optional[pd.DataFrame] = None,
        fields: Optional[Mapping[str, FieldMeta]] = None,
        explain_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> "Dataset":
        """Build a Dataset straight from a fitted model.

        Explains ``sales`` with :func:`opengrids.contributions.explain`, then
        fills ``universe["prediction"]`` from the model if that column is
        missing. Any model input that isn't numeric is marked categorical
        unless ``fields`` says otherwise. ``background`` defaults to the
        universe, so the baseline is "the average parcel", or to the sales
        when no universe is given.
        """
        from .contributions import explain, predict_dollars

        key = kwargs.get("key", "key")
        sale_key = kwargs.get("sale_key", "key_sale")
        bg = background if background is not None else (universe if universe is not None else sales)
        contrib = explain(model, sales, features, background=bg, method=method, target=target,
                          key=key, sale_key=sale_key, **(explain_kwargs or {}))
        feats = [c for c in contrib.columns if c not in BOOKKEEPING_COLUMNS and c not in (key, sale_key)]
        pf = kwargs.get("prediction_field", "prediction")
        if universe is not None and pf not in universe.columns:
            universe = universe.copy()
            universe[pf] = predict_dollars(model, universe[feats], target)
        meta = dict(fields or {})
        for f in feats:
            if f not in meta and f in sales.columns and not pd.api.types.is_numeric_dtype(sales[f].dtype):
                meta[f] = FieldMeta(is_categorical=True)
        return cls(sales, contrib, universe, fields=meta, **kwargs)

    def check(self, config: Optional[dict] = None):
        """Pre-flight data report; see :mod:`opengrids.check`."""
        from .check import run_check

        return run_check(self, config)

    # ------------------------------------------------------------ introspection
    def meta(self, name: str) -> FieldMeta:
        return self.fields.get(name) or FieldMeta()

    def is_categorical(self, name: str) -> bool:
        return bool(self.meta(name).is_categorical)

    @property
    def bookkeeping_columns(self) -> set[str]:
        return {*BOOKKEEPING_COLUMNS, self.key, self.sale_key, self.base_value_field, self.prediction_field}

    @property
    def modeled_features(self) -> list[str]:
        """Numeric feature columns on ``contributions``, i.e. what the model explains."""
        return [
            c
            for c in self.contributions.columns
            if c not in self.bookkeeping_columns and pd.api.types.is_numeric_dtype(self.contributions[c].dtype)
        ]

    def integer_fields(self) -> set[str]:
        out = {c for c in self.sales.columns if pd.api.types.is_integer_dtype(self.sales[c].dtype)}
        for name, m in self.fields.items():
            if m.is_integer is True:
                out.add(name)
            elif m.is_integer is False:
                out.discard(name)
        return out

    def canon_column(self, col: str) -> pd.Series:
        """Cached ``sales[col].map(canon)`` for fast equality on text columns."""
        s = self._canon_cache.get(col)
        if s is None:
            s = self.sales[col].map(canon)
            self._canon_cache[col] = s
        return s

    # ---------------------------------------------------------------- lookups
    def subject(self, subject_key: Any) -> dict:
        """Universe row for ``subject_key`` as a plain dict (NaN -> None)."""
        if self._universe is None:
            raise SubjectNotFoundError(
                "no universe was given to this Dataset; pass the subject row itself instead of a key"
            )
        k = str(subject_key)
        if k not in self._universe.index:
            raise SubjectNotFoundError(k)
        return row_to_dict(self._universe.loc[k])

    def sales_row(self, sale_key_or_key: Any) -> Optional[dict]:
        """The sale row shown in a grid for a given property key.

        This is the last sale per key in table order.
        """
        if self._display_rows is None:
            last = self.sales[self.key].drop_duplicates(keep="last")
            self._display_rows = dict(zip(last.values, last.index))
        pos = self._display_rows.get(str(sale_key_or_key))
        return None if pos is None else row_to_dict(self.sales.loc[pos])

    def has_sale(self, key: Any) -> bool:
        return self.sales_row(key) is not None


def row_to_dict(row: Union[pd.Series, Mapping]) -> dict:
    """Series/mapping -> dict of native Python values with NaN -> None."""
    items = row.items() if isinstance(row, (pd.Series, Mapping)) else dict(row).items()
    return {k: py(v) for k, v in items}
