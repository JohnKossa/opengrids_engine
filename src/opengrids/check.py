"""Pre-flight data check: find what will break or quietly degrade grids, before running any.

    report = engine.check()          # or dataset.check(config)
    print(report)                    # grouped, plain-language report
    report.raise_if_errors()

Severity:

* ``error``: grids will fail or come out wrong for some subjects.
* ``warning``: grids run, but something silently degrades (fewer sales in
  the fit, a $0 adjustment, a filter that does nothing).
* ``info``: worth knowing, e.g. which fields will go through FWL.

What's checked:

1. **Model contributions (SHAP)**: sales with no contribution row,
   orphan or duplicate rows, missing contribution values, non-numeric
   contribution columns, and whether ``base_value + Σ contributions =
   prediction``.
2. **Adjustment-field types**: numbers stored as text, text not marked
   categorical, the same field typed differently in sales and universe,
   date columns, and "categorical" fields that look continuous.
3. **Nulls in adjustment fields**: in sales (a missing value in a
   continuous modeled field makes the rate fit fail for any subject whose
   pool includes that sale) and in the universe (the subject's adjustment for
   that field becomes $0).
4. **Schema and config**: required columns, and config entries (filters,
   ranking weights, overrides) that name a column that doesn't exist. Those
   are otherwise skipped without a word.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from .config import normalize_config
from .errors import OpenGridsError

if TYPE_CHECKING:  # pragma: no cover
    from .dataset import Dataset

SEVERITIES = ("error", "warning", "info")


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    field: Optional[str] = None
    count: Optional[int] = None
    examples: list = dc_field(default_factory=list)
    fix: Optional[str] = None


class DataCheckError(OpenGridsError):
    def __init__(self, report: "CheckReport"):
        self.report = report
        super().__init__(f"data check found {len(report.errors)} error(s):\n{report}")


@dataclass
class CheckReport:
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> "CheckReport":
        if self.errors:
            raise DataCheckError(self)
        return self

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{**i.__dict__, "examples": ", ".join(map(str, i.examples))} for i in self.issues],
                            columns=["severity", "code", "field", "count", "message", "examples", "fix"])

    def __str__(self) -> str:
        n = {s: sum(i.severity == s for i in self.issues) for s in SEVERITIES}
        head = f"opengrids data check: {n['error']} error(s), {n['warning']} warning(s), {n['info']} note(s)"
        if not self.issues:
            return head + "\n  nothing to report"
        lines = [head]
        for sev in SEVERITIES:
            for i in (x for x in self.issues if x.severity == sev):
                tag = {"error": "ERROR", "warning": "WARN ", "info": "note "}[sev]
                where = f" [{i.field}]" if i.field else ""
                lines.append(f"  {tag}{where} {i.message}")
                if i.examples:
                    lines.append(f"         e.g. {', '.join(map(str, i.examples))}")
                if i.fix:
                    lines.append(f"         fix: {i.fix}")
        return "\n".join(lines)

    __repr__ = __str__

    def _repr_html_(self) -> str:  # nice in notebooks
        return self.to_frame().to_html(index=False)


def _pct(n: int, d: int) -> str:
    return f"{n:,} of {d:,} ({n / d:.1%})" if d else f"{n:,}"


def _ex(values, k: int = 5) -> list:
    return [str(v) for v in list(dict.fromkeys(values))[:k]]


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s.dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return "date"
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "category"
    return "text"


def run_check(ds: "Dataset", config: Optional[dict] = None) -> CheckReport:
    cfg = normalize_config(config)
    adj_fields = cfg.get("adj_fields")
    if adj_fields is None:
        adj_fields = ds.modeled_features
    issues: list[Issue] = []
    add = issues.append
    sales, contrib, uni = ds.sales, ds.contributions, ds._universe
    k, sk = ds.key, ds.sale_key
    n_sales = len(sales)

    # ------------------------------------------------------------ 4. schema
    for col, why in ((ds.sale_price_time_adj_field, "the baseline every adjustment is added to"),
                     (ds.sale_date_field, "needed to keep each parcel's latest sale"),
                     (ds.lat_field, "needed for radius filters and distances"),
                     (ds.lng_field, "needed for radius filters and distances")):
        if col not in sales.columns:
            add(Issue("error" if col == ds.sale_price_time_adj_field else "warning", "missing_column",
                      f"sales has no '{col}' column ({why}).", col,
                      fix="rename your column or pass the matching *_field= argument to Dataset"))
        elif col in (ds.sale_price_time_adj_field, ds.lat_field, ds.lng_field) and _kind(sales[col]) != "numeric":
            add(Issue("error", "wrong_type", f"sales '{col}' is {_kind(sales[col])}, not numeric.", col,
                      fix=f"sales['{col}'] = pd.to_numeric(sales['{col}'], errors='coerce')"))
    if ds.sale_date_field in sales.columns:
        bad = pd.to_datetime(sales[ds.sale_date_field], errors="coerce").isna() & sales[ds.sale_date_field].notna()
        if bad.any():
            add(Issue("warning", "bad_dates", f"{_pct(int(bad.sum()), n_sales)} sale dates don't parse as dates.",
                      ds.sale_date_field, int(bad.sum()), _ex(sales.loc[bad, ds.sale_date_field])))
    if ds.sale_price_time_adj_field in sales.columns:
        nul = sales[ds.sale_price_time_adj_field].isna()
        if nul.any():
            add(Issue("warning", "null_price", f"{_pct(int(nul.sum()), n_sales)} sales have no time-adjusted price; "
                      "they can't be comps.", ds.sale_price_time_adj_field, int(nul.sum()), _ex(sales.loc[nul, k])))
    if uni is not None and ds.prediction_field not in uni.columns:
        add(Issue("warning", "no_prediction", f"universe has no '{ds.prediction_field}' column, so closest-price "
                  "ranking, bracketing and comp-match metrics have nothing to compare to.", ds.prediction_field,
                  fix="add the model's value for every parcel, or build the Dataset with Dataset.from_model(...)"))

    # ----------------------------------------------------------- 1. SHAP rows
    on = sk if (sk in sales.columns and sk in contrib.columns) else k
    s_ids, c_ids = sales[on].astype(str), contrib[on].astype(str)
    for name, ids in (("sales", s_ids), ("contributions", c_ids)):
        dup = ids.duplicated(keep=False)
        if dup.any():
            add(Issue("error", "duplicate_rows", f"{name} has {int(dup.sum()):,} rows sharing a '{on}' with another "
                      f"row, so sales and contributions can't be matched one-to-one.", on, int(dup.sum()),
                      _ex(ids[dup]), fix="each sale should appear exactly once in each table"))
    missing = ~s_ids.isin(set(c_ids))
    if missing.any():
        frac = missing.mean()
        add(Issue("error" if frac > 0.5 else "warning", "missing_contributions",
                  f"{_pct(int(missing.sum()), n_sales)} sales have no row of model contributions. They can still "
                  "appear as comps, but they're left out when the adjustment rates are fitted.", on,
                  int(missing.sum()), _ex(s_ids[missing]),
                  fix="explain the same sales table the grids use (see opengrids.contributions.explain)"))
    orphan = ~c_ids.isin(set(s_ids))
    if orphan.any():
        add(Issue("warning", "orphan_contributions", f"{_pct(int(orphan.sum()), len(contrib))} contribution rows "
                  "match no sale and are ignored.", on, int(orphan.sum()), _ex(c_ids[orphan])))

    feature_cols = [c for c in contrib.columns if c not in ds.bookkeeping_columns]
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(contrib[c].dtype):
            add(Issue("error" if c in adj_fields else "warning", "contribution_not_numeric",
                      f"contribution column '{c}' is {_kind(contrib[c])}. Contributions are dollar amounts and must be "
                      "numeric, even for a categorical feature. The engine won't treat this field as modeled.", c,
                      fix=f"contributions['{c}'] = pd.to_numeric(contributions['{c}'], errors='coerce')"))
    modeled = ds.modeled_features
    for c in modeled:
        nul = contrib[c].isna()
        if nul.any():
            add(Issue("warning", "null_contributions", f"{_pct(int(nul.sum()), len(contrib))} contribution rows have no "
                      f"value for '{c}'; the rate fit treats them as missing data.", c, int(nul.sum()),
                      _ex(contrib.loc[nul, on])))
    bv, pf = ds.base_value_field, ds.prediction_field
    if modeled and bv in contrib.columns and pf in contrib.columns:
        total = contrib[bv] + contrib[modeled].sum(axis=1, min_count=1)
        rel = ((contrib[pf] - total).abs() / contrib[pf].abs().clip(lower=1.0)).replace([np.inf], np.nan)
        off = rel > 1e-3
        if off.any():
            med = float(rel[off].median())
            own = " (your table's own check_delta column shows the same gap)" if "check_delta" in contrib.columns else ""
            systematic = med > 0.10 or off.mean() > 0.5
            add(Issue("error" if systematic else "warning", "not_additive",
                      f"on {_pct(int(off.sum()), len(contrib))} rows, base_value + contributions misses the "
                      f"prediction by more than 0.1% (median miss {med:.1%}){own}. "
                      + ("A gap this large or this widespread usually means the values aren't in dollars (e.g. log space) or "
                         "belong to a different model run, and the rates would be wrong."
                         if systematic else
                         "Small gaps are common with approximate explainers or ensembles and don't change the "
                         "rates much, since the grid uses how each contribution varies with its feature. A feature "
                         "missing from the table would also cause this."),
                      None, int(off.sum()), _ex(contrib.loc[off, on]),
                      fix="regenerate with opengrids.contributions.explain for an exact decomposition"
                      if systematic else None))
    elif modeled:
        add(Issue("info", "additivity_unchecked", f"contributions has no '{bv}' and/or '{pf}' column, so additivity "
                  "can't be verified (and surrogate fidelity needs the prediction)."))

    # ----------------------------------------------- 2 & 3. adjustment fields
    for f in adj_fields:
        meta = ds.meta(f)
        in_sales = f in sales.columns
        in_uni = uni is not None and f in uni.columns
        if not in_sales:
            add(Issue("error", "adj_field_missing", f"adjustment field '{f}' is not a sales column.", f,
                      fix="check the spelling, or remove it from adj_fields"))
            continue
        if uni is not None and not in_uni:
            add(Issue("error", "adj_field_missing", f"adjustment field '{f}' is not in the universe, so subjects "
                      "have no value to compare.", f, fix="include it in the universe (universe_columns=...)"))

        s = sales[f]
        kind = _kind(s)
        if kind == "date":
            add(Issue("error", "adj_field_date", f"'{f}' is a date. Adjustments need numbers or categories.", f,
                      fix="convert to a number such as age in years, or a year"))
        elif kind in ("text", "category") and not meta.is_categorical:
            parsed = pd.to_numeric(s.dropna().astype(str).str.replace(",", "").str.strip(), errors="coerce")
            if len(parsed) and parsed.notna().all():
                add(Issue("error", "text_not_categorical", f"'{f}' holds numbers stored as text (e.g. "
                          f"{_ex(s.dropna(), 3)}) and isn't marked categorical, so subtracting subject and comp "
                          "values will fail.", f,
                          fix=f"a measurement: sales['{f}'] = pd.to_numeric(sales['{f}']) (and the same in the "
                              f"universe); a code: fields['{f}'] = FieldMeta(is_categorical=True)"))
            else:
                add(Issue("error", "text_not_categorical", f"'{f}' is text (e.g. {_ex(s.dropna(), 3)}) but isn't "
                          "marked categorical.", f, fix=f"fields['{f}'] = FieldMeta(is_categorical=True)"))
        elif kind == "numeric" and meta.is_categorical:
            nun = s.nunique(dropna=True)
            if nun > 50 or (len(s) and nun > 0.2 * s.notna().sum() and nun > 10):
                add(Issue("warning", "many_variants", f"'{f}' is marked categorical and has {nun:,} distinct values. "
                          "Each value gets its own rate and counts as a parameter for peel-back, so many rates will "
                          "rest on one or two sales.", f,
                          fix="if it's a measurement, drop is_categorical to fit one per-unit rate; if it's a code "
                              "such as a neighborhood ID, consider using it as a spatial filter instead"))

        if in_uni:
            uk = _kind(uni[f])
            if uk != kind and not ({uk, kind} <= {"numeric", "boolean"}):
                add(Issue("error", "type_mismatch", f"'{f}' is {kind} in sales but {uk} in the universe, so subject "
                          "and comp values won't compare.", f, examples=[f"sales: {_ex(s.dropna(), 2)}",
                                                                         f"universe: {_ex(uni[f].dropna(), 2)}"],
                          fix="load both tables with the same dtype for this column"))

        nul = s.isna()
        if nul.any():
            continuous_modeled = f in modeled and not meta.is_categorical
            if continuous_modeled:
                msg = (f"{_pct(int(nul.sum()), n_sales)} sales have no value. Any subject whose candidate pool "
                       "includes one of them will fail to compute this rate (the grid raises GridConfigError).")
                sev, fix = "error", "fill the values, drop those sales, or remove the field from adj_fields"
            elif meta.is_categorical:
                msg = (f"{_pct(int(nul.sum()), n_sales)} sales have no value; they are treated as their own "
                       "'None' variant.")
                sev, fix = "info", None
            else:
                msg = (f"{_pct(int(nul.sum()), n_sales)} sales have no value; those sales are dropped from this "
                       "field's FWL fit, and a comp missing it gets a $0 adjustment.")
                sev, fix = "warning", None
            add(Issue(sev, "nulls_in_sales", msg, f, int(nul.sum()), _ex(sales.loc[nul, k]), fix))
        if in_uni:
            un = uni[f].isna()
            if un.any():
                add(Issue("warning", "nulls_in_universe", f"{_pct(int(un.sum()), len(uni))} parcels have no value; "
                          "for those subjects this field's adjustment is $0 on every comp.", f, int(un.sum()),
                          _ex(uni.loc[un, k])))

    fwl = [f for f in adj_fields if f in sales.columns and f not in modeled]
    if fwl:
        add(Issue("info", "fwl_fields", f"not model features, so their rates are fitted with Frisch-Waugh-Lovell "
                  f"on sale prices (holding the model's opinion fixed): {', '.join(fwl)}.",
                  fix="if you expected the model to have one of these, check the contribution column's spelling"))

    # ------------------------------------------------- 4. config references
    cols = set(sales.columns)
    for flt in cfg.get("filters", []):
        fld = flt.get("field")
        if flt.get("enabled", True) and fld and fld not in cols:
            add(Issue("warning", "unknown_filter_field", f"filter '{flt.get('id')}' ({flt.get('kind')}) uses "
                      f"'{fld}', which is not a sales column, so the filter is skipped.", fld,
                      fix="fix the field name or remove the filter"))
        if flt.get("enabled", True) and flt.get("kind") == "spatial" and not ds.spatial_ladder:
            add(Issue("warning", "no_spatial_ladder", "the config has a spatial filter but the Dataset has no "
                      "spatial_ladder, so it does nothing.", fix="pass spatial_ladder=[...] to Dataset"))
    for lvl in ds.spatial_ladder:
        if lvl.field not in cols:
            add(Issue("warning", "unknown_ladder_field", f"spatial ladder level '{lvl.field}' is not a sales column.",
                      lvl.field))
    rank = cfg.get("ranking", {}) or {}
    if rank.get("method") == "weighted_euclidean":
        for w in (rank.get("weights") or {}):
            if w not in cols:
                add(Issue("warning", "unknown_weight_field", f"ranking weight on '{w}' is ignored; not a sales column.", w))
    for o in (cfg.get("adjustment_overrides") or {}):
        base = o.split(":", 1)[0]
        if base not in adj_fields:
            add(Issue("warning", "override_unused", f"override '{o}' has no effect: '{base}' is not in adj_fields.", base))

    order = {s: i for i, s in enumerate(SEVERITIES)}
    issues.sort(key=lambda i: order[i.severity])
    return CheckReport(issues)
