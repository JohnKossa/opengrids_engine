"""Load a dataset from a comp_adj_app SQLite database.

This lets an office already running comp_adj_app reuse its imported data
and saved office-default config without re-exporting anything::

    from opengrids.compat.comp_adj_app import load
    dataset, config = load("comp_adj.db", prefix="sf")
    engine = CompGridEngine(dataset, config)

The database is opened read-only (``immutable=1``) and never written.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..dataset import Dataset
from ..fields import CostOfSale, FieldMeta, SpatialLevel


def _connect(db_path: str) -> sqlite3.Connection:
    uri = Path(db_path).resolve().as_uri() + "?immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def load(db_path: str, prefix: Optional[str] = None, *, universe_columns: Optional[Iterable[str]] = None,
         load_universe: bool = True) -> tuple[Dataset, dict]:
    """Return ``(Dataset, config)`` for one registered dataset.

    ``config`` is the office-default search config saved in
    ``dataset_registry``, with ``max_adjustment_pct`` folded into its
    ``grid`` block, or ``{}`` if none was saved. By default the universe is
    loaded with only the columns a grid can use (key, prediction,
    coordinates, every sales-side column the config or field metadata
    references, and ``address``). Pass ``universe_columns`` to choose them
    yourself.
    """
    conn = _connect(db_path)
    try:
        if prefix is None:
            row = conn.execute("SELECT table_prefix FROM dataset_registry ORDER BY id LIMIT 1").fetchone()
            if row is None:
                raise ValueError("no datasets registered in this database")
            prefix = row["table_prefix"]
        reg = conn.execute("SELECT * FROM dataset_registry WHERE table_prefix = ?", (prefix,)).fetchone()
        if reg is None:
            raise ValueError(f"unknown dataset prefix {prefix!r}")
        reg = dict(reg)

        sales = pd.read_sql_query(f"SELECT * FROM {prefix}_sales ORDER BY rowid", conn)
        contributions = pd.read_sql_query(f"SELECT * FROM {prefix}_contributions_sales ORDER BY rowid", conn)

        fields: dict[str, FieldMeta] = {}
        spatial_tagged: list[str] = []
        if _table_exists(conn, f"{prefix}_field_metadata"):
            for r in conn.execute(f"SELECT * FROM {prefix}_field_metadata").fetchall():
                r = dict(r)
                fields[r["field_name"]] = FieldMeta(
                    label=r.get("label"),
                    category=r.get("classification") or "unknown",
                    impact=r.get("expected_impact") or "mixed",
                    override_rate=r.get("override_rate"),
                    is_categorical=bool(r.get("is_categorical")),
                    is_boolean=bool(r.get("is_boolean")),
                )
                if r.get("is_spatial"):
                    spatial_tagged.append(r["field_name"])

        levels = {}
        if _table_exists(conn, f"{prefix}_spatial_levels"):
            for r in conn.execute(f"SELECT field_name, rank, clean_to_next FROM {prefix}_spatial_levels"):
                levels[r["field_name"]] = (r["rank"], bool(r["clean_to_next"]))
        big = 10 ** 9
        ladder = [SpatialLevel(f, levels.get(f, (big, False))[1])
                  for f in sorted(spatial_tagged, key=lambda f: (levels.get(f, (big, False))[0], f))]

        cos = None
        if reg.get("use_cost_of_sale"):
            cos = CostOfSale(pct=reg.get("cost_of_sale_pct") or 0.0,
                             source=reg.get("cost_of_sale_source") or "computed",
                             field=reg.get("cost_of_sale_field"),
                             model_trained_on_cost_of_sale=bool(reg.get("model_trained_on_cost_of_sale")))

        config: dict = {}
        if reg.get("default_search_config"):
            try:
                config = json.loads(reg["default_search_config"])
            except json.JSONDecodeError:
                config = {}
        mx = reg.get("max_adjustment_pct")
        if mx and mx > 0:
            config.setdefault("grid", {})["max_adjustment_pct"] = mx

        roles = dict(
            lat_field=reg.get("lat_field") or "latitude",
            lng_field=reg.get("lng_field") or "longitude",
            sale_price_field=reg.get("sale_price_field") or "sale_price",
            sale_price_time_adj_field=reg.get("sale_price_time_adj_field") or "sale_price_time_adj",
            sale_date_field=reg.get("sale_date_field") or "sale_date",
        )

        universe = None
        if load_universe:
            ucols = set(_columns(conn, f"{prefix}_universe"))
            if universe_columns is None:
                wanted = {"key", "prediction", "address", roles["lat_field"], roles["lng_field"],
                          *(l.field for l in ladder), *fields.keys(), *contributions.columns}
                wanted |= {f.get("field") for f in config.get("filters", []) if f.get("field")}
                wanted |= set((config.get("ranking") or {}).get("weights", {}) or {})
            else:
                wanted = {"key", *universe_columns}
            cols = [c for c in _columns(conn, f"{prefix}_universe") if c in wanted and c in ucols]
            sel = ", ".join(f"[{c}]" for c in cols)
            universe = pd.read_sql_query(f"SELECT {sel} FROM {prefix}_universe ORDER BY rowid", conn)

        ds = Dataset(sales, contributions, universe, fields=fields, spatial_ladder=ladder, cost_of_sale=cos,
                     valuation_date=reg.get("valuation_date"), **roles)
        return ds, config
    finally:
        conn.close()
