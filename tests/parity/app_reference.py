"""Run the original comp_adj_app grid code, read-only, as a reference.

Imports comp_adj_app's modules from ``COMP_ADJ_APP`` (the checkout folder)
and points every SQLite connection at an ``immutable=1`` read-only handle on
its ``comp_adj.db``, so nothing in the app's database can change. The photo
module (which needs SQL Server / ODBC) is stubbed out.
"""
from __future__ import annotations

import copy
import os
import sqlite3
import sys
import types
from pathlib import Path


class AppReference:
    def __init__(self, app_dir: str, prefix: str = "sf", fwl_params_from_sales: bool = False):
        self.app_dir = str(Path(app_dir).resolve())
        self.db_path = os.path.join(self.app_dir, "comp_adj.db")
        self.prefix = prefix
        sys.dont_write_bytecode = True  # leave no __pycache__ in the app checkout
        if self.app_dir not in sys.path:
            sys.path.insert(0, self.app_dir)

        # Stub photos (pyodbc/SQL Server) before grid_service imports it.
        photos = types.ModuleType("photos")

        class PhotoFetcher:  # noqa: D401 - stub
            def __init__(self, *a, **k): ...
            def get_photo_ids_bulk(self, *a, **k): return None

        photos.PhotoFetcher = PhotoFetcher
        photos.fetch_photo_bytes = lambda *a, **k: None
        photos.fetch_photo_data_uris_bulk = lambda *a, **k: {}
        sys.modules["photos"] = photos

        uri = Path(self.db_path).as_uri() + "?immutable=1"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if fwl_params_from_sales:
            # The app's FWL step reads SHAP-feature *values* from
            # {prefix}_params_sales, which holds per-row model parameters,
            # not feature values. Shadow it with the sales table so the
            # reference computes what the code intends. A TEMP view lives in
            # memory only and shadows the main-schema table of the same name.
            self.conn.execute(f"CREATE TEMP VIEW {prefix}_params_sales AS SELECT * FROM {prefix}_sales")

        # helpers.py opens (and would create) a database at import time via a
        # relative path. Route every sqlite3.connect during import to the
        # read-only handle so nothing is created or written anywhere.
        real_connect = sqlite3.connect
        sqlite3.connect = lambda *a, **k: self.conn  # type: ignore[assignment]
        try:
            import helpers  # type: ignore
            import adjustment_engine  # type: ignore
            import comp_finder  # type: ignore
            import grid_service  # type: ignore
        finally:
            sqlite3.connect = real_connect

        getter = lambda db_path=None: self.conn  # noqa: E731
        for mod in (helpers, adjustment_engine, comp_finder):
            mod.get_pooled_connection = getter
        helpers.DB_PATH = self.db_path
        grid_service.DB_PATH = self.db_path
        self.helpers, self.grid_service = helpers, grid_service
        self._max_adj = grid_service.get_dataset_max_adjustment_pct

    def config(self, overrides: dict | None = None) -> dict:
        """The office-default config with ``overrides`` applied at the top
        level, normalized by the app's own normalizer."""
        base = self.helpers.normalize_search_config(None, ptype=self.prefix)
        cfg = copy.deepcopy(base)
        for k, v in (overrides or {}).items():
            if k == "grid":
                continue
            cfg[k] = copy.deepcopy(v)
        return self.helpers.normalize_search_config(cfg, ptype=self.prefix)

    def grid(self, subject_id: str, cfg: dict, user_comp_id: str | None = None,
             max_adjustment_pct: float | None = None) -> dict | None:
        gs = self.grid_service
        gs.get_dataset_max_adjustment_pct = (lambda p: max_adjustment_pct) if max_adjustment_pct else self._max_adj
        try:
            return gs.get_grid_data(self.prefix, subject_id, cfg, user_comp_id=user_comp_id)
        finally:
            gs.get_dataset_max_adjustment_pct = self._max_adj
