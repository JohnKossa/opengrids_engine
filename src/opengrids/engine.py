"""The main entry point: :class:`CompGridEngine`."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Iterable, Iterator, Optional, Union

import pandas as pd

from .adjustments import compute_adjustments
from .config import normalize_config
from .dataset import Dataset, row_to_dict
from .filters import build_filter_stack, peel_policy
from .finder import CompFinder
from .grid import CompGrid, build_grid

SubjectLike = Union[str, int, Mapping, pd.Series]


class CompGridEngine:
    """Builds comp grids against one :class:`Dataset` and a search config.

    The engine holds no state between calls beyond the dataset and the
    config, so it's safe to keep one around for the life of a process and
    call it repeatedly::

        engine = CompGridEngine(dataset, config={"adj_fields": [...]})
        grid = engine.grid("10168801")          # key -> looked up in dataset.universe
        grid = engine.grid(subject_row_dict)    # or pass the row itself
        grid.to_frame(); grid.to_dict(); grid.metrics()

    For a one-off variation, pass ``config=`` to :meth:`grid` (merged over
    the engine's config) or derive a new engine with :meth:`with_config`.
    """

    def __init__(self, dataset: Dataset, config: Optional[dict] = None):
        self.dataset = dataset
        self._config = self._resolve(config)

    # ------------------------------------------------------------------ config
    def _resolve(self, cfg: Optional[dict], base: Optional[dict] = None) -> dict:
        out = normalize_config(cfg, base=base)
        if out.get("adj_fields") is None:
            out["adj_fields"] = self.dataset.modeled_features
        return out

    @property
    def config(self) -> dict:
        """A copy of the resolved config."""
        return copy.deepcopy(self._config)

    def with_config(self, config: dict) -> "CompGridEngine":
        """A new engine on the same dataset, with ``config`` merged over this
        engine's config."""
        eng = CompGridEngine.__new__(CompGridEngine)
        eng.dataset = self.dataset
        eng._config = self._resolve(config, base=self._config)
        return eng

    def _cfg(self, config: Optional[dict]) -> dict:
        return self._config if config is None else self._resolve(config, base=self._config)

    def _subject(self, subject: SubjectLike) -> dict:
        if isinstance(subject, (str, int)):
            return self.dataset.subject(subject)
        if isinstance(subject, pd.Series):
            return row_to_dict(subject)
        return row_to_dict(dict(subject))

    def check(self, config: Optional[dict] = None):
        """Pre-flight data report for this engine's dataset and config (see :mod:`opengrids.check`)."""
        from .check import run_check

        return run_check(self.dataset, self._cfg(config))

    # -------------------------------------------------------------- public API
    def grid(self, subject: SubjectLike, *, user_comp: Optional[Any] = None,
             config: Optional[dict] = None) -> CompGrid:
        """Build the full-detail grid for one subject.

        Args:
            subject: A key (requires ``dataset.universe``) or the subject's
                row as a dict or Series. The row needs the key, the
                adjustment fields and, ideally, ``prediction``.
            user_comp: Optional key of a hand-picked comp. It always takes
                a grid slot and is exempt from the comp-quality checks.
            config: Optional per-call config, merged over the engine's.
        """
        return build_grid(self.dataset, self._subject(subject), self._cfg(config),
                          user_comp_id=None if user_comp is None else str(user_comp))

    def grids(self, subjects: Iterable[SubjectLike], *, config: Optional[dict] = None,
              errors: str = "raise") -> Iterator[tuple[Any, Union[CompGrid, Exception]]]:
        """Yield ``(subject, grid)`` for many subjects. With
        ``errors="return"``, a failed subject yields its exception instead
        of stopping the batch."""
        cfg = self._cfg(config)
        for s in subjects:
            try:
                yield s, build_grid(self.dataset, self._subject(s), cfg)
            except Exception as e:  # noqa: BLE001
                if errors != "return":
                    raise
                yield s, e

    def candidates(self, subject: SubjectLike, *, config: Optional[dict] = None) -> tuple[pd.DataFrame, list[dict]]:
        """Peel-back candidate pool for a subject, and the peel trace."""
        cfg = self._cfg(config)
        subj = self._subject(subject)
        ds = self.dataset
        fields = cfg.get("adj_fields") or []

        def _pc(c: pd.DataFrame) -> int:
            return sum(((int(c[f].nunique(dropna=True)) or 1) if (not c.empty and f in c.columns) else 1)
                       if ds.is_categorical(f) else 1 for f in fields)

        return CompFinder(ds).find_candidates_peeled(subj, build_filter_stack(cfg, ds.spatial_ladder),
                                                     peel_policy(cfg), param_count=_pc)

    def adjustments(self, subject: SubjectLike, comp_keys: Iterable[Any], *, config: Optional[dict] = None) -> dict:
        """Unit adjustments fitted over an explicit pool (the subject plus
        ``comp_keys``), skipping discovery. See
        :func:`opengrids.adjustments.compute_adjustments` for the result's
        shape."""
        cfg = self._cfg(config)
        subj = self._subject(subject)
        key = str(subj.get(self.dataset.key))
        return compute_adjustments(self.dataset, key, [key, *map(str, comp_keys)], cfg)
