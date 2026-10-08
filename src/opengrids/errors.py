"""Exceptions raised by the engine."""
from __future__ import annotations


class OpenGridsError(Exception):
    """Base class for all library errors."""


class SubjectNotFoundError(OpenGridsError, KeyError):
    """A subject key was given but no universe row exists for it."""


class GridConfigError(OpenGridsError):
    """The search config references fields the dataset can't support.

    ``problems`` is a list of dicts, each with ``section``, ``field``,
    ``reason`` and ``suggestion`` keys, so a caller can list every bad field
    at once instead of failing on the first one.
    """

    def __init__(self, problems: list[dict]):
        self.problems = problems
        fields = ", ".join(str(p.get("field")) for p in problems)
        super().__init__(f"search config references unusable field(s): {fields}")
