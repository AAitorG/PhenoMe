"""
Metadata source implementations: Default, PathTemplate, DataFrame.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd

from ..utils.path_utils import filename_identifier_keys, normalize_identifier_path
from .base import MetadataBase


class DefaultMetadata(MetadataBase):
    """Minimal metadata extractor: file_path and filename from path.

    Single filename column. Auto-generates ID from path when not provided.
    """

    def _extract(self, path: str) -> dict[str, Any]:
        """Return ``file_path`` and filename stem for ``path``."""
        filename_stem = os.path.splitext(os.path.basename(path))[0]
        return {"file_path": path, "filename": filename_stem}


class PathTemplateMetadata(MetadataBase):
    """Metadata extractor from path template with capture groups.

    Uses parentheses for capture groups, e.g. ``.../(drug)/(time)/(crop_name).*``.
    group_by is the last capture group.
    """

    def __init__(
        self,
        template: str,
        filename_columns: str | list[str] = "filename",
        unique_id_column: str = "id",
        mask_filename_column: str | None = None,
        mask_dir: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        super().__init__(
            filename_columns=filename_columns,
            unique_id_column=unique_id_column,
            mask_filename_column=mask_filename_column,
            mask_dir=mask_dir,
            data_dir=data_dir,
        )
        self._template = template
        self._compiled_re, self._placeholders = self._build_regex()

    def _build_regex(self) -> tuple:
        """Compile the path template into a regex and return ``(compiled, placeholders)``."""
        clean_template = self._template.replace("\\", "/")
        match_suffix = False
        if clean_template.startswith("./"):
            clean_template = clean_template[2:]
            match_suffix = True
        elif clean_template.startswith(".../"):
            clean_template = clean_template[4:]
            match_suffix = True

        parts = re.split(r"\((\w+)\)", clean_template)
        placeholders = [p for i, p in enumerate(parts) if i % 2 == 1]

        regex_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                p = re.escape(part)
                if p.endswith(r"\.\*"):
                    p = p[:-4] + r"\.[^/]+"
                p = p.replace(r"\*", r"[^/]*")
                regex_parts.append(p)
            else:
                regex_parts.append(f"(?P<{part}>[^/]+)")

        regex_str = "".join(regex_parts) + "$"
        is_absolute = clean_template.startswith("/") or re.match(r"^[a-zA-Z]:", clean_template)
        if match_suffix or not is_absolute:
            regex_str = r"(?:^|/)" + regex_str
            if not regex_str.startswith(".*"):
                regex_str = ".*?" + regex_str

        compiled = re.compile(regex_str)
        return compiled, placeholders

    @property
    def group_by(self) -> str:
        """Return the last capture group name, or ``filename`` if there are no groups."""
        return self._placeholders[-1] if self._placeholders else "filename"

    def _extract(self, path: str) -> dict[str, Any]:
        """Parse ``path`` with the compiled template; return capture groups plus file fields."""
        normalized_path = path.replace("\\", "/")
        match = self._compiled_re.search(normalized_path)
        if match:
            meta: dict[str, Any] = dict(match.groupdict())
            meta["file_path"] = path
            meta["filename"] = os.path.splitext(os.path.basename(path))[0]
            return dict(meta)
        return {}


class DataFrameMetadata(MetadataBase):
    """Metadata lookup from DataFrame. Supports single and multi-channel modes."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        filename_columns: str | list[str] = "filename",
        unique_id_column: str = "id",
        mask_filename_column: str | None = None,
        mask_dir: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        super().__init__(
            filename_columns=filename_columns,
            unique_id_column=unique_id_column,
            mask_filename_column=mask_filename_column,
            mask_dir=mask_dir,
            data_dir=data_dir,
        )
        self._df = metadata_df
        self._multi_channel = len(self._filename_columns) > 1
        self._lookup_df: pd.DataFrame | None = None
        self._key_to_row_ix: dict[str, int] | None = None
        self._stem_to_row_ch: dict[str, tuple] | None = None
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Index the dataframe by filename stem (single- or multi-channel)."""
        for col in self._filename_columns:
            if col not in self._df.columns:
                raise ValueError(
                    f"metadata_df must contain a '{col}' column (from filename_columns)."
                )

        if not self._multi_channel:
            self._lookup_df = self._df.copy()
            key_to_row_ix: dict[str, int] = {}
            for row_ix in range(len(self._df)):
                val = self._df.iloc[row_ix][self._filename_columns[0]]
                for key in filename_identifier_keys(val):
                    if key in key_to_row_ix and key_to_row_ix[key] != row_ix:
                        raise ValueError(
                            f"Duplicate metadata filename key {key!r} at rows "
                            f"{key_to_row_ix[key]} and {row_ix} in column "
                            f"{self._filename_columns[0]!r}."
                        )
                    key_to_row_ix[key] = row_ix
            self._key_to_row_ix = key_to_row_ix
            self._stem_to_row_ch = None
        else:
            self._lookup_df = None
            self._key_to_row_ix = None

            def _stem(s: Any) -> str:
                normalized = normalize_identifier_path(s)
                if os.path.splitext(normalized)[1]:
                    normalized = os.path.splitext(normalized)[0]
                return str(normalized)

            stem_to_row_ch: dict[str, tuple] = {}
            for row_ix in range(len(self._df)):
                for ch_idx, col in enumerate(self._filename_columns):
                    val_str = str(self._df.iloc[row_ix][col]).strip()
                    for key in filename_identifier_keys(val_str):
                        if key in stem_to_row_ch and stem_to_row_ch[key] != (row_ix, ch_idx):
                            prev_row, prev_ch = stem_to_row_ch[key]
                            raise ValueError(
                                f"Duplicate metadata filename key {key!r} at rows "
                                f"({prev_row}, ch {prev_ch}) and ({row_ix}, ch {ch_idx})."
                            )
                        stem_to_row_ch[key] = (row_ix, ch_idx)
                    stem = _stem(val_str)
                    if stem in stem_to_row_ch and stem_to_row_ch[stem] != (row_ix, ch_idx):
                        prev_row, prev_ch = stem_to_row_ch[stem]
                        raise ValueError(
                            f"Duplicate metadata filename stem {stem!r} at rows "
                            f"({prev_row}, ch {prev_ch}) and ({row_ix}, ch {ch_idx})."
                        )
                    stem_to_row_ch[stem] = (row_ix, ch_idx)
            self._stem_to_row_ch = stem_to_row_ch

    def _extract(self, path: str) -> dict[str, Any]:
        """Look up metadata for ``path`` in the configured dataframe index."""
        return self._extract_with_data_dir(path, self._data_dir)

    def _extract_with_data_dir(self, path: str, data_dir: str | None = None) -> dict[str, Any]:
        """Look up metadata for ``path`` using partial-path-aware filename keys."""
        filename = os.path.basename(path)
        filename_stem = os.path.splitext(filename)[0]
        lookup_keys = filename_identifier_keys(path, data_dir or self._data_dir)

        if not self._multi_channel:
            if self._lookup_df is None or self._key_to_row_ix is None:
                return {}
            row_ix = None
            for key in lookup_keys:
                if key in self._key_to_row_ix:
                    row_ix = self._key_to_row_ix[key]
                    break
            if row_ix is None:
                raise KeyError(
                    f"Metadata for file '{filename_stem}' (or '{filename}') not found in "
                    f"dataframe '{self._filename_columns[0]}' column."
                )
            row = self._lookup_df.iloc[row_ix]
            meta: dict[str, Any] = dict(row.to_dict())
            meta[self._filename_columns[0]] = filename_stem
            meta["file_path"] = path
            return meta

        match = None
        if self._stem_to_row_ch:
            for key in lookup_keys:
                match = self._stem_to_row_ch.get(key)
                if match is not None:
                    break
        if match is None:
            raise KeyError(
                f"Metadata for file '{filename_stem}' (or '{filename}') not found in "
                f"dataframe (filename_columns={self._filename_columns})."
            )
        row_ix, ch_idx = match
        row = self._df.iloc[row_ix]
        channel_meta: dict[str, Any] = dict(row.to_dict())
        channel_meta["channel_index"] = ch_idx
        channel_meta["file_path"] = path
        return channel_meta

    def metadata_fn(
        self,
        path: str,
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        """Extract metadata with data_dir-aware partial path lookup."""
        meta = self._extract_with_data_dir(path, data_dir)
        if not meta:
            return meta
        paths = meta.get("file_path", path)
        self.ensure_id(meta, paths, data_dir)
        return meta
