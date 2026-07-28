"""Dimensionality reduction and centroid plot methods for PhenoMeVisualization."""

from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..._logging import get_logger
from ...core import get_all_metadata_keys, run_dimensionality_reduction
from ...utils.display_names import capitalize_preserve, format_dr_method
from ._helpers import (
    build_hover_columns as _build_hover_columns,
)
from ._helpers import (
    get_color_column as _get_color_column_fn,
)
from ._helpers import (
    get_marker_styling as _get_marker_styling,
)

logger = get_logger(__name__)


class _DRPlotsMixin:
    """Mixin providing dimensionality reduction and centroid plots."""

    results: object
    device: Any

    def _get_color_column(self, color_by: str, df: pd.DataFrame) -> tuple[str, bool]:
        """Determine the column to use for coloring and whether it's continuous."""
        return _get_color_column_fn(self, color_by, df)

    def _plot_dr_scatter(
        self,
        method: Literal["pca", "tsne", "umap"],
        n_components: int = 2,
        color_by: str | None = None,
        figsize: tuple[int, int] = (10, 8),
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        filters: dict[str, Any | list[Any]] | None = None,
        exclude: dict[str, Any | list[Any]] | None = None,
        hover_features: list[str] | None = None,
        return_fig: bool = False,
        render_mode: Literal["auto", "svg", "webgl"] = "webgl",
        normalize: bool = True,
        sample_size: int | None = None,
        **dr_kwargs: Any,
    ) -> Any:
        """Shared logic for PCA, t-SNE, UMAP scatter plots."""
        df, data_type, dr_obj = run_dimensionality_reduction(
            self,
            method=method,
            n_components=n_components,
            source=source,
            property_keys=property_keys,
            filters=filters,
            exclude=exclude,
            normalize=normalize,
            device=self.device,
            use_gpu=getattr(self, "use_gpu_for_dr", True),
            **dr_kwargs,
        )
        if df is None:
            return

        # Optimization for large datasets: sample points if requested
        if sample_size is not None and len(df) > sample_size:
            seed = getattr(self, "seed", None)
            df = df.sample(n=sample_size, random_state=seed).sort_index()

        if method == "pca" and dr_obj is not None:
            x_col, y_col = "PC1", "PC2"
            z_col = "PC3" if n_components == 3 else None
            if (
                hasattr(dr_obj, "explained_variance_ratio_")
                and len(dr_obj.explained_variance_ratio_) >= n_components
            ):
                axis_labels = {
                    "x": f"PC1 ({dr_obj.explained_variance_ratio_[0]:.1%} variance)",
                    "y": f"PC2 ({dr_obj.explained_variance_ratio_[1]:.1%} variance)",
                }
                if n_components == 3:
                    axis_labels["z"] = f"PC3 ({dr_obj.explained_variance_ratio_[2]:.1%} variance)"
            else:
                axis_labels = {"x": "PC1", "y": "PC2"}
                if n_components == 3:
                    axis_labels["z"] = "PC3"
            rename = {f"Component {i}": f"PC{i}" for i in range(1, n_components + 1)}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        else:
            x_col, y_col = "Component 1", "Component 2"
            z_col = "Component 3" if n_components == 3 else None
            dr_display = format_dr_method(method)
            axis_labels = {
                "x": f"{dr_display} 1",
                "y": f"{dr_display} 2",
            }
            if n_components == 3:
                axis_labels["z"] = f"{dr_display} 3"
        title = f"{format_dr_method(method)} of {data_type}" + (
            f" (colored by {color_by})" if color_by else ""
        )
        return self._plot_embedding_scatter(
            df=df,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            color_by=color_by,
            title=title,
            figsize=figsize,
            axis_labels=axis_labels,
            hover_features=hover_features,
            return_fig=return_fig,
            render_mode=render_mode,
        )

    def plot_pca(
        self,
        n_components: int = 2,
        color_by: str | None = None,
        figsize: tuple[int, int] = (10, 8),
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        filters: dict[str, Any | list[Any]] | None = None,
        exclude: dict[str, Any | list[Any]] | None = None,
        hover_features: list[str] | None = None,
        return_fig: bool = False,
        render_mode: Literal["auto", "svg", "webgl"] = "webgl",
        normalize: bool = True,
        sample_size: int | None = None,
        **dr_kwargs: Any,
    ) -> Any:
        """Plot PCA of embeddings, properties, or combined features (Plotly, WebGL by default)."""
        return self._plot_dr_scatter(
            "pca",
            n_components=n_components,
            color_by=color_by,
            figsize=figsize,
            source=source,
            property_keys=property_keys,
            filters=filters,
            exclude=exclude,
            hover_features=hover_features,
            return_fig=return_fig,
            render_mode=render_mode,
            normalize=normalize,
            sample_size=sample_size,
            **dr_kwargs,
        )

    def plot_tsne(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        color_by: str | None = None,
        figsize: tuple[int, int] = (10, 8),
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        filters: dict[str, Any | list[Any]] | None = None,
        exclude: dict[str, Any | list[Any]] | None = None,
        hover_features: list[str] | None = None,
        return_fig: bool = False,
        render_mode: Literal["auto", "svg", "webgl"] = "webgl",
        normalize: bool = True,
        sample_size: int | None = None,
        **dr_kwargs: Any,
    ) -> Any:
        """Plot t-SNE of embeddings, properties, or combined features (Plotly, WebGL by default)."""
        return self._plot_dr_scatter(
            "tsne",
            n_components=n_components,
            color_by=color_by,
            figsize=figsize,
            source=source,
            property_keys=property_keys,
            filters=filters,
            exclude=exclude,
            hover_features=hover_features,
            return_fig=return_fig,
            render_mode=render_mode,
            normalize=normalize,
            perplexity=perplexity,
            sample_size=sample_size,
            **dr_kwargs,
        )

    def plot_umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        color_by: str | None = None,
        figsize: tuple[int, int] = (10, 8),
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        filters: dict[str, Any | list[Any]] | None = None,
        exclude: dict[str, Any | list[Any]] | None = None,
        hover_features: list[str] | None = None,
        return_fig: bool = False,
        render_mode: Literal["auto", "svg", "webgl"] = "webgl",
        normalize: bool = True,
        sample_size: int | None = None,
        **dr_kwargs: Any,
    ) -> Any:
        """Plot UMAP of embeddings, properties, or combined features (Plotly, WebGL by default)."""
        return self._plot_dr_scatter(
            "umap",
            n_components=n_components,
            color_by=color_by,
            figsize=figsize,
            source=source,
            property_keys=property_keys,
            filters=filters,
            exclude=exclude,
            hover_features=hover_features,
            return_fig=return_fig,
            render_mode=render_mode,
            normalize=normalize,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            sample_size=sample_size,
            **dr_kwargs,
        )

    def _plot_embedding_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str | None,
        color_by: str | None,
        title: str,
        figsize: tuple[int, int],
        axis_labels: dict | None = None,
        hover_features: list[str] | None = None,
        return_fig: bool = False,
        render_mode: Literal["auto", "svg", "webgl"] = "webgl",
    ) -> Any:
        """Create and display a 2D or 3D scatter plot using Plotly."""
        n_points = len(df)
        color_column, is_continuous = (
            _get_color_column_fn(self, color_by, df) if color_by else (None, False)
        )
        hover_cols = _build_hover_columns(
            df, x_col, y_col, z_col, color_column, hover_features, n_points
        )
        index_only_hover = hover_cols == ["Index"] and "Index" in df.columns

        kwargs = {
            "data_frame": df,
            "x": x_col,
            "y": y_col,
            "title": title,
            "width": figsize[0] * 100,
            "height": figsize[1] * 100,
        }
        if index_only_hover:
            kwargs["custom_data"] = ["Index"]
        else:
            kwargs["hover_data"] = hover_cols
            # Display labels: first-letter capitalize without mangling channel names.
            label_cols = set(hover_cols)
            if color_column:
                label_cols.add(color_column)
            kwargs["labels"] = {c: capitalize_preserve(str(c)) for c in label_cols}
        if not z_col:
            kwargs["render_mode"] = render_mode
        if z_col:
            kwargs["z"] = z_col
        if color_column:
            kwargs["color"] = color_column
            if not is_continuous:
                kwargs["category_orders"] = {
                    color_column: sorted(df[color_column].dropna().unique())
                }

        fig = px.scatter_3d(**kwargs) if z_col else px.scatter(**kwargs)
        marker_size, opacity = _get_marker_styling(n_points, z_col is not None)
        fig.update_traces(marker={"size": marker_size, "opacity": opacity})
        if index_only_hover:
            fig.update_traces(
                hovertemplate="<b>Index</b>: %{customdata[0]}<extra></extra>",
            )
        if not color_column:
            # Plotly default first discrete color (matches plotly.express defaults)
            fig.update_traces(marker_color="#636EFA")
        if axis_labels:
            titles = {f"{k}axis_title": v for k, v in axis_labels.items()}
            if z_col:
                fig.update_layout(scene=titles)
            else:
                fig.update_layout(**titles)
        if return_fig:
            return fig
        fig.show()

    def _compute_group_centroids_in_dr_space(
        self,
        group_by: str | list[str],
        method: Literal["pca", "tsne", "umap"],
        n_components: int,
        filters: dict[str, Any | list[Any]] | None,
        exclude: dict[str, Any | list[Any]] | None = None,
        source: str = "embeddings",
        property_keys: list[str] | None = None,
        normalize: bool = True,
        **dr_kwargs: Any,
    ) -> tuple[
        pd.DataFrame | None,
        pd.DataFrame | None,
        list[str],
        dict[str, str] | None,
        bool,
        str,
    ]:
        """Compute one centroid per group in reduced space."""
        dr_method_kwargs = {
            "pca": {},
            "tsne": {"perplexity": 30.0},
            "umap": {"n_neighbors": 15, "min_dist": 0.1},
        }
        kwargs = {**dr_method_kwargs.get(method, {}), **dr_kwargs}

        if method == "pca":
            df, _, dr_obj = run_dimensionality_reduction(
                self,
                method="pca",
                n_components=n_components,
                source=source,
                property_keys=property_keys,
                filters=filters,
                exclude=exclude,
                normalize=normalize,
                device=self.device,
                use_gpu=getattr(self, "use_gpu_for_dr", True),
            )
        elif method == "tsne":
            df, _, _ = run_dimensionality_reduction(
                self,
                method="tsne",
                n_components=n_components,
                source=source,
                property_keys=property_keys,
                filters=filters,
                exclude=exclude,
                normalize=normalize,
                device=self.device,
                use_gpu=getattr(self, "use_gpu_for_dr", True),
                **{k: v for k, v in kwargs.items() if k not in ("n_neighbors", "min_dist")},
            )
            dr_obj = None
        else:
            df, _, _ = run_dimensionality_reduction(
                self,
                method="umap",
                n_components=n_components,
                source=source,
                property_keys=property_keys,
                filters=filters,
                exclude=exclude,
                normalize=normalize,
                device=self.device,
                use_gpu=getattr(self, "use_gpu_for_dr", True),
                **kwargs,
            )
            dr_obj = None

        if df is None or len(df) == 0:
            return None, None, [], None, False, ""
        df_work: pd.DataFrame = df  # Narrow type for closure
        df = df_work  # Working copy for modifications

        group_keys = [group_by] if isinstance(group_by, str) else list(group_by)

        def _resolve_cols(keys: list[str]) -> tuple[str, list[str]]:
            cols = []
            for k in keys:
                c = next((col for col in df_work.columns if col.lower() == str(k).lower()), None)
                if c is None:
                    return "", []
                cols.append(c)
            if len(cols) == 1:
                return cols[0], cols
            return "_group", cols

        group_col, group_cols = _resolve_cols(group_keys)

        if not group_col:
            return None, None, [], None, False, ""

        if len(group_cols) > 1:
            df = df.copy()
            df[group_col] = df[group_cols].astype(str).agg(" | ".join, axis=1)
        comp_cols = [c for c in df.columns if c.startswith("Component ")]
        if comp_cols:
            coord_cols = sorted(comp_cols, key=lambda x: int(x.split()[1]))[:n_components]
        else:
            exclude_cols = {"Index", "Image", group_col}
            coord_cols = [c for c in df.columns if c not in exclude_cols][:n_components]

        axis_labels: dict[str, str] = {}
        if dr_obj is not None and hasattr(dr_obj, "explained_variance_ratio_") and method == "pca":
            assert dr_obj is not None  # Narrow type for mypy
            evr = dr_obj.explained_variance_ratio_
            axis_labels = {
                "x": f"Component 1 ({evr[0]:.1%} variance)",
                "y": f"Component 2 ({evr[1]:.1%} variance)",
            }
            if n_components >= 3 and len(evr) >= 3:
                axis_labels["z"] = f"Component 3 ({evr[2]:.1%} variance)"
        else:
            axis_labels = {"x": coord_cols[0], "y": coord_cols[1]}
            if n_components >= 3 and len(coord_cols) >= 3:
                axis_labels["z"] = coord_cols[2]

        agg_dict = dict.fromkeys(coord_cols, "mean")
        agg_dict["Index"] = "count"
        cent = df.groupby(group_col, dropna=False).agg(agg_dict).reset_index()
        cent = cent.rename(columns={"Index": "n_samples"})
        is_3d = n_components >= 3
        return (
            df,
            cent,
            coord_cols,
            axis_labels or {},
            is_3d,
            group_col,
        )

    def plot_centroids(
        self,
        group_by: str | list[str],
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        filters: dict[str, Any | list[Any]] | None = None,
        exclude: dict[str, Any | list[Any]] | None = None,
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        show_points: bool = True,
        show_centroids: bool = True,
        figsize: tuple[int, int] = (10, 8),
        return_fig: bool = False,
        normalize: bool = True,
        **dr_kwargs: Any,
    ) -> Any:
        """Plot centroids of groups in reduced embedding space."""
        available_keys = get_all_metadata_keys(self.results)
        group_keys = [group_by] if isinstance(group_by, str) else list(group_by)

        def _resolve_keys(keys: list[str]) -> list[str]:
            resolved = []
            for k in keys:
                meta_key = next((ak for ak in available_keys if ak.lower() == str(k).lower()), None)
                if meta_key is None:
                    raise ValueError(
                        f"'{k}' not in metadata. Available: {', '.join(available_keys)}"
                    )
                resolved.append(meta_key)
            return resolved

        group_meta_keys = _resolve_keys(group_keys)

        result = self._compute_group_centroids_in_dr_space(
            group_by=group_meta_keys,
            method=method,
            n_components=n_components,
            filters=filters,
            exclude=exclude,
            source=source,
            property_keys=property_keys,
            normalize=normalize,
            **dr_kwargs,
        )
        if result[0] is None or result[1] is None:
            return None

        (
            df_full,
            df_centroids,
            coord_cols,
            axis_labels,
            is_3d,
            group_col,
        ) = result

        is_3d = is_3d and len(coord_cols) >= 3
        x_col, y_col = coord_cols[0], coord_cols[1]
        z_col = coord_cols[2] if is_3d else None
        assert df_full is not None and df_centroids is not None  # Narrow for mypy

        group_display = (
            " | ".join(group_meta_keys) if len(group_meta_keys) > 1 else group_meta_keys[0]
        )
        title = f"Centroids by {group_display}"

        unique_groups = sorted(df_centroids[group_col].dropna().unique(), key=str)
        if not unique_groups:
            logger.warning("No groups found for centroid plot.")
            return None
        _plotly_default = px.colors.qualitative.Plotly
        color_hex = {
            g: _plotly_default[i % len(_plotly_default)] for i, g in enumerate(unique_groups)
        }

        fig = go.Figure()
        if show_points and df_full is not None and len(df_full) > 0:
            for grp in unique_groups:
                mask = df_full[group_col] == grp
                sub = df_full[mask]
                c = color_hex.get(grp, _plotly_default[0])
                show_leg = not show_centroids
                if is_3d:
                    fig.add_trace(
                        go.Scatter3d(
                            x=sub[x_col],
                            y=sub[y_col],
                            z=sub[z_col],
                            mode="markers",
                            name=str(grp),
                            marker={"size": 4, "opacity": 0.4, "color": c},
                            legendgroup=str(grp),
                            showlegend=show_leg,
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=sub[x_col],
                            y=sub[y_col],
                            mode="markers",
                            name=str(grp),
                            marker={"size": 4, "opacity": 0.4, "color": c},
                            legendgroup=str(grp),
                            showlegend=show_leg,
                        )
                    )

        for grp in unique_groups:
            sub = df_centroids[df_centroids[group_col] == grp]
            if not show_centroids or len(sub) == 0:
                continue
            if len(sub) > 1:
                logger.warning(
                    "Multiple centroid rows for group %r; expected one per group. "
                    "Using the first row only.",
                    grp,
                )
            one = sub.iloc[:1]
            c = color_hex.get(grp, _plotly_default[0])
            label = str(grp)
            if is_3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=one[x_col],
                        y=one[y_col],
                        z=one[z_col],
                        mode="markers+text",
                        name=str(grp),
                        marker={"size": 12, "color": c, "symbol": "diamond"},
                        text=[label],
                        textposition="top center",
                        textfont={"size": 12},
                        legendgroup=str(grp),
                        showlegend=True,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=one[x_col],
                        y=one[y_col],
                        mode="markers+text",
                        name=str(grp),
                        marker={"size": 12, "color": c, "symbol": "diamond"},
                        text=[label],
                        textposition="top center",
                        textfont={"size": 12},
                        legendgroup=str(grp),
                        showlegend=True,
                    )
                )

        fig.update_layout(
            title=title,
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            legend_title_text=f"{group_display}",
        )
        if axis_labels:
            if is_3d:
                fig.update_layout(
                    scene={
                        "xaxis_title": axis_labels.get("x", x_col),
                        "yaxis_title": axis_labels.get("y", y_col),
                        "zaxis_title": axis_labels.get("z", z_col or ""),
                    }
                )
            else:
                fig.update_layout(
                    xaxis_title=axis_labels.get("x", x_col),
                    yaxis_title=axis_labels.get("y", y_col),
                )
        if return_fig:
            return fig
        fig.show()
        return None
