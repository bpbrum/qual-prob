
import numpy as np
import pandas as pd

import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from pathlib import Path
from typing import List
import json


class Grapher:

    basic_colors = dict(red="#f72b2b", green="#72cf20")

    bar_layout_light = go.Layout(height=350, width=400, margin=dict(t=10))

    def __init__(
        self,
        parameter: str,
        df: pd.DataFrame,
        limit: float,
        df_obs: List[pd.DataFrame] = None,
        df_det: pd.DataFrame = None,
    ):
        self._parameter = parameter
        self._df = df
        self._limit = limit
        self._iterations = len(df.columns)
        if df_obs is not None:
            self._df_obs = df_obs
        if df_det is not None:
            self._df_det = df_det

        self.scatter_layout_light = go.Layout(  # TODO
            xaxis=dict(title="Length of the river (km)", range=[0, 43.5]),
            yaxis=dict(title="Concentration"),
            hovermode="closest",
        )

    def update_bar_point(self, point: float):
        colors = [self.basic_colors["red"], self.basic_colors["green"]]
        if self._parameter == "do":
            colors = [colors[1], colors[0]]

        # Computation
        over = (self._df.iloc[point] > self._limit).sum() / self._iterations

        x = ["Over " + str(self._limit), "Under " + str(self._limit)]
        y = [over, 1 - over]

        data = [go.Bar(x=x, y=y, marker_color=colors)]

        return go.Figure(data=data, layout=self.bar_layout_light)

    def update_scatter_plot(self, members: int):
        # x-axis values
        df = self._df  # shorthand
        x_val = df.index / 10  # in km

        data = list()
        # Gets n random members
        for i in np.random.randint(1, self._iterations, members):
            # Define the colors
            if self._parameter == "do":
                color = [
                    self.basic_colors["red"]
                    if value < self._limit
                    else self.basic_colors["green"]
                    for value in df[i]
                ]
            else:
                color = [
                    self.basic_colors["green"]
                    if value < self._limit
                    else self.basic_colors["red"]
                    for value in df[i]
                ]

            # Adds the traces
            data.append(
                go.Scatter(
                    x=x_val,
                    y=df[i],
                    mode="markers",
                    name="Run " + str(i),
                    legendgroup="Model Runs",
                    showlegend=False,
                    marker=dict(size=2, color=color),
                )
            )

        # Adds other curves
        # Median
        data.append(
            go.Scatter(
                x=x_val,
                y=df.median(axis=1),
                mode="lines",
                line={"width": 4, "color": "black"},
                name="Median",
            )
        )

        # Other traces
        traces_aux = [
            dict(
                name="Quartiles",
                val=df.quantile(0.75, axis=1),
                line_style=dict(width=3, dash="dash", color="#2f2f2f"),
                showlegend=True,
            ),
            dict(
                name="Quartiles",
                val=df.quantile(0.25, axis=1),
                line_style=dict(width=3, dash="dash", color="#2f2f2f"),
                showlegend=False,
            ),
            dict(
                name="Member extremes",
                val=df.min(axis=1),
                line_style=dict(width=3, dash="dashdot", color="#2f2f2f"),
                showlegend=True,
            ),
            dict(
                name="Member extremes",
                val=df.max(axis=1),
                line_style=dict(width=3, dash="dashdot", color="#2f2f2f"),
                showlegend=False,
            ),
            dict(
                name="Legal Limit",
                val=np.ones(435) * self._limit,
                line_style=dict(width=3, dash="dot", color="#00008b"),
                showlegend=True,
            ),
        ]
        for trace in traces_aux:
            data.append(
                go.Scatter(
                    x=x_val,
                    y=trace["val"],
                    mode="lines",
                    line=trace["line_style"],
                    name=trace["name"],
                    showlegend=trace["showlegend"],
                )
            )

        # Deterministic
        if hasattr(self, "_df_det"):
            data.append(
                go.Scatter(
                    x=self._df_det.x,
                    y=self._df_det[self._parameter],
                    mode="lines",
                    name="DetBG",
                    line=dict(width=4, color="yellow", dash="dashdot"),
                )
            )

        # Observation
        if hasattr(self, "_df_obs"):
            for obs in self._df_obs:
                data.append(
                    go.Scatter(
                        x=obs.x,
                        y=obs[self._parameter],
                        mode="markers",
                        name="Obs",
                        marker=dict(size=6, color="purple"),
                    )
                )

        fig = go.Figure(data=data, layout=self.scatter_layout_light)
        fig.layout.template = "simple_white"
        return fig

    def plot_accordance(self):
        data = list()

        y = (
            self._df > self._limit
            if (self._parameter != "do")
            else self._df < self._limit
        )
        yy = y.sum(axis=1) / 10  # in percentage

        df_det = self._df_det[self._parameter]
        ydet = (
            df_det > self._limit if (self._parameter != "do") else df_det < self._limit
        )
        ydet *= 100  # in %

        x_val = self._df.index / 10  # in km

        # Adds the traces
        data.append(
            go.Scatter(
                x=x_val, y=yy, mode="lines", name="QUAL-PROB", line=dict(color="Orange")
            )
        )

        data.append(
            go.Scatter(
                x=x_val,
                y=ydet,
                mode="lines",
                name="DetBG",
                line=dict(color="Blue", dash="dashdot"),
            )
        )

        layout = go.Layout(
            xaxis=dict(title="Length of the river (km)", range=[0, x_val[-1]]),
            yaxis=dict(title="Probability of non-accordance (%)", range=[0, 100]),
            hovermode="closest",
        )

        fig = go.Figure(data=data, layout=layout)
        fig.layout.template = "simple_white"
        return fig


class GraphHelper(Grapher):
    def __init__(
        self,
        config: json,
        dfs: List[pd.DataFrame],
        df_det: pd.DataFrame = None,
        df_obs: List[pd.DataFrame] = None,
    ):
        self._df_obs = df_obs
        self._df_det = df_det
        self._parameters_df = pd.DataFrame(config)
        self._parameters_df = self._parameters_df.set_index("name", drop=True)
        self._parameters_list = self._parameters_df["val"].to_list()
        self._dfs = dfs
        self._limits = self._parameters_df["limit"]
        self._units = self._parameters_df["units"]

    def plot_all(
        self, num_members: int = 100, file_path: str = ".", file_name: str = "plot.html"
    ):
        def y_axis_layout(unit):
            return dict(title=dict(text=unit, standoff=0.5, font=dict(size=12)))

        titles = self._parameters_df.index
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=(titles),
            shared_xaxes=False,
            vertical_spacing=0.07,
            horizontal_spacing=0.06,
        )

        x_vals = len(self._dfs[0]) / 10  # in km, assume same length for all parameters

        layout = dict(
            width=1200,
            height=1750,
            showlegend=True,
            legend=dict(
                title="Legend", yanchor="bottom", y=0.1, xanchor="left", x=0.52
            ),
        )

        xaxis_layout = dict(
            title=dict(text="Distance [km]", standoff=0.5, font=dict(size=12)),
            dtick=10,
            range=(0, x_vals),
        )

        for i, param in enumerate(self._parameters_list):
            g = Grapher(
                parameter=param,
                df=self._dfs[i],
                limit=self._limits[i],
                df_det=self._df_det[["x", param]],
                df_obs=[obs[["x", param]] for obs in self._df_obs],
            )
            plot_fig = g.update_scatter_plot(members=num_members).data
            plot_indx = (i // 2 + 1, i % 2 + 1)
            for trace in plot_fig:
                if i > 0:
                    trace["showlegend"] = False
                fig.add_trace(trace, plot_indx[0], plot_indx[1])

            axis_num = str(i + 1) if i > 0 else ""
            layout[f"yaxis{axis_num}"] = y_axis_layout(unit=self._units[i])
            layout[f"xaxis{axis_num}"] = xaxis_layout

        layout = go.Layout(**layout)
        fig.layout.template = "simple_white"
        fig.update_layout(layout)
        fig.update_yaxes(rangemode="tozero")

        out_path = Path(file_path)
        out_path.mkdir(exist_ok=True, parents=True)
        out_path = out_path / file_name
        pyo.plot(fig, filename=str(out_path))
