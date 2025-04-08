import os
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import umap
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_bootstrap_templates import load_figure_template
from plotly.subplots import make_subplots
from utils.activations import load_activations


def get_colorscale_options() -> List[Dict[str, str]]:
    return [
        {"label": "Viridis", "value": "Viridis"},
        {"label": "Plasma", "value": "Plasma"},
        {"label": "Inferno", "value": "Inferno"},
        {"label": "Magma", "value": "Magma"},
        {"label": "Cividis", "value": "Cividis"},
        {"label": "Turbo", "value": "Turbo"},
        {"label": "Jet", "value": "Jet"},
    ]


def compute_embedding(
    train_act: np.ndarray,
    test_act: Optional[np.ndarray],
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
):
    """Compute the UMAP embedding for training data and optionally for test data."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        if test_act is not None:
            combined_act = np.vstack([train_act, test_act])
            combined_embedding = reducer.fit_transform(combined_act)
            train_size = train_act.shape[0]
            return combined_embedding[:train_size], combined_embedding[train_size:]
        else:
            return reducer.fit_transform(train_act), None


def create_dash_app(
    activation_dir: str,
    epochs: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
):
    """
    Create and configure a Dash app for visualizing neural network activations with UMAP projections.
    This version computes axis ranges based on the UMAP values so that the plot always centers the data.
    The axes (and the point positions) animate smoothly between states.
    """
    train_dir = os.path.join(activation_dir, "train")
    epoch_files = [
        f
        for f in os.listdir(train_dir)
        if f.startswith("epoch_") and f.endswith(".npz")
    ]
    available_epochs = sorted(
        float(f.replace("epoch_", "").replace(".npz", "").replace("_", "."))
        for f in epoch_files
    )
    filtered_epochs = [e for e in available_epochs if e <= epochs]

    activation_data = load_activations(activation_dir, filtered_epochs)
    train_data = activation_data["train"]
    test_data = activation_data["test"]

    train_labels = train_data["labels"]
    test_labels = test_data["labels"]

    epoch_list = sorted(train_data["activations"].keys())
    first_epoch = epoch_list[0]
    available_layers = [
        layer.replace("act_", "") for layer in train_data["activations"][first_epoch]
    ]

    embeddings = {}
    for layer in available_layers:
        embeddings[layer] = {}
        for epoch in epoch_list:
            layer_key = (
                f"act_{layer}"
                if f"act_{layer}" in train_data["activations"][epoch]
                else layer
            )
            train_act = train_data["activations"][epoch][layer_key]
            test_act = None
            if epoch in test_data["activations"]:
                test_act = test_data["activations"][epoch].get(layer_key)
            embeddings[layer][epoch] = compute_embedding(
                train_act, test_act, n_neighbors, min_dist, metric, random_state
            )

    load_figure_template("darkly")
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    show_every_nth = max(1, len(epoch_list) // 10)
    epoch_marks = {
        i: f"{epoch_list[i]:.2f}" for i in range(0, len(epoch_list), show_every_nth)
    }
    epoch_marks[0] = f"{epoch_list[0]:.2f}"
    epoch_marks[len(epoch_list) - 1] = f"{epoch_list[-1]:.2f}"

    app.layout = dbc.Container(
        fluid=True,
        className="bg-dark text-light p-3",
        children=[
            html.H1(
                "Neural Network Activation UMAP Visualization",
                className="text-center mb-4",
                style={"fontWeight": "bold", "fontSize": "36px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Layer:", className="h5"),
                            dcc.Dropdown(
                                id="layer-dropdown",
                                options=[
                                    {"label": l, "value": l} for l in available_layers
                                ],
                                value=available_layers[0],
                                clearable=False,
                                className="mb-3",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.Label("Color Scale:", className="h5"),
                            dcc.Dropdown(
                                id="colorscale-dropdown",
                                options=get_colorscale_options(),
                                value="Viridis",
                                clearable=False,
                                className="mb-3",
                            ),
                        ],
                        md=3,
                    ),
                ]
            ),
            dbc.Card(
                dbc.CardBody([dcc.Graph(id="umap-graph")]),
                className="border border-secondary mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Epoch Slider:", className="h5"),
                                            dcc.Slider(
                                                id="epoch-slider",
                                                min=0,
                                                max=len(epoch_list) - 1,
                                                step=1,
                                                value=0,
                                                marks=epoch_marks,
                                            ),
                                            html.Div(
                                                id="current-epoch-display",
                                                className="text-center mt-2",
                                                children=f"Current Epoch: {epoch_list[0]:.2f}",
                                            ),
                                        ],
                                        width=7,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Animation Speed:", className="h5"
                                            ),
                                            dcc.Slider(
                                                id="animation-speed-slider",
                                                min=500,
                                                max=2000,
                                                step=100,
                                                value=1000,
                                                marks={
                                                    500: "Fast",
                                                    1000: "Medium",
                                                    2000: "Slow",
                                                },
                                                className="mb-3",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        html.Button(
                                            html.I(className="fas fa-play"),
                                            id="play-button",
                                            className="btn btn-primary mb-2 fa-play",
                                            style={"fontSize": "24px"},
                                        ),
                                        width=2,
                                        className="d-flex justify-content-center align-items-center",
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Instructions:", className="mt-2"),
                                    html.Ul(
                                        [
                                            html.Li(
                                                "Select a layer from the dropdown to view its activation projections."
                                            ),
                                            html.Li(
                                                "Training data points are shown as circles; test data (if available) as triangles."
                                            ),
                                            html.Li(
                                                "Choose different color scales to visualize the data."
                                            ),
                                            html.Li(
                                                "Use the slider to select an epoch or click play to animate through epochs."
                                            ),
                                        ],
                                        style={"fontSize": "16px"},
                                    ),
                                ]
                            ),
                            className="border border-secondary",
                        ),
                    ]
                )
            ),
            html.Div(id="animation-control", style={"display": "none"}),
            dcc.Interval(
                id="interval-component", interval=1000, n_intervals=0, disabled=True
            ),
            dcc.Store(id="epoch-data", data={"epochs": epoch_list}),
        ],
    )

    @app.callback(
        Output("interval-component", "interval"),
        Input("animation-speed-slider", "value"),
    )
    def update_animation_speed(speed_value):
        return speed_value

    @app.callback(
        [Output("interval-component", "disabled"), Output("play-button", "className")],
        [Input("play-button", "n_clicks")],
        [State("interval-component", "disabled")],
    )
    def toggle_animation(n_clicks, currently_disabled):
        if n_clicks is None:
            return True, "btn btn-primary mb-2 fa-play"
        new_state = not currently_disabled
        return new_state, (
            "btn btn-danger mb-2 fa-pause"
            if not new_state
            else "btn btn-primary mb-2 fa-play"
        )

    @app.callback(
        Output("epoch-slider", "value"),
        Input("interval-component", "n_intervals"),
        [
            State("epoch-slider", "value"),
            State("epoch-slider", "max"),
            State("interval-component", "disabled"),
        ],
    )
    def advance_epoch(n_intervals, current_epoch, max_epoch, animation_disabled):
        if animation_disabled:
            return current_epoch
        return (current_epoch + 1) % (max_epoch + 1)

    @app.callback(
        [Output("umap-graph", "figure"), Output("current-epoch-display", "children")],
        [
            Input("layer-dropdown", "value"),
            Input("colorscale-dropdown", "value"),
            Input("epoch-slider", "value"),
        ],
        State("epoch-data", "data"),
    )
    def update_graph(
        selected_layer: str, colorscale: str, epoch_index: int, epoch_data: Dict
    ) -> Any:
        selected_epoch = epoch_data["epochs"][epoch_index]
        train_embedding, test_embedding = embeddings[selected_layer][selected_epoch]

        # Create the subplot figure
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["<b>Training Data</b>", "<b>Test Data</b>"],
            horizontal_spacing=0.1,
        )

        # Trace for training data
        fig.add_trace(
            go.Scatter(
                x=train_embedding[:, 0],
                y=train_embedding[:, 1],
                mode="markers",
                marker=dict(
                    color=train_labels,
                    colorscale=colorscale,
                    showscale=True,
                    symbol="circle",
                    size=8,
                    opacity=0.8,
                    colorbar=dict(title="Class"),
                ),
                name="Training Data",
                hovertemplate="<b>Class:</b> %{marker.color}<br><b>UMAP 1:</b> %{x:.3f}<br><b>UMAP 2:</b> %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Trace for test data (if available)
        if test_embedding is not None:
            fig.add_trace(
                go.Scatter(
                    x=test_embedding[:, 0],
                    y=test_embedding[:, 1],
                    mode="markers",
                    marker=dict(
                        color=test_labels,
                        colorscale=colorscale,
                        showscale=True,
                        symbol="triangle-up",
                        size=8,
                        opacity=0.8,
                        colorbar=dict(title="Class"),
                    ),
                    name="Test Data",
                    hovertemplate="<b>Class:</b> %{marker.color}<br><b>UMAP 1:</b> %{x:.3f}<br><b>UMAP 2:</b> %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

        def compute_range(values, padding_ratio=0.1):
            vmin, vmax = values.min(), values.max()
            padding = (vmax - vmin) * padding_ratio
            return [vmin - padding, vmax + padding]

        train_x_range = compute_range(train_embedding[:, 0])
        train_y_range = compute_range(train_embedding[:, 1])
        fig.update_xaxes(range=train_x_range, title_text="UMAP 1", row=1, col=1)
        fig.update_yaxes(range=train_y_range, title_text="UMAP 2", row=1, col=1)

        if test_embedding is not None:
            test_x_range = compute_range(test_embedding[:, 0])
            test_y_range = compute_range(test_embedding[:, 1])
            fig.update_xaxes(range=test_x_range, title_text="UMAP 1", row=1, col=2)
            fig.update_yaxes(range=test_y_range, title_text="UMAP 2", row=1, col=2)

        # Remove the dynamic uirevision so that the axis changes animate instead of snapping.
        fig.update_layout(
            title=dict(
                text=f"<b>UMAP Projection of {selected_layer} Activations - Epoch {selected_epoch:.2f}</b>",
                font=dict(size=26),
                x=0.5,
            ),
            template="darkly",
            width=1400,
            height=600,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=50, r=50, t=120, b=50),
            uirevision="constant",
            transition=dict(duration=500, easing="cubic-in-out"),
        )

        return fig, f"Current Epoch: {selected_epoch:.2f}"

    return app
