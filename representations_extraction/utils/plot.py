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
    """Return a list of available colorscale options for the dropdown."""
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
    """Create and configure a Dash app for visualizing neural network activations with UMAP projections."""

    # Load activations and labels with the new structure
    activation_data = load_activations(activation_dir, epochs)
    train_data = activation_data["train"]
    test_data = activation_data["test"]

    train_labels = train_data["labels"]
    test_labels = test_data["labels"]

    # Get available layers from first epoch's data
    available_layers = list(train_data["activations"][0].keys())

    # Pre-compute embeddings for each layer and epoch
    embeddings = {}
    for layer in available_layers:
        embeddings[layer] = {}
        for epoch in range(epochs):
            train_act = train_data["activations"][epoch][layer]
            test_act = test_data["activations"][epoch].get(layer, None)

            embeddings[layer][epoch] = compute_embedding(
                train_act, test_act, n_neighbors, min_dist, metric, random_state
            )

    # Load the "Darkly" figure template for Plotly figures
    load_figure_template("darkly")

    # Initialize the Dash app with the "Darkly" Bootstrap theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

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
                                    {"label": layer, "value": layer}
                                    for layer in available_layers
                                ],
                                value=available_layers[0],
                                clearable=False,
                                className="mb-3",
                            ),
                        ],
                        width=16,
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
                        width=8,
                        md=3,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        dcc.Graph(id="umap-graph"),
                    ]
                ),
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
                                                max=epochs - 1,
                                                step=1,
                                                value=0,
                                                marks={
                                                    i: str(i) for i in range(epochs)
                                                },
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
                                        width=12,
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
                                ],
                            ),
                        ],
                        width=12,
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
                                                "Use the slider below the plot to select an epoch or click the play button to animate through epochs."
                                            ),
                                        ],
                                        style={"fontSize": "16px"},
                                    ),
                                ]
                            ),
                            className="border border-secondary",
                        ),
                    ],
                    width=12,
                ),
                className="mt-4",
            ),
            # Hidden div for animation control
            html.Div(id="animation-control", style={"display": "none"}),
            # Interval component for the animation
            dcc.Interval(
                id="interval-component",
                interval=1000,  # Default to 1 second, will be updated by the slider
                n_intervals=0,
                disabled=True,
            ),
        ],
    )

    @app.callback(
        Output("interval-component", "interval"),
        Input("animation-speed-slider", "value"),
    )
    def update_animation_speed(speed_value):
        """Update the animation interval based on the speed slider."""
        return speed_value

    @app.callback(
        [Output("interval-component", "disabled"), Output("play-button", "className")],
        [Input("play-button", "n_clicks")],
        [State("interval-component", "disabled")],
    )
    def toggle_animation(n_clicks, currently_disabled):
        """Toggle the animation on/off when the play button is clicked."""
        if n_clicks is None:
            return True, "btn btn-primary mb-2 fa-play"

        new_disabled_state = not currently_disabled
        button_class = (
            "btn btn-danger mb-2 fa-pause"
            if new_disabled_state
            else "btn btn-primary mb-2 fa-play"
        )
        return new_disabled_state, button_class

    @app.callback(
        Output("epoch-slider", "value"),
        [Input("interval-component", "n_intervals")],
        [
            State("epoch-slider", "value"),
            State("epoch-slider", "max"),
            State("interval-component", "disabled"),
        ],
    )
    def advance_epoch_slider(n_intervals, current_epoch, max_epoch, animation_disabled):
        """Advance the epoch slider when the animation interval triggers."""
        if animation_disabled:
            return current_epoch

        next_epoch = (current_epoch + 1) % (max_epoch + 1)
        return next_epoch

    @app.callback(
        Output("umap-graph", "figure"),
        [
            Input("layer-dropdown", "value"),
            Input("colorscale-dropdown", "value"),
            Input("epoch-slider", "value"),
        ],
    )
    def update_graph(selected_layer: str, colorscale: str, selected_epoch: int) -> Any:
        """
        Update the UMAP graph when the selected layer or colorscale changes or when the epoch slider is adjusted.
        """
        # Get the pre-computed embeddings for this layer and epoch
        train_embedding, test_embedding = embeddings[selected_layer][selected_epoch]

        # Create the figure with subplots for training and test data
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["<b>Training Data</b>", "<b>Test Data</b>"],
            horizontal_spacing=0.1,
        )

        # Trace for training data
        train_trace = go.Scatter(
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
        )
        fig.add_trace(train_trace, row=1, col=1)

        # Trace for test data
        if test_embedding is not None:
            test_trace = go.Scatter(
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
            )
            fig.add_trace(test_trace, row=1, col=2)

        # Update axes labels for both subplots
        fig.update_xaxes(title_text="UMAP 1", row=1, col=1)
        fig.update_yaxes(title_text="UMAP 2", row=1, col=1)
        fig.update_xaxes(title_text="UMAP 1", row=1, col=2)
        fig.update_yaxes(title_text="UMAP 2", row=1, col=2)

        # Update the layout with the "darkly" theme and additional styling
        fig.update_layout(
            title=dict(
                text=f"<b>UMAP Projection of {selected_layer} Activations - Epoch {selected_epoch}</b>",
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
        )
        return fig

    return app
