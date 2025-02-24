import dash
from dash import dcc, html, callback, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from components import *
from cards import *
from simulation import run_simulation

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])


def generate_field_graph():
    fig = go.Figure()
    fig.update_layout(
        title="Field and Ionization Fraction",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def generate_ionization_graph():
    fig = go.Figure()
    fig.update_layout(
        title="Peak Ionization vs. Intensity",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Img(src="/assets/JILA_web_white_logo.png", height="50px"),
                    width="auto",
                ),
                dbc.Col(
                    html.H3(
                        "HHG Simulation",
                        className="text-center text-primary mb-3",
                    ),
                    width=True,
                ),
            ],
            align="center",
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col([input_card(), computed_values_card2()], md=6, sm=12),
                dbc.Col(
                    [
                        computed_values_card1(),
                        dcc.Graph(
                            id="peak-ionization-graph",
                            figure=generate_ionization_graph(),
                            style={"width": "100%", "height": "300px"},
                            mathjax=True,
                        ),
                    ],
                    md=6,
                    sm=12,
                ),
            ],
            align="start",
            className="mb-4",
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             dcc.Graph(
        #                 id="field-ionization-graph",
        #                 figure=generate_field_graph(),
        #                 style={"width": "100%", "height": "300px"},
        #                 mathjax=True
        #             ),
        #             md=6,
        #             sm=12,
        #         ),
        #         dbc.Col(
        #             md=6,
        #             sm=12,
        #         ),
        #     ],
        #     className="mb-4",
        # ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Show Documentation",
                            id="toggle-docs",
                            color="info",
                            className="mb-3",
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                [
                                    dbc.CardHeader("How the simulation Works"),
                                    dbc.CardBody(
                                        [
                                            dcc.Markdown(
                                                """
                        ### **HHG Simulation Model**
                        Model description 
                        $$
                        P = \\frac{E}{t} = \\frac{hc}{\\lambda}
                        $$

                        """,
                                                mathjax=True,
                                            )
                                        ]
                                    ),
                                ]
                            ),
                            id="docs-collapse",
                            is_open=False,
                        ),
                    ],
                    md=12,
                )
            ],
            className="mb-4",
        ),
    ],
    fluid=True,
)


hbar = 4.135669e-15
c = 299792458


@callback(
    Output("target-energy-ev", "value"),
    Output("target-wavelength-fm", "value"),
    Input("target-energy-ev", "value"),
    Input("target-wavelength-fm", "value"),
)
def wavelength_to_energy(E, l):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "target-energy-ev":
        if not E:
            l = 0
        else:
            l = round((hbar * c) / (float(E) * 1e-9), 3)
    elif l:
        if not l:
            E = 0
        else:
            E = round((hbar * c) / (float(l) * 1e-9), 3)

    return E, l


@callback(
    [
        Output("parameter-check", "children"),
        Output("energy-cutoff", "children"),
        Output("energy-critical", "children"),
        Output("ionization-critical", "children"),
        Output("power-transmission", "children"),
        Output("pressure-minimum", "children"),
        Output("absorption-length", "children"),
        Output("fiber-length", "children"),
        Output("beam-waist", "children"),
        Output("confocal-parm", "children"),
        Output("peak-ionization-graph", "figure"),
        # Output("field-ionization-graph", "figure"),
    ],
    [Input("run-btn", "n_clicks")],
    [
        State("driving-wavelength-um", "value"),
        State("radius-um", "value"),
        State("target-energy-ev", "value"),
        State("pressure-bar", "value"),
        State("pulse_duration-fs", "value"),
        State("gas_type", "value"),
        State("temp_gas", "value"),
    ],
    prevent_initial_call=True,
    running=[
        (Output("run-btn", "disabled"), True, False),
        (Output("loading-sim", "display"), "show", "hide"),
        ],
)
def update_simulation(c, lamL, R, eq, ppm, tau, gas, temp):
    n = -1.0
    p = 3
    if c:
        results = run_simulation(lamL, R, eq, ppm, tau, gas, temp)
        return [
            [results.get("err", "Parameters OK!")],
            round(results.get("E_cut", n), p),
            round(results.get("E_cr", n), p),
            round(results.get("eta_cr", n), p),
            round(results.get("P_tran", n), p),
            round(results.get("p0", n), p),
            round(results.get("L_abs", n), p),
            round(results.get("L_f", n), p),
            round(results.get("b", n), p),
            round(results.get("w0", n), p),
            results.get("peak-ionization-graph", generate_ionization_graph()),
        ]
    return [""], n, n, n, n, n, n, n, n, n, generate_ionization_graph()


@callback(
    Output("docs-collapse", "is_open"),
    [Input("toggle-docs", "n_clicks")],
    [State("docs-collapse", "is_open")],
)
def toggle_docs(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=True)
