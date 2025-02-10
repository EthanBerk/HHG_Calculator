import dash
from dash import dcc, html, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from components import *
from cards import *

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.CYBORG]
)  


def generate_field_graph():
    time = np.linspace(-150, 150, 300)
    field = np.sin(2 * np.pi * time / 50) * np.exp(-(time**2) / (2 * 50**2))
    envelope = np.exp(-(time**2) / (2 * 50**2))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time, y=field, mode="lines", name="Electric Field", line=dict(color="red")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=envelope,
            mode="lines",
            name="Envelope",
            line=dict(dash="dash", color="blue"),
        )
    )
    fig.update_layout(
        title="Field and Ionization Fraction",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def generate_ionization_graph():
    intensity = np.linspace(3.5, 5.0, 100)
    peak_ionization = 0.1 * (intensity - 3.5) ** 2

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=intensity,
            y=peak_ionization,
            mode="lines",
            name="Ionization",
            line=dict(color="white"),
        )
    )
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
                dbc.Col(
                    [
                       input_card() 
                    ],
                    md=6,
                    sm=12
                ),
                dbc.Col(
                    [
                        computed_values_card()
                    ],
                    md=6,
                    sm=12,
                ),
                
            ],
            align="start",
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="field-ionization-graph",
                        figure=generate_field_graph(),
                        style={"width": "100%", "height": "300px"},
                    ),
                    md=6,
                    sm=12,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="peak-ionization-graph",
                        figure=generate_ionization_graph(),
                        style={"width": "100%", "height": "300px"},
                    ),
                    md=6,
                    sm=12,
                ),
            ],
            className="mb-4",
        ),
        
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
c= 299792458 


@callback(
    Output("target-energy-ev", "value"),
    Output("target-wavelength-um", "value"),
    Input("target-energy-ev", "value"),
    Input("target-wavelength-um", "value")
)
def wavelength_to_energy(E, l):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "target-energy-ev" and E :
        l =( hbar *c)/(float(E) *1e-6) 
    elif l:
        E = (hbar *c)/(float(l) * 1e-6)

    return E, l




@callback(
    [Output("parameter-check", "children")],
    [Input("run-btn", "n_clicks")],
    [
        State("driving-wavelength-um", "value"),
        State("radius-um", "value"),
        State("target-energy-ev", "value"),
        State("pressure-bar", "value"),
        State("pulse_duration-fs", "value"),
        State("gas_type", "value"),
    ],
)
def update_simulation(n, lamL, R, eq, ppm, tau, gas):
    if(n):
        return ["Parameters OK!"]
    return [""]
    


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
