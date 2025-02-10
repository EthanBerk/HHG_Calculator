from dash import dcc, html
import dash_bootstrap_components as dbc
from components import *


def input_card():
    return dbc.Card(
        [
            dbc.CardHeader("Simulation Parameters"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            create_input_row(
                                "\\lambda_L",
                                "μm",
                                "Wavelength of driving laser",
                                "driving-wavelength-um",
                            ),
                            create_input_row(
                                "R",
                                "μm",
                                "Radius of hollow capillary",
                                "radius-um",
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            create_input_row(
                                "\\lambda_q",
                                "μm",
                                "Target harmonic wavelength",
                                "target-wavelength-um",
                            ),
                            create_input_row(
                                "E_q",
                                "eV",
                                "Target harmonic energy",
                                "target-energy-ev",
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            create_input_row(
                                "p",
                                "Bar",
                                "Pressure of capillary",
                                "pressure-bar",
                            ),
                            create_input_row(
                                "T_c",
                                "fs",
                                "Pulse duration",
                                "pulse_duration-fs",
                            ),
                        ]
                    ),
                    html.Label("Gas Type"),
                    dbc.RadioItems(
                        id="gas_type",
                        options=[
                            {"label": i, "value": i} for i in ["He", "Ne", "N2", "Ar"]
                        ],
                        value="He",
                        inline=True,
                        className="mb-3",
                    ),
                    dbc.Button(
                        "RUN",
                        id="run-btn",
                        color="success",
                        className="w-100 mt-2",
                    ),
                    html.Div(
                        id="parameter-check",
                        className="text-danger mt-2 text-center",
                    ),
                ]
            ),
        ], class_name="my-3"
    )


def computed_values_card():
    return dbc.Card(
        [
            dbc.CardHeader("Computed Values"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.ListGroup(
                                        [
                                            create_output_row(
                                                "E_\\text{cut}",
                                                "ev",
                                                "Intensity/pulse energy that reaches cutoff for the desired harmonic",
                                                "cutoff-energy",
                                            ),
                                            create_output_row(
                                                "E_\\text{min}",
                                                "mJ",
                                                "Intensity that reaches critical ionization at the pulse peak",
                                                "min-energy",
                                            ),
                                            create_output_row(
                                                "E",
                                                "mJ",
                                                "Approximate optimal energy",
                                                "optimal-energy",
                                            ),
                                        ]
                                    )
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.ListGroup(
                                        [
                                            create_output_row(
                                                "L_\\text{abs}",
                                                "ev",
                                                "Absorption length",
                                                "absorption-length",
                                            ),
                                            create_output_row(
                                                "L_\\text{f}",
                                                "mm",
                                                "fiber length",
                                                "fiber-length",
                                            ),
                                            create_output_row(
                                                "P",
                                                "W",
                                                "Power transmission of fundamental mode",
                                                "power-transmission",
                                            ),
                                        ]
                                    )
                                ],
                                width=6,
                            ),
                        ]
                    )
                ]
            ),
        ], class_name="my-3"
    )
