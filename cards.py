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
                                1.5,
                            ),
                            create_input_row(
                                "R",
                                "μm",
                                "Radius of hollow capillary",
                                "radius-um",
                                100,
                            ),
                        ],
                    ),
                    dbc.Row(
                        [
                            create_input_row(
                                "\\lambda_q",
                                "fm",
                                "Target harmonic wavelength",
                                "target-wavelength-fm",
                                4.132,
                            ),
                            create_input_row(
                                "E_q",
                                "eV",
                                "Target harmonic energy",
                                "target-energy-ev",
                                300,
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            create_input_row(
                                "p", "Bar", "Pressure of capillary", "pressure-bar", 10
                            ),
                            create_input_row(
                                "\\tau", "cycles", "Pulse duration", "pulse-duration", 10
                            ),
                            
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Gas Type"),
                                    dbc.RadioItems(
                                        id="gas_type",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in ["He", "Ne", "N2", "Ar"]
                                        ],
                                        value="He",
                                        inline=True,
                                        className="mb-3",
                                    ),
                                ]
                            ),
                            create_input_row(
                                "T_c", "fs", "length of optical cycle", "optical-cycle", 5.003
                            ),
                        ]
                    ),
                    dcc.Loading(
                        [
                            dbc.Button(
                                "RUN",
                                id="run-btn",
                                color="success",
                                className="w-100 mt-2",
                            ),
                        ],
                        id="loading-sim",
                        overlay_style={"visibility":"visible"},
                        type="circle",
                    ),
                    html.Div(
                        id="parameter-check",
                        className="text-warning  mt-2 text-center",
                    ),
                ]
            ),
        ],
        class_name="my-3",
    )


def computed_values_card1():
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
                                                "mJ",
                                                "Intensity/pulse energy that reaches cutoff for the desired harmonic",
                                                "energy-cutoff",
                                            ),
                                            create_output_row(
                                                "E_\\text{cr}",
                                                "mJ",
                                                "Intensity/pulse energy that reaches critical ionization at the pulse peak",
                                                "energy-critical",
                                            ),
                                            create_output_row(
                                                "\\eta_\\text{cr}",
                                                "%",
                                                "Critical ionization",
                                                "ionization-critical",
                                            ),
                                        ]
                                    )
                                ],
                                md=6,
                                sm=12,
                            ),
                            dbc.Col(
                                [
                                    dbc.ListGroup(
                                        [
                                            create_output_row(
                                                "P",
                                                "%",
                                                "Power transmission of fundamental mode",
                                                "power-transmission",
                                            ),
                                            create_output_row(
                                                "p_0",
                                                "bar",
                                                "minimum pressure required to phase match",
                                                "pressure-minimum",
                                            ),
                                            create_output_row(
                                                "p_0",
                                                "torr",
                                                "minimum pressure in torr",
                                                "pressure-minimum-torr",
                                            ),
                                        ]
                                    )
                                ],
                                md=6,
                                sm=12,
                            ),
                        ]
                    )
                ]
            ),
        ],
        class_name="my-3",
    )


def computed_values_card2():
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
                                                "L_\\text{abs}",
                                                "mm",
                                                "Absorption depth, or the length at which the harmonic intensity decays by 1/e",
                                                "absorption-length",
                                            ),
                                            create_output_row(
                                                "L_\\text{f}",
                                                "mm",
                                                "Absorption limited length of the medium. Defined as 6 times the Absorption depth",
                                                "fiber-length",
                                            ),
                                        ]
                                    )
                                ],
                                md=6,
                                sm=12,
                            ),
                            dbc.Col(
                                [
                                    dbc.ListGroup(
                                        [
                                            create_output_row(
                                                "w_0",
                                                "μm",
                                                "Gaussian beam (beam focused into capillary) waist  that optimizes coupling into the fundamental mode ",
                                                "beam-waist",
                                            ),
                                            create_output_row(
                                                "b",
                                                "mm",
                                                "confocal parameter of the optimized gaussian beam waist",
                                                "confocal-parm",
                                            ),
                                        ]
                                    )
                                ],
                                md=6,
                                sm=12,
                            ),
                        ]
                    )
                ]
            ),
        ],
        class_name="my-3",
    )
