from dash import dcc, html
import dash_bootstrap_components as dbc


def create_output_row(label: str, unit: str, description: str, output_id: str):
    return dbc.ListGroupItem(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dcc.Markdown(f""" ${label}[${unit}$]$  """, mathjax=True),
                                    html.P(
                                        "-", className="text-warning border text-center rounded-3", id=output_id
                                    ),
                                ],
                                
                            ),
                        ],
                        width=5,
                    ),
                    dbc.Col(
                        [
                            html.Small(description, className="my-0"),
                        ],
                        width=7,
                    ),
                ]
            )
        ]
    )


def create_input_row(label: str, unit: str, description: str, input_id: str, default = 0):
    return dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Markdown(f""" ${label}$ """, mathjax=True),
                        width=1,
                    ),
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        id=input_id,
                                        type="number",
                                        value=default
                                    ),
                                    dbc.InputGroupText(unit, className="bg-secondary"),
                                ],
                                className="my-0",
                            ),
                            dbc.FormText(description, className="ms-1"),
                        ],
                        width=11,
                    ),
                ]
            )
        ],
        width=6,
    )
