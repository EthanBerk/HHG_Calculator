from dash import dcc


def generate_docs():
    return dcc.Markdown(
        """
                        ## **HHG Simulation Model**
                        #### Single atom cutoff energy calculation
                        $$
                        E_\\text{cut}[ev] = 1.22 I_p[ev] + 3.17Up[ev]
                        $$
                        Were $I_p$ is the and $U_p$ 
                        $$
                        U_p = A U \\lambda^2
                        $$
                        
                        #### Fundamental mode of the (Bessel function)

                        
                        $$
                        J_0(z) = \\left(\\frac{z}{2}\\right) \\sum_{k=0}^{\\infty}\\frac{\\left(\\frac{-z^2}{4}\\right)^k}{k!\\Gamma(k+1)}
                        $$
                        
                        #### Phase Matching
                        We define the phase $$
                        
                        $$
                        \\Delta k = \\frac{2 kh}{\\lambda} (n(\\lambda_g) - n(\\lambda))
                        $$
                        
                        ##### Refractive index of infrared light in the capillary
                         The equation and Sellmeier coefficients used follow Borzsonyi et al.
                         "Dispersion measurement of inert gases and gas mixtures at 800 nm"(2008).
                        $$
                            n_0 = 1 + \\frac{1}{2} \\left(\\frac{T_0}{T}\\right) \\left( B_1 \\frac{\\lambda^2}{\\lambda^2 - C_1} + B_2 \\frac{\\lambda^2}{\\lambda^2 - C_2} \\right)
                        $$
                        Where, $[B_1, B_2, C_1, C_2]$ are Sellmeier coefficients of the given gas; T_0 is a reference temp of 293; T is the temp of the gas (set to 293 for simplicity).
                        
                        ##### Refractive index of generated harmonics(UV/Xrays) in the capillary
                    


                        """,
        mathjax=True,
    )
