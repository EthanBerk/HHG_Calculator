import numpy as np
import scipy as sp
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator, interp1d
import pandas as pd
import plotly.graph_objects as go
from math import factorial, gamma

plot_template = "plotly_dark";


def run_simulation(lamL, R, eq, ppm, tau, gas, T):
    results = {}
    # inputs
    lamL = lamL * 1e-6  # laser wavelength [m]
    R = R * 1e-6  # fiber radius [m]

    # Temperatures:
    T0 = 273  # standard temperature [K]

    # Driving laser:
    c = 2.99792e8  # speed of light [m/s]
    TC = lamL / c  # length of optical cycle [s]
    pulseType = "sech"  # pulse profile ("sech" or "gauss")
    hc = 1240e-9  # hbar*c [eV m]
    e0 = hc / lamL  # energy/photon of driving laser [eV]
    kL = 2 * np.pi / lamL  # wavenumber of driver [1/m]
    wL = 2 * np.pi * c / lamL  # center angular frequency [rad/s]
    eps0 = 8.85e-12  # vacuum permittivity [F/m]

    #  Initialize vectors:
    lamH = hc / eq  # target wavelength [nm]
    q = round(eq / e0)  # target harmonic order (integer)
    eq = q * e0  # actual target harmonic [eV]

    # Calculate number density at 1 bar and use to calculate the
    # refractive index of the gas for the desired harmonic:

    Nbar = pressure_to_density(1, T)
    # SiO = RefractiveIndexMaterial(shelf='main', book='SiO', page='Hass')
    # SiO.get_refractive_index()
    nH, _ = calculate_form_factors(gas, Nbar, 1, lamH)

    # Temporal profile:
    tau = tau * TC  # pulse FWHM [s]
    LT = 5 * tau  # temporal window size [s]
    NC = round(LT / TC)  # num of cycles in window
    NT = 100 * NC + 1  # num of samples (at least 10/cycle)
  

    # Spatial initialization:
    # r = np.linspace(0, R, 1000)
    um = pd.read_csv("./data/um.csv", header=None)
    um = um.iloc[0, 0]
    # Fm = sp.special.jv(0, um*r/R)

    # Calculate form factors for the harmonic field:
    qe = 1.602e-19  # charge of electron [C]
    re = 2.8179e-15  # classical electron radius [m]

    # Load gas-dependent constants:
    Ip, n20, B1, B2, C1, C2 = gas_constants(gas)

    # Use Sellmeier formula to calculate index of refraction:
    sellmeier = B1 * lamL**2 / (lamL**2 - C1) + B2 * lamL**2 / (lamL**2 - C2)
    n0 = 1 + (1 / 2) * (T0 / T) * sellmeier

    # Waveguide contribution to refractive index:
    delta_wg = um**2 * lamL**2 / (8 * np.pi**2 * R**2)

    # Calculate minimum pressure required to phase match:
    p0 = delta_wg / (n0 - 1)
    results["p0"] = p0
    
    results["p0_torr"] = p0 * 750.061683
    f = 1 - p0 / ppm
    if f < 0:
        results["err"] = (
            "ERROR: ionization fraction is negative! Choose a pressure higher than p0."
        )
        return results

    # Derivative of Sellmeier expression (for GVM calculation):
    #  % syms lam
    #  % sell = (1/2)*(T0/T)*(B1*lam.^2./(lam.^2-C1) + B2*lam.^2./(lam.^2-C2));
    #  % dsell_dlam = diff(sell,lam,1);
    #  % lam = lamL;
    #  % dsell_dlam = double(subs(dsell_dlam));
    dsell_dlam = 0

    # Calculate cutoff peak intensity [W/m^2] for desired harmonic:
    Cint = 3.17 * (9.33e-6)
    I_cut = (eq - 1.32 * Ip) / (Cint * lamL**2)

    # Calculate cutoff energy [mJ] using cutoff peak intensity:
    E_cut = 0.901 * I_cut * R**2 * tau * 1e3
    results["E_cut"] = E_cut


    t = np.linspace(-LT / 2, LT / 2, NT)  # vector of time points [s]

    # Initialize temporal amplitude profile:
   

    # Calculate critical ionization using difference:
    eta_cr = 1.0 / (1 + lamL**2 * re * Nbar / (2 * np.pi * (n0 - nH)))
    results["eta_cr"] = eta_cr * 100 
    
    A = np.exp(-2 * np.log(2) * (t / tau) ** 2)

    # Interpolate to find appropriate intensity:
    I_cr, _, results["peak-ionization-graph"] = interpEta(
        gas, A, wL, t, f, eta_cr, I_cut, NT
    )

    # E_cut = 1.32 * Ip + Cint * lamL**2 * I0

    # Calculate cutoff peak energy [mj] for desired harmonic:
    E_cr = 0.901 * I_cr * R**2 * tau * 1e3
    results["E_cr"] = E_cr

    if I_cut > I_cr:
        gas_err = "."
        if gas == "Ar":
            gas_err = "or using Ne/He."
        elif gas == "Ne":
            gas_err = "or using He."

        results["err"] = (
            """ERROR: Phase matching is not possible, as the intensity required to reach cutoff at the pulse peak results
            in an ionization above critical ionization. Either lower the cutoff intensity by increasing the laser wavelength or decreasing the target harmonic energy, or raise the critical intensity by decreasing pulse duration """
            + gas_err
        )
        return results

    # Normalize the envelope using the intensity and then calculate
    # the electric field:
    
    A = np.sqrt(2 * I_cr / c / eps0) * (A / max(A))
 
    E = A * np.cos(wL * t)


    # Calculate ionization rate and ionization fraction at peak:
    
  
    _,W_ADK = ADK_mod(E,gas);


   
    eta = 1-np.exp(-cumulative_trapezoid(W_ADK,t));
    
    tfs = t * 1e15
    
    # Create figure:
    fig = go.Figure()
    
    # Plot Electric Field:
    fig.add_trace(go.Scatter(x=tfs, y=E, mode='lines', line=dict(color='red', width=1), name='Electric Field'))
    
    # Plot Envelope:
    fig.add_trace(go.Scatter(x=tfs, y=A, mode='lines', line=dict(color='blue', dash='dash', width=2), name='Envelope'))
    fig.add_trace(go.Scatter(x=tfs, y=-A, mode='lines', line=dict(color='blue', dash='dash', width=2), showlegend=False))
    
    # Plot Ionization Fraction on secondary y-axis:
    fig.add_trace(go.Scatter(x=tfs, y=eta * 100, mode='lines', line=dict(color='magenta', width=3), name='Ion. Fraction', yaxis='y2'))
    
    # Update layout with dual y-axes:
    fig.update_layout(
        title="Field and Ionization Fraction",
        template=plot_template,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="Time Delay [fs]"),
        yaxis=dict(title="Field [N/C]", side="left", showgrid=False),
        yaxis2=dict(title="Ion. Fraction [%]", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.5)")
    )
    results["field-ionization-graph"] = fig

    #  Calculate fraction of critical ionization that we are phase
    #   matching (for ppm=p0, f=0; for ppm->inf, f->1; for ppm<p0,
    #        throw error)

    _, L_abs = calculate_form_factors(gas, ppm * Nbar, ppm, lamH)
    results["L_abs"] = L_abs
    L_f = 6 * L_abs
    results["L_f"] = L_f
    
    # Gaussian beam waist [µm] that optimizes coupling efficiency:
    w0 = 0.7 * R
    results["w0"] = w0 * 1e6

    # Confocal parameter of beam focused into capillary [mm]:
    b = kL * w0**2
    results["b"] = b * 1000

    
    nc = 1.45# refractive index of silica cladding
    
    nuv = (1 / 2) * (nH**2 + 1) / np.sqrt(nc**2 - 1)
    
    alpha1 = (um / 2 / np.pi) ** 2 * (lamL**2 / R**3) * nuv
    
    L = 1 / (2 * alpha1)
    

    #convert to meters
    L_f = L_f / 1000
    
    
    P_tran = np.exp(-L_f / L)
    results["P_tran"] = P_tran *100
    if P_tran < 0.9:
        results["err"] = """Transmission is below threshold (90%)."""

    

    

    return results


def gas_constants(gas):
    """
    Retrieves ionization potentials and frequency-dependent index
    of refraction for the specified noble gas.
    """
    constants = {
        "N2": {
            "Ip": 15.58,
            "ng2_0": 6.7e-19,
            "B1": 39209.95e-8,
            "B2": 18806.48e-8,
            "C1": 1e-12 * 1146.24e-6,
            "C2": 1e-12 * 13.476e-3,
        },
        "He": {
            "Ip": 24.5874,
            "ng2_0": 3.1e-21,
            "B1": 4977.77e-8,
            "B2": 1856.94e-8,
            "C1": 1e-12 * 28.54e-6,
            "C2": 1e-12 * 7.760e-3,
        },
        "Ne": {
            "Ip": 21.5645,
            "ng2_0": 8.7e-21,
            "B1": 9154.48e-8,
            "B2": 4018.63e-8,
            "C1": 1e-12 * 656.97e-6,
            "C2": 1e-12 * 5.728e-3,
        },
        "Ar": {
            "Ip": 15.7596,
            "ng2_0": 9.7e-20,
            "B1": 20332.29e-8,
            "B2": 34458.31e-8,
            "C1": 1e-12 * 206.12e-6,
            "C2": 1e-12 * 8.066e-3,
        },
        "Kr": {
            "Ip": 13.9996,
            "ng2_0": 2.2e-19,
            "B1": 26102.88e-8,
            "B2": 56946.82e-8,
            "C1": 1e-12 * 2.01e-6,
            "C2": 1e-12 * 10.043e-3,
        },
        "Xe": {
            "Ip": 12.1298,
            "ng2_0": 5.8e-18,
            "B1": 103701.61e-8,
            "B2": 31228.61e-8,
            "C1": 1e-12 * 12750e-6,
            "C2": 1e-12 * 0.561e-3,
        },
    }

    gas_data = constants[gas]
    ng2_0 = gas_data.get("ng2_0", 0) * 1e-4  # Convert to SI units [m^2/W]

    return (
        gas_data.get("Ip"),
        ng2_0,
        gas_data.get("B1", 0),
        gas_data.get("B2", 0),
        gas_data.get("C1", 0),
        gas_data.get("C2", 0),
    )


import numpy as np


def pressure_to_density(p, T):
    """
    Converts pressure [atm] to density [1/m^3] using the ideal gas law.

    Parameters:
    p (float or np.array): Pressure in atmospheres.
    T (float or np.array): Temperature in Kelvin.

    Returns:
    """
    kB = 1.380649e-23  # Boltzmann constant [J/K]

    # Convert pressures to Pascals
    p = p * 101325

    # Calculate densities using the ideal gas law
    Ng = p / (kB * T)

    return Ng


def calculate_form_factors(gas, Ng, p, lamH):
    hc = 1240e-9  # [eV m]
    re = 2.8179e-15  # classical electron radius [m]
    c = 2.99792e8  # speed of light [m/s]

    # Load data
    data = pd.read_csv(f"./data/{gas.lower()}.txt", sep=r"\s+").values

    if gas == "N2":
        energies_c, delta_c, beta_c = data[:, 0], data[:, 1], data[:, 2]
        lambdas_c = hc / energies_c

        # Check bounds
        # if np.any(lamH > np.max(lambdas_c)) or np.any(lamH < np.min(lambdas_c)):
        #     raise ValueError("Harmonic wavelength out of bounds!")

        # Interpolation
        lambdas = np.linspace(np.min(lambdas_c), np.max(lambdas_c), 10000)
        delta_vec = PchipInterpolator(lambdas_c, delta_c)(lambdas)
        beta_vec = PchipInterpolator(lambdas_c, beta_c)(lambdas)

        # Compute index of refraction and absorption length
        n_vec = 1 - p * delta_vec
        k = 2 * np.pi / lambdas
        alpha_vec = 2 * k * p * beta_vec
        Labs_vec = 1000 / alpha_vec
    else:
        energy_c, f1, f2 = data[:, 0], data[:, 1], data[:, 2]
        valid_idx = f1 != -9999
        energy_c, f1, f2 = energy_c[valid_idx], f1[valid_idx], f2[valid_idx]
        lambdas_c = hc / energy_c

        # Check bounds
        # if np.any(lamH > np.max(lambdas_c)) or np.any(lamH < np.min(lambdas_c)):
        #     raise ValueError("Harmonic wavelength out of bounds!")

        # Interpolation

        lambdas_c = np.flip(lambdas_c)
        f1 = np.flip(f1)
        f2 = np.flip(f2)
        lambdas = np.linspace(np.min(lambdas_c), np.max(lambdas_c), 10000)

        f1_interp = PchipInterpolator(lambdas_c, f1)(lambdas)
        f2_interp = PchipInterpolator(lambdas_c, f2)(lambdas)

        ff = f1_interp - 1j * f2_interp
        nc_vec = 1 - (Ng * re * lambdas**2 / (2 * np.pi)) * ff
        n_vec = np.real(nc_vec)

        k = 2 * np.pi / lambdas
        alpha_vec = 2 * k * np.imag(nc_vec)
        Labs_vec = 1000 / alpha_vec

    # Interpolate results
    nH = np.interp(lamH, lambdas, n_vec)
    Labs = np.interp(lamH, lambdas, Labs_vec)

    # Compute group velocity
    # dlam = np.gradient(lambdas)
    # dn_dlam = np.gradient(n_vec, dlam)
    # vgH_list = c / (n_vec - lambdas * dn_dlam)
    # vgH_vec = np.interp(lamH, lambdas, vgH_list)

    return nH, Labs


def ADK_mod(E, gas):
    # ADK Calculates the ionization rate based on ADK theory (see Dimitar Popmintchev's thesis)

    # Ionization potential and parameters based on the gas type
    if gas == "N2":
        Ipev = 15.58  # ionization potential (eV)

        # Data taken from Tong et al. (2002)
        m = 0
        c0, c2, c4 = 2.02, 0.78, 0.04  # coefficients
        Bm2 = (c0 * np.sqrt(1 / 2) + c2 * np.sqrt(5 / 2) + c4 * np.sqrt(9 / 2)) ** 2
    elif gas == "He":
        # Electron configuration: 1s2
        Ipev = 24.587  # ionization potential (eV)
        l, m, alpha = (
            0,
            0,
            7.0,
        )  # orbital quantum number, magnetic quantum number, alpha
    elif gas == "Ne":
        # Electron configuration: 1s2 2s2 p6
        Ipev = 21.564
        l, m, alpha = (
            1,
            0,
            9.0,
        )  # orbital quantum number, magnetic quantum number, alpha
    elif gas == "Ar":
        # Electron configuration: 1s2 2s2 p6 3s2 p6
        Ipev = 15.759
        l, m, alpha = (
            1,
            0,
            9.0,
        )  # orbital quantum number, magnetic quantum number, alpha

    # Constants:
    hb = 1.0546e-34  # reduced Planck constant [J s]
    qe = 1.6022e-19  # electron charge [C]
    me = 9.1094e-31  # mass of electron [kg]

    Ipev *= qe  # Convert ionization potential to Joules
    Z = 1  # Effective charge residue
    Iph = 13.6 * qe  # Ionization energy of H [J]
    wp = Ipev / hb  # Transition frequency [rad/s]
    wt = np.finfo(float).eps + qe * np.abs(E) / np.sqrt(
        2 * me * Ipev
    )  # Tunneling frequency [rad/s]


    
    n = Z * np.sqrt(Iph / Ipev)  # Effective principal quantum number
    ln = n - 1

    if gas == "N2":
        # Calculate constants:
        Glm = 1 / (2 ** abs(m) * factorial(abs(m)))

        # Calculate ADK rate:
        W_ADK = (
            wp
            * Bm2
            * Glm
            * (4 * wp / wt) ** (2 * n - abs(m) - 1)
            * np.exp(-4 * wp / (3 * wt))
        )
        W_ADK_2 = W_ADK
    else:
        # Calculate constants:
        # Cnl2 = 2^(2*n)/(n*gamma(n+1)*gamma(n));
        Cnl2 = 2 ** (2 * n) / (n * gamma(n + 1 + ln) * gamma(n - ln))

        Glm = (
            (2 * l + 1)
            * factorial(l + abs(m))
            / (2 ** abs(m) * factorial(abs(m)) * factorial(l - abs(m)))
        )

        # Calculate ADK rate:
        W_ADK = (
            wp
            * Cnl2
            * Glm
            * (4 * wp / wt) ** (2 * n - abs(m) - 1)
            * np.exp(-4 * wp / (3 * wt))
        )
        

        # Correct near barrier-ionization (Tong & Lin 2005):
        W_ADK_2 = W_ADK * np.exp(-alpha * (Z**2 * Iph / Ipev) * (wt / wp))

    return W_ADK, W_ADK_2


def interpEta(gas, A, wL, t, f, eta_cr, I_cut, NT):
    # INTERPETA Interpolates a peak ionization fraction vs intensity plot to
    # find the right intensity to reach f * etaCR.
    # Constants:
    c = 2.99792e8  # speed of light [m/s]
    eps0 = 8.85e-12  # vacuum permittivity [F/m]
    # Intensity list [W/m^2]:
    I_list = np.linspace(5e13, 5e15, 1000) * 1e4
    # Initialize field:
    A = np.sqrt(2 * I_list[:, None] / c / eps0) * (A / np.max(A))
    E = A * np.cos(wL * t)
    # Calculate ionization rate and ionization fraction at peaks:
    _, W_ADK = ADK_mod(E, gas)
    eta1 = 1 - np.exp(-cumulative_trapezoid(W_ADK, t, axis=1))
    eta_peak = eta1[:, NT // 2]
    # Cut out repeats:
    unique_eta_peak, unique_indices = np.unique(eta_peak, return_index=True)
    # Interpolate to find intensities:
    interp_I = interp1d(
        unique_eta_peak, I_list[unique_indices], fill_value="extrapolate"
    )
    # I0 = interp_I(f * eta_cr)
    I_cr = interp_I(eta_cr)
    # I0 = interp_I(f*eta_cr)
    interp_eta = interp1d(I_list, eta_peak, fill_value="extrapolate")
    eta_cut = interp_eta(I_cut)
    # Convert W/m^2 to 10^14 W/cm^2:
    # I0p = I0 * 1e-4 * 1e-14
    I_cr = I_cr * 1e-4 * 1e-14
    I_cut = I_cut * 1e-4 * 1e-14
    I_listp = I_list * 1e-4 * 1e-14

    # scale eta
    eta_peak = eta_peak * 100
    eta_cr = eta_cr * 100
    eta_cut = eta_cut * 100

    # Create plot:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=I_listp,
            y=eta_peak,
            mode="lines",
            line=dict(color="white", width=3),
            name="Peak Ionization",
        )
    )

    # Add vertical lines:
    # fig.add_vline(
    #     x=I0p,
    #     line=dict(color="blue", dash="dash"),
    #     annotation_text="$I_L$",
    #     annotation_position="top left",
    # )
    fig.add_vline(
        x=I_cut,
        line=dict(color="red"),
        annotation_text="$I_{cut}$",
        annotation_position="top left",
    )
    fig.add_vline(
        x=I_cr,
        line=dict(color="red"),
        annotation_text="$I_{CR}$",
        annotation_position="top left",
    )

    # Add horizontal lines:
    fig.add_hline(y=eta_cut, line=dict(color="red"), annotation_text=r"$\eta_{cut}$")
    # fig.add_hline(
    #     y=f * eta_cr * 100,
    #     line=dict(color="blue", dash="dash"),
    #     annotation_text=f"{f}$\\eta_{{CR}}$",
    # )
    fig.add_hline(y=eta_cr, line=dict(color="red"), annotation_text=r"$\eta_{CR}$")

    # Add marker for I0:
    # fig.add_trace(
    #     go.Scatter(
    #         x=[I0p],
    #         y=[f * eta_cr * 100],
    #         mode="markers",
    #         marker=dict(color="blue", size=10),
    #         name="I0 Point",
    #     )
    # )

    # Adjust x-axis limits:
    l_bound = min(I_cr, I_cut)
    r_bound = max(I_cr, I_cut)
    dI = 0.25 * abs(r_bound - l_bound)
    fig.update_xaxes(
        title_text=r"$\text{Peak Intensity} [10^{14} W/\text{cm}^2]$",
        range=[l_bound - dI, r_bound + dI],
    )

    l_bound = min(eta_cr, eta_cut)
    r_bound = max(eta_cr, eta_cut)
    dI = 0.25 * abs(r_bound - l_bound)
    fig.update_yaxes(
        title_text="Peak Ionization [%]", range=[l_bound - dI, r_bound + dI]
    )
    fig.update_layout(
        title="Peak Ionization vs. Intensity",
        template=plot_template,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # return I0, I_max, eta_min, fig and remove conversion
    return I_cr / (1e-4 * 1e-14), eta_cut / 100, fig
