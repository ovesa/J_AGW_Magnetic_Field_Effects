import numpy as np

gamma = 5.0 / 3.0  # adiabatic exponent
grav = 0.274  # km/s^2


def surface_fmodes(g, kh):
    """Computes the dispersion for the f-mode (surface gravity waves).

    Input:
    g = gravity on the Sun [km/s^2]
    kh = horizontal wavenumber [1/km]

    Output:
    omega = frequency of surface gravity waves [Hz]

    Note: important to convert from Hz to mHz when plotting
    Note 2: important to note if plotting in omega or nu (factor of (2*pi)^-1)
    """

    omega = np.sqrt(g * kh)
    return omega


def lamb_frequency(cs, kh):
    """Computes the isothermal dispersion relation for Lamb waves.

    Input:
    cs = sound speed of the Sun as a function of height [(]km/s]
    kh = horizontal wavenumber [1/km]

    Output:
    omega = frequency of isothermal Lamb waves [Hz]

    Note: important to convert from Hz to mHz when plotting
    Note 2: important to note if plotting in omega or nu (factor of (2*pi)^-1)
    """

    omega = cs * kh
    return omega


def acoustic_cutoff_frequency(cs, density_scale_height):
    """Computes the isothermal acoustic cut-off frquency for acoustic waves. Below this
    frequency, acoustic waves cannot escape. The derivative of the
    density scale height with respect to height is ignored.

    Input:
    cs = sound speed of the Sun as a function of height [cm/s]
    density_scale_height = density scale height [cm]

    Output:
    wac_squared =  acoustic cut-off frequency [Hz^2]

    Note: important to convert from Hz to mHz when plotting
    Note 2: important to note if plotting in omega or nu (factor of (2*pi)^-1)
    """

    wac_squared = (cs**2) / (4 * density_scale_height**2)
    return wac_squared


def densityscaleheight(drho_dz, density):
    """Computes the density scale height on the Sun.

    Input:
    drho_dz = change in density with change in height
    rho = solar density as a function of height [g/cm^3]

    Output:
    H_rho = density scale height [cm]
    """

    H_rho = -1 * (drho_dz / density)
    H_rho = H_rho ** (-1)
    return H_rho


def Brunt_Vaisala_frequency_Squared(g, dP_dz, cs, density, drho_dz):
    """Computes the Brunt-Vaisala frequency. Above this frequency,
    atmospheric gravity waves cannot propagate.

    Input:
    g = gravity on the Sun [cm/s^2]
    dP_dz = change in pressure with change in height
    cs = sound speed of the Sun as a function of height [cm/s]
    density = solar density as a function of height [g/cm^3]
    drho_dz = change in density with change in height

    Output:
    N_squared = Brunt-Vaisala frequency [Hz^2]

    Note: important to convert from Hz to mHz when plotting
    Note 2: important to note if plotting in omega or nu (factor of (2*pi)^-1)
    """

    first_term = dP_dz / ((cs**2) * density)
    second_term = drho_dz / density
    N_squared = g * (first_term - second_term)
    return N_squared


def quadraticforumala(a, b, c):
    """Solve the quadratic formula.

    Input:
    a,b,c = coefficients

    Output:
    sol1 = solution 1
    sol2 = solution 2
    """

    sol1 = (-b + np.sqrt((b**2 - 4 * a * c))) / (2 * a)
    sol2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    return sol1, sol2


def isothermal_Brunt_vaisala(cs):
    """This is the Brunt-Vaisala (buoyancy) Frequency that describes the maximum frequency
    that gravity waves can attain [Hz].

    Input:
    cs = adiabatic sound speed [km/s]

    Output:
    Brunt-Vaisala (buoyancy) Frequency [Hz]
    """

    grav = 0.274  # grravity [km/s^2]
    gamma = 5.0 / 3.0  # adiabatic exponent

    return (np.sqrt((gamma - 1)) * grav) / cs


def isthermal_wac(cs):
    """
    Isothermal acoustic cut-off frequency.

    Arguments:
        cs -- sound speed [km/s]

    Returns:
        Isothermal acoustic cut-off frequency.
    """

    grav = 0.274  # gravity [km/s^2]
    gamma = 5.0 / 3.0  # adiabatic exponent
    return (gamma * grav) / (2 * cs)
